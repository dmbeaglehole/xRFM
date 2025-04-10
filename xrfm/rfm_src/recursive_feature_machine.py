from .eigenpro import KernelModel
    
import torch, numpy as np
from torchmetrics.functional.classification import accuracy
from .kernels import Kernel, LaplaceKernel, ProductLaplaceKernel, SumPowerLaplaceKernel
from tqdm.contrib import tenumerate
from .utils import matrix_power, SmoothClampedReLU
from sklearn.metrics import roc_auc_score
import time
from typing import Union

class RFM(torch.nn.Module):
    """
    Main object for RFMs with sklearn style interface. Subclasses must implement the kernel and update_M methods. 
    The subclasses may be either specific kernels (Laplace, Gaussian, GeneralizedLaplace, etc.), in which case the kernel method is automatically derived,
    or generic kernels (GenericKernel), in which case a Kernel object must be provided. I.e. one can either define:
    ```python
        from rfm import RFM
        model = RFM(kernel=LaplaceKernel(bandwidth=1, exponent=1.2), device='cpu', reg=1e-3, iters=3, bandwidth_mode='constant')
    ```
    """

    def __init__(self, kernel: Union[Kernel, str], agop_power=0.5, device=None, diag=False, reg=1e-3, verbose=True,
                 iters=4, bandwidth=10., exponent=1., centering=False, bandwidth_mode='constant', mem_gb=None,
                 classification=False, M_batch_size=None, tuning_metric='mse', categorical_info=None, early_stop_rfm=True, 
                 early_stop_multiplier=1.1):
        """
        :param device: device to run the model on
        :param diag: if True, Mahalanobis matrix M will be diagonal
        :param centering: if True, update_M will center the gradients before taking an outer product
        :param bandwidth_mode: 'constant' or 'adaptive'
        :param mem_gb: memory in GB for AGOP/EigenPro
        :param numerical_indices: torch.Tensor(n_num,)
        :param categorical_indices: List of torch.Tensor(d_cat_i,) for each categorical feature
        :param categorical_vectors: List of torch.Tensor(d_cat_i, d_cat_i) for each categorical feature. Each row is the encoding for that index.
        """
        super().__init__()
        if isinstance(kernel, str):
            kernel = self.kernel_from_str(kernel, bandwidth=bandwidth, exponent=exponent)
        self.kernel_obj = kernel
        self.agop_power = agop_power
        self.M = None
        self.sqrtM = None
        self.reg = reg
        self.iters = iters
        self.diag = diag # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering # if True, update_M will center the gradients before taking an outer product
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agop_power = 0.5 # power for root of agop
        self.max_lstsq_size = 70_000 # max number of points to use for direct solve
        self.bandwidth_mode = bandwidth_mode
        self.proba_beta = 500
        self.M_batch_size = M_batch_size
        self.verbose = verbose
        self.classification = classification
        self.tuning_metric = tuning_metric
        self.early_stop_rfm = early_stop_rfm
        self.early_stop_multiplier = early_stop_multiplier


        print(f"Early stop multiplier: {self.early_stop_multiplier}")
        
        if categorical_info is not None: 
            if isinstance(self.kernel_obj, ProductLaplaceKernel):
                self.set_categorical_indices(**categorical_info)
            else:
                print("Ignoring categorical indices for non-ProductLaplaceKernel.")

        if mem_gb is not None:
            self.mem_gb = mem_gb
        elif torch.cuda.is_available():
            # find GPU memory in GB, keeping aside 1GB for safety
            self.mem_gb = torch.cuda.get_device_properties(self.device).total_memory//1024**3 - 1 
        else:
            self.mem_gb = 8
        
    def kernel(self, x, z):
        return self.kernel_obj.get_kernel_matrix(x, z, self.sqrtM)

    def kernel_from_str(self, kernel_str, bandwidth, exponent):
        if kernel_str in ['laplace', 'l2']:
            return LaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ['product_laplace', 'l1']:
            return ProductLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ['sum_power_laplace', 'l1_power']:
            return SumPowerLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        else:
            raise ValueError(f"Invalid kernel: {kernel_str}")
        
    def update_M(self, samples):
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)

        if self.M is None:
            if self.diag:
                self.M = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.M = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        if self.sqrtM is None:
            if self.diag:
                self.sqrtM = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.sqrtM = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        agop_func = self.kernel_obj.get_agop_diag if self.diag else self.kernel_obj.get_agop
        agop = agop_func(x=self.centers, z=samples, coefs=self.weights.t(), mat=self.sqrtM, center_grads=self.centering)
        return agop
    
    def reset_adaptive_bandwidth(self):
        self.kernel_obj._reset_adaptive_bandwidth()
        return 

    def tensor_copy(self, tensor):
        """
        Create a CPU copy of a tensor.
        :param tensor: Tensor to copy.
        :param keep_device: If True, the device of the original tensor is kept.
        :return: CPU copy of the tensor.
        """
        if tensor is None:
            return None
        elif self.keep_device or tensor.device.type == 'cpu':
            return tensor.clone()
        else:
            return tensor.cpu()
        
    def set_categorical_indices(self, numerical_indices, categorical_indices, categorical_vectors, device=None):
        """
        :param numerical_indices: torch.Tensor(n_num,)
        :param categorical_indices: List of torch.Tensor(d_cat_i,) for each categorical feature
        :param categorical_vectors: List of torch.Tensor(d_cat_i, d_cat_i) for each categorical feature. Each row is the encoding for that index.
        """
        if numerical_indices is None and categorical_indices is None and categorical_vectors is None:
            if self.verbose:
                print("No categorical indices provided, ignoring")
            return
        assert numerical_indices is not None, "Numerical indices must be provided if one of categorical indices/vectors are provided"
        assert categorical_vectors is not None, "Categorical vectors must be provided if categorical indices are provided"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        self.kernel_obj.set_categorical_indices(numerical_indices, categorical_indices, categorical_vectors, device=self.device if device is None else device)
        return

    def update_best_params(self, best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, current_metric, current_iter):
        # if classification and accuracy higher, or if regression and mse lower
        if self.tuning_metric in ['accuracy', 'auc'] and current_metric > best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        elif self.tuning_metric == 'mse' and current_metric < best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        return best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth
        
    def fit_predictor(self, centers, targets, bs=None, lr_scale=1, solver='solve', **kwargs):
        
        if self.bandwidth_mode == 'adaptive':
            # adaptive bandwidth will be reset on next kernel computation
            print("Resetting adaptive bandwidth")
            self.reset_adaptive_bandwidth()

        self.centers = centers

        if self.fit_using_eigenpro:
            if self.prefit_eigenpro:
                random_indices = torch.randperm(centers.shape[0])[:self.max_lstsq_size]
                if self.verbose:
                    print(f"Prefitting Eigenpro with {len(random_indices)} points")
                sub_weights = self.fit_predictor_lstsq(centers[random_indices], targets[random_indices], solver=solver)
                initial_weights = torch.zeros_like(targets)
                initial_weights[random_indices] = sub_weights.to(targets.device, dtype=targets.dtype)
            else:
                initial_weights = None

            self.weights = self.fit_predictor_eigenpro(centers, targets, bs=bs, lr_scale=lr_scale, 
                                                       initial_weights=initial_weights, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets, solver=solver)

    def fit_predictor_lstsq(self, centers, targets, solver='solve'):
        assert(len(centers)==len(targets))

        if centers.device != self.device:
            centers = centers.to(self.device)
            targets = targets.to(self.device)

        kernel_matrix = self.kernel(centers, centers)    

        if self.reg > 0:
            kernel_matrix.diagonal().add_(self.reg)
        
        
        if solver == 'solve':
            out = torch.linalg.solve(kernel_matrix, targets)
        elif solver == 'cholesky':
            L = torch.linalg.cholesky(kernel_matrix, out=kernel_matrix)
            out = torch.cholesky_solve(targets, L)
        elif solver == 'lu':
            P, L, U = torch.linalg.lu(kernel_matrix)
            out = torch.linalg.lu_solve(P, L, U, targets)
        else:
            raise ValueError(f"Invalid solver: {solver}")
        
        return out

    def fit_predictor_eigenpro(self, centers, targets, bs, lr_scale, initial_weights=None, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        ep_model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        if initial_weights is not None:
            ep_model.weight = initial_weights.to(ep_model.weight.device, dtype=ep_model.weight.dtype)
        _ = ep_model.fit(centers, targets, verbose=self.verbose, mem_gb=self.mem_gb, bs=bs, 
                         lr_scale=lr_scale, classification=self.classification, **kwargs)
        return ep_model.weight.clone()

    def predict(self, samples, max_batch_size=50_000):
        samples, original_format = self.validate_samples(samples)
        out = []
        for i in range(0, samples.shape[0], max_batch_size):
            out_batch = self.kernel(samples[i:i+max_batch_size].to(self.device), self.centers.to(self.device)) @ self.weights.to(self.device)
            out.append(out_batch)
        return self.convert_to_format(torch.cat(out, dim=0), original_format)

    def validate_samples(self, samples):
        original_format = {}
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
            original_format['type'] = 'numpy'
            original_format['device'] = 'cpu'
        elif isinstance(samples, torch.Tensor):
            original_format['type'] = 'torch'
            original_format['device'] = samples.device
        else:
            raise ValueError(f"Invalid sample type: {type(samples)}")
        return samples.to(self.device), original_format
    
    def convert_to_format(self, tensor, original_format):
        if original_format['type'] == 'numpy':
            return tensor.cpu().numpy()
        elif original_format['type'] == 'torch':
            return tensor.to(original_format['device'])

    def validate_data(self, train_data, val_data):
        assert train_data is not None, "Train data must be provided"
        assert val_data is not None, "Validation data must be provided"

        X_train, y_train = train_data
        X_val, y_val = val_data

        X_train, _ = self.validate_samples(X_train)
        X_val, _ = self.validate_samples(X_val)
        y_train, _ = self.validate_samples(y_train)
        y_val, _ = self.validate_samples(y_val)

        if len(y_val.shape) == 1:
            y_val = y_val.unsqueeze(-1)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(-1)

        return X_train, y_train, X_val, y_val
    
    def adapt_params_to_data(self, n, d):

        if self.tuning_metric == 'accuracy' and self.early_stop_rfm:
            if n <= 30_000:
                self.early_stop_multiplier = min(self.early_stop_multiplier, 1.003)
            else:
                self.early_stop_multiplier = min(self.early_stop_multiplier, 1.006)
            print(f"More aggressive early stop multiplier for accuracy: {self.early_stop_multiplier}")
            

        self.keep_device = d > n # keep previous Ms on GPU if more features than samples
        ep_epochs = 8
        total_points_to_sample = 20_000
        iters_to_use = 4
        if isinstance(self.kernel_obj, ProductLaplaceKernel):
            ep_epochs = 2
            if n > 1000: # only handle cateogricals specially for high-dimensional data
                if n <= 10_000:
                    # For smallest datasets: use default values
                    pass
                elif 10_000 < n <= 20_000 and d <= 2000:
                    # Medium-small datasets with moderate dimensionality
                    total_points_to_sample = min(total_points_to_sample, 10_000)
                    iters_to_use = min(iters_to_use, 4)
                elif 20_000 < n <= 50_000 and d <= 2000:
                    # Medium-sized datasets with moderate dimensionality
                    total_points_to_sample = min(total_points_to_sample, 2500)
                    iters_to_use = min(iters_to_use, 2)
                elif 10_000 < n <= 20_000 and d <= 3000:
                    # Medium-small datasets with higher dimensionality
                    total_points_to_sample = 2500
                    iters_to_use = min(iters_to_use, 2)
                elif d < 1000:
                    # Largest datasets or highest dimensionality
                    total_points_to_sample = 2000
                    iters_to_use = min(iters_to_use, 1)
                elif d < 4000:
                    # Largest datasets or highest dimensionality
                    total_points_to_sample = 1000
                    iters_to_use = min(iters_to_use, 1)
                else:
                    # For highest dimensionality
                    total_points_to_sample = 250
                    iters_to_use = min(iters_to_use, 1)
        if n >= 70_000:
            # for large datasets, use fewer iterations for all kernel types
            iters_to_use = min(iters_to_use, 2)

        ep_epochs = ep_epochs if self.ep_epochs is None else self.ep_epochs
        total_points_to_sample = total_points_to_sample if self.total_points_to_sample is None else self.total_points_to_sample
        iters_to_use = iters_to_use if self.iters is None else self.iters

        self.iters = iters_to_use
        self.total_points_to_sample = total_points_to_sample
        self.ep_epochs = ep_epochs
        return
    
    def fit(self, train_data, val_data=None, iters=None, method='lstsq', reg=None,
            verbose=None, M_batch_size=None, ep_epochs=None, return_best_params=True, bs=None, 
            return_Ms=False, lr_scale=1, total_points_to_sample=None, solver='solve', fit_last_M=False, 
            tuning_metric=None, prefit_eigenpro=True, **kwargs):
        """
        :param train_data: tuple of (X, y)
        :param val_data: tuple of (X, y)
        :param iters: number of iterations to run
        :param method: 'lstsq' or 'eigenpro'
        :param classification: if True, the model will tune for (and report) accuracy, else just MSE loss
        :param verbose: if True, print progress
        :param M_batch_size: batch size over samples for AGOP computation
        :param return_best_params: if True, return the best parameters
        :param bs: batch size for eigenpro
        :param return_Ms: if True, return the Mahalanobis matrix at each iteration
        :param lr_scale: learning rate scale for EigenPro
        :param total_points_to_sample: number of points to sample for AGOP computation
        :param solver: 'solve' or 'cholesky' or 'lu', used in LSTSQ computation
        :param fit_last_M: if True, fit the Mahalanobis matrix one last time after training
        :param prefit_eigenpro: if True, prefit EigenPro with a subset of <= max_lstsq_size samples
        """

        start_rfm_time = time.time()
        self.verbose = verbose if verbose is not None else self.verbose
        self.fit_using_eigenpro = (method.lower()=='eigenpro')
        self.prefit_eigenpro = prefit_eigenpro
        self.reg = reg if reg is not None else self.reg
        self.M_batch_size = M_batch_size
        self.total_points_to_sample = total_points_to_sample
        self.iters = iters if iters is not None else self.iters
        self.ep_epochs = ep_epochs
        self.tuning_metric = tuning_metric if tuning_metric is not None else self.tuning_metric
        self.minimize = self.tuning_metric in ['mse']

        X_train, y_train, X_val, y_val = self.validate_data(train_data, val_data)

        n, d = X_train.shape
        print("="*70)
        print(f"Fitting RFM with ntrain: {n}, d: {d}, and nval: {X_val.shape[0]}")
        print("="*70)

        self.adapt_params_to_data(n, d)
        
        metrics, Ms = [], []
        best_alphas, best_M, best_sqrtM = None, None, None
        best_metric = float('inf') if self.tuning_metric == 'mse' else 0
        best_iter = None
        early_stopped = False
        best_bandwidth = self.kernel_obj.bandwidth+0
        for i in range(self.iters):
            start = time.time()
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, 
                               bs=bs, lr_scale=lr_scale, solver=solver, 
                               **kwargs)
                        
            if self.tuning_metric == 'accuracy':
                val_metrics = self.score(X_val, y_val, metrics=['accuracy'])
                val_acc = val_metrics['accuracy']
                if self.verbose:
                    print(f"Round {i}, Val Acc: {100*val_acc:.2f}%")
            elif self.tuning_metric == 'auc':
                val_metrics = self.score(X_val, y_val, metrics=['auc'])
                val_auc = val_metrics['auc']
                if self.verbose:
                    print(f"Round {i}, Val AUC: {val_auc:.4f}")
            else:
                val_metrics = self.score(X_val, y_val, metrics=['mse'])
                val_mse = val_metrics['mse']
                if self.verbose:
                    print(f"Round {i}, Val MSE: {val_mse:.4f}")

            return_val = val_metrics[self.tuning_metric]

            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(best_metric, best_alphas, 
                                                                                                                best_M, best_sqrtM, 
                                                                                                                best_iter, best_bandwidth, 
                                                                                                                val_metrics[self.tuning_metric], i)
             
            if self.early_stop_rfm:
                val_metric = val_metrics[self.tuning_metric]
                if (self.minimize and val_metric > best_metric * self.early_stop_multiplier) or\
                    (not self.minimize and val_metric < best_metric / self.early_stop_multiplier):
                    print(f"Early stopping at iteration {i}")
                    if not return_best_params:
                        self.fit_M(X_train, y_train, **kwargs) # need to fit last M from final fit, to match default behavior

                    early_stopped = True
                    break

            self.fit_M(X_train, y_train, **kwargs)
            
            del self.weights
            
            if return_Ms:
                Ms.append(self.tensor_copy(self.M))
                metrics.append(val_metrics[self.tuning_metric])

            print(f"Time taken for round {i}: {time.time() - start} seconds")

        if not early_stopped: # handle final iteration if early stopping didn't occur
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, bs=bs, **kwargs)        
            if self.tuning_metric == 'accuracy':
                final_val_metrics = self.score(X_val, y_val, metrics=['accuracy'])
                final_val_acc = final_val_metrics['accuracy']
                if self.verbose:
                    print(f"Final Val Acc: {100*final_val_acc:.2f}")
            elif self.tuning_metric == 'auc':
                final_val_metrics = self.score(X_val, y_val, metrics=['auc'])
                final_val_auc = final_val_metrics['auc']
                if self.verbose:
                    print(f"Final Val AUC: {final_val_auc:.4f}")
            else:
                final_val_metrics = self.score(X_val, y_val, metrics=['mse'])
                final_val_mse = final_val_metrics['mse']
                if self.verbose:
                    print(f"Final Val MSE: {final_val_mse:.4f}")

            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(best_metric, best_alphas, best_M, 
                                                                                                                    best_sqrtM, best_iter, best_bandwidth, 
                                                                                                                    final_val_metrics[self.tuning_metric], 
                                                                                                                    iters)
                
            if fit_last_M: # fit last M from final fit
                self.fit_M(X_train, y_train, verbose=verbose, M_batch_size=M_batch_size, fit_last_M=fit_last_M, **kwargs)
                Ms.append(self.tensor_copy(self.M))

            return_val = final_val_metrics[self.tuning_metric]

        if return_best_params:
            self.M = None if best_M is None else best_M.to(self.device)
            self.sqrtM = None if best_sqrtM is None else best_sqrtM.to(self.device)
            self.weights = best_alphas.to(self.device)
            self.kernel_obj.bandwidth = best_bandwidth

        self.best_iter = best_iter

        if return_Ms and fit_last_M:
            self.agop_best_model = Ms[best_iter]

        if return_Ms:
            return Ms, metrics
        total_time = time.time() - start_rfm_time
        return return_val, total_time
    
    def _compute_optimal_M_batch(self, n, c, d, scalar_size=4, mem_constant=2., max_batch_size=10_000, max_cheap_batch_size=20_000):
        """Computes the optimal batch size for AGOP."""
        if self.device in ['cpu', torch.device('cpu')] or isinstance(self.kernel_obj, LaplaceKernel):
            # cpu and LaplaceKernel are less memory intensive, use a single batch
            M_batch_size = min(n, max_cheap_batch_size)
        else:
            total_memory_possible = torch.cuda.get_device_properties(self.device).total_memory
            curr_mem_use = torch.cuda.memory_allocated()
            available_memory = total_memory_possible - curr_mem_use
            M_batch_size = int(available_memory / (mem_constant*n*c*d*scalar_size))
            M_batch_size = min(M_batch_size, max_batch_size)
        print(f"Optimal M batch size: {M_batch_size}")
        return M_batch_size
    
    def fit_M(self, samples, labels, M_batch_size=None, **kwargs):
        """Applies AGOP to update the Mahalanobis matrix M."""
        
        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype, device=self.device) 
            if self.diag else torch.zeros(d, d, dtype=samples.dtype, device=self.device))
        

        if M_batch_size is None: 
            BYTES_PER_SCALAR = samples.element_size()
            c = labels.shape[-1]
            M_batch_size = self._compute_optimal_M_batch(n, c, d, scalar_size=BYTES_PER_SCALAR)
        
        batches = torch.arange(n).split(M_batch_size)

        num_batches = 1 + self.total_points_to_sample//M_batch_size
        batches = batches[:num_batches]
        if self.verbose:
            print(f'Sampling AGOP on maximum of {num_batches*M_batch_size} total points')

        if self.verbose:
            for i, bids in tenumerate(batches):
                M.add_(self.update_M(samples[bids]))
        else:
            for bids in batches:
                M.add_(self.update_M(samples[bids]))
        
        self.M = M / (M.max() + 1e-30)
        self.sqrtM = matrix_power(self.M, self.agop_power)
        del M

        
    def score(self, samples, targets, metrics):
        """
        samples: torch.Tensor of shape (n, d)
        targets: torch.Tensor of shape (n, c)
        metrics: list of metrics to compute
        """
        
        out_metrics = {}
        if 'accuracy' in metrics:
            preds = self.predict_proba(samples.to(self.device)).to(targets.device)
            if preds.shape[-1]==1:
                num_classes = len(torch.unique(targets))
                if num_classes==2:
                    preds = torch.where(preds > 0.5, 1, 0).reshape(targets.shape)
                    out_metrics['accuracy'] = accuracy(preds, targets, task="binary").item()
                else:
                    out_metrics['accuracy'] = accuracy(preds, targets, task="multiclass", num_classes=num_classes).item()
            else:
                preds_ = torch.argmax(preds,dim=-1)
                targets_ = torch.argmax(targets,dim=-1)
                out_metrics['accuracy'] = accuracy(preds_, targets_, task="multiclass", num_classes=preds.shape[-1]).item()
        
        if 'mse' in metrics:
            preds = self.predict(samples.to(self.device)).to(targets.device)
            out_metrics['mse'] = (targets - preds).pow(2).mean()

        if 'auc' in metrics:
            preds = self.predict_proba(samples.to(self.device))
            if preds.shape[-1]==1:
                num_classes = len(torch.unique(targets))
                if num_classes==2:
                    out_metrics['auc'] = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())
                else:
                    out_metrics['auc'] = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy(), multi_class='ovr')
            else:
                out_metrics['auc'] = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy(), multi_class='ovr')

        return out_metrics
    
    def predict_proba(self, samples, eps=1e-3):
        predictions = self.predict(samples)
        if predictions.shape[1] == 1:
            smooth_clamped = SmoothClampedReLU(beta=self.proba_beta)
            predictions = smooth_clamped(predictions)
        else:
            min_preds = predictions.min(dim=1, keepdim=True).values
            max_preds = predictions.max(dim=1, keepdim=True).values 
            predictions = (predictions - min_preds) / (max_preds - min_preds) # normalize predictions to [0, 1]
            predictions = torch.clamp(predictions, eps, 1-eps) # clamp predictions to [eps, 1-eps]
            predictions /= predictions.sum(dim=1, keepdim=True) # normalize predictions to sum to 1
        return predictions
