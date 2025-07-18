import torch
from xrfm.rfm_src import RFM, matrix_power
from torchmetrics.functional.classification import accuracy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import copy

from .tree_utils import get_param_tree

class xRFM:
    """
    Tree-based Recursive Feature Machine (RFM).
    
    This model recursively splits the training data using random projections 
    and fits the base RFM model on each subset once the subset size is small enough.
    
    Parameters
    ----------
    rfm_params : dict, default=None
        Parameters to pass to the RFM model at each leaf node.
        If None, default parameters are used.
    
    min_subset_size : int, default=60000
        The minimum size of a subset to further split. If a subset has fewer 
        samples than this, a base RFM model is fit on it directly.
    
    max_depth : int, default=None
        Maximum depth of the recursive splitting tree. If None, splitting continues
        until all subsets are smaller than min_subset_size.
    
    device : str, default=None
        Device to use for computation. If None, uses cuda if available, otherwise cpu.
        
    n_trees : int, default=1
        Number of trees to build. Predictions will be averaged across all trees.
        
    n_tree_iters : int, default=0
        Number of iterations to build each tree. Later iterations use the average
        of model.M from all leaf nodes to generate better projection directions.
        If n_tree_iters=0, the original random projection method is used.

    split_method : str, default='top_vector_agop_on_subset'
        Method to use for splitting the data.
        'top_vector_agop_on_subset' : use the top eigenvector of the AGOP on the subset
        'random_agop_on_subset' : use a random eigenvector of the AGOP on the subset
        'top_pc_agop_on_subset' : use the top principal component of the AGOP on the subset
        'random_pca' : use a random principal component of the data
        'linear' : use linear regression coefficients as projection direction
        'fixed_vector' : use a fixed vector for projection (requires fixed_vector parameter)

    tuning_metric : str, default='mse'
        Metric to use for tuning the model.
        'mse' : mean squared error
        'accuracy' : accuracy
        'auc' : area under the ROC curve
        
    categorical_info : dict, default=None
        Information about the categorical features.
        If None, it is assumed that there are no categorical features.
        If not None, it should be a dictionary with the following keys:
        'categorical_indices' : list of indices of the categorical features
        'categorical_vectors' : list of vectors of the categorical features
    
    default_rfm_params : dict, default=None
        Default parameters for the RFM model used for generating split directions
        when using AGOP-based split methods. If None, uses built-in default parameters
        with kernel='l2', exponent=1.0, bandwidth=10.0, etc.
    
    fixed_vector : torch.Tensor, default=None
        Fixed projection vector to use when split_method='fixed_vector'.
        Must be provided if using 'fixed_vector' split method.
    
    callback : function, default=None
        Callback function to call after each iteration of each Leaf RFM.
        The function must accept an 'iteration' argument.
    
    Notes
    -----
    The model follows sklearn's estimator interface with fit, predict, and score methods.
    """
    
    def __init__(self, rfm_params=None, min_subset_size=60_000,
                 max_depth=None, device=None, n_trees=1, n_tree_iters=0, 
                 split_method='top_vector_agop_on_subset', tuning_metric='mse', 
                 categorical_info=None, default_rfm_params=None,
                 fixed_vector=None, callback=None):
        self.min_subset_size = min_subset_size
        self.rfm_params = rfm_params
        self.max_depth = max_depth
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trees = None
        self.projections = None
        self.models = None
        self.n_trees = n_trees
        self.n_tree_iters = n_tree_iters
        self.tuning_metric = tuning_metric
        self.split_method = split_method
        self.maximizing_metric = tuning_metric in ['accuracy', 'auc', 'f1']
        self.categorical_info = categorical_info
        self.fixed_vector = fixed_vector
        self.callback = callback
        
        # parameters for refilling the validation set at leaves
        self.min_val_size = 1500
        self.val_size_frac = 0.2

        # default parameters for the split direction model
        print(default_rfm_params)
        if default_rfm_params is None:
            self.default_rfm_params = {
                'model': {
                    "kernel": 'l2',
                    "exponent": 1.0,
                    "bandwidth": 10.0,
                    "diag": False,
                    "bandwidth_mode": "constant"
                },
                'fit' : {
                    "get_agop_best_model": True,
                    "return_best_params": False,
                    "reg": 1e-3,
                    "iters": 0,
                    "early_stop_rfm": False,
                    "verbose": False
                }
            }
        else:
            self.default_rfm_params = default_rfm_params

        if self.rfm_params is None:
            self.rfm_params = self.default_rfm_params
            self.rfm_params['return_best_params'] = True
    
    def tree_copy(self, tree):
        """
        Deep copy a tree structure.
        
        Parameters
        ----------
        tree : dict
            Tree to copy
            
        Returns
        ------- 
        dict
            Copied tree structure
        """
        return copy.deepcopy(tree)
    
    
    def _generate_random_projection(self, dim):
        """
        Generate a random unit vector for data projection.
        
        Parameters
        ----------
        dim : int
            Dimension of the projection vector
            
        Returns
        -------
        torch.Tensor
            Random unit vector of shape (dim,)
        """
        projection = torch.randn(dim, device=self.device)
        return projection / torch.norm(projection)
    
    def _generate_projection_from_M(self, dim, M):
        """
        Generate a projection vector using the covariance matrix M.
        
        This method samples from a multivariate normal distribution with 
        covariance M (typically the AGOP matrix).
        
        Parameters
        ----------
        dim : int
            Dimension of the projection vector
        M : torch.Tensor
            Covariance matrix, either diagonal (1D) or full matrix (2D)
            
        Returns
        -------
        torch.Tensor
            Projection vector of shape (dim,), normalized to unit length
        """
        if M.dim() == 1:  # If M is diagonal
            std_devs = torch.sqrt(M)
            projection = torch.normal(0, std_devs).to(self.device)
        else:  # If M is a full matrix
            # Generate random vector from standard normal distribution
            z = torch.randn(dim, device=self.device)
            
            try:
                sqrtM = matrix_power(M, 0.5)

                # Transform z to get vector with covariance M
                projection = sqrtM @ z
            except:
                print(f"Matrix power failed, defaulting to random projection")

                # Fallback to random projection if matrix power fails
                projection = torch.randn(dim, device=self.device)
        
        # Normalize to unit vector
        return projection / torch.norm(projection)
    
    def _collect_leaf_nodes(self, node):
        """
        Recursively collect all leaf nodes in a tree.
        
        Parameters
        ----------
        node : dict
            Current tree node
            
        Returns
        -------
        list
            List of all TabRFM models at leaf nodes
        """
        if node['type'] == 'leaf':
            return [node]
        
        left_nodes = self._collect_leaf_nodes(node['left'])
        right_nodes = self._collect_leaf_nodes(node['right'])
        
        return left_nodes + right_nodes
    
    def _collect_attr(self, attr_name):
        """
        Collect a specific attribute from all leaf nodes in all trees.
        
        Parameters
        ----------
        attr_name : str
            Name of the attribute to collect from each leaf model
            
        Returns
        -------
        list
            List of attribute values from all leaf models across all trees
        """
        best_agops = []
        for t in self.trees:
            leaf_nodes = self._collect_leaf_nodes(t)
            best_agops += [getattr(node['model'], attr_name) for node in leaf_nodes]
        return best_agops
    
    def collect_best_agops(self):
        """
        Collect the best AGOP matrices from all leaf nodes across all trees.
        
        Returns
        -------
        list
            List of AGOP matrices from all leaf models
        """
        return self._collect_attr('agop_best_model')
        # best_agops = []
        # for t in self.trees:
        #     leaf_nodes = self._collect_leaf_nodes(t)
        #     best_agops += [node['model'].agop_best_model for node in leaf_nodes]
        # return best_agops

    def collect_Ms(self):
        """
        Collect the Mahalanobis matrices (M) from all leaf nodes across all trees.
        
        Returns
        -------
        list
            List of M matrices from all leaf models
        """
        return self._collect_attr('M')
    
    def _average_M_across_leaves(self, tree):
        """
        Average the M parameter across all leaf nodes in a tree.
        
        This method collects the Mahalanobis matrices from all leaf nodes
        and computes their average. This averaged matrix is used to generate
        better projection directions in subsequent iterations.
        
        Parameters
        ----------
        tree : dict
            Tree to analyze
            
        Returns
        -------
        torch.Tensor
            Averaged M parameter, either diagonal (1D) or full matrix (2D)
        """
        leaf_nodes = self._collect_leaf_nodes(tree)
        leaf_models = [node['model'] for node in leaf_nodes]

        # Collect M matrices from all leaf models
        M_matrices = []
        for model in leaf_models:
            if hasattr(model, 'M') and model.M is not None:
                M_matrices.append(model.M)
            else:
                identity = torch.ones(self.data_dim, device=self.device) if model.diag else torch.eye(self.data_dim, device=self.device) 
                M_matrices.append(identity)

        if M_matrices[0].dim() == 1:  # If M is diagonal
            avg_M = torch.stack(M_matrices).mean(dim=0)
        else:  # If M is a full matrix
            avg_M = torch.stack(M_matrices).mean(dim=0)
        
        return avg_M

    def _get_balanced_split(self, projections, train_median):
        """
        Get balanced left and right masks by assigning median points to the smaller split.
        
        Parameters
        ----------
        projections : torch.Tensor
            Projected values for all samples
        train_median : float
            Median value to split on
            
        Returns
        -------
        tuple
            (left_mask, right_mask) balanced masks
        """
        # Initial split
        left_mask = projections < train_median
        right_mask = projections > train_median
        median_mask = projections == train_median

        # Count elements in each split
        n_left, n_right = left_mask.sum(), right_mask.sum()
        
        # If one split is larger, assign median points to smaller split
        if n_left != n_right and median_mask.any():
            median_indices = torch.where(median_mask)[0]
            
            if n_left < n_right:
                # Add median points to left split
                n_to_add = min(median_indices.size(0), n_right - n_left)
                left_mask[median_indices[:n_to_add]] = True
            else:
                # Add median points to right split
                n_to_add = min(median_indices.size(0), n_left - n_right)
                right_mask[median_indices[:n_to_add]] = True
                
            # Update median mask to only include unused median points
            if n_to_add > 0:
                median_mask[median_indices[:n_to_add]] = False
        
        # Distribute any remaining median points evenly between left and right splits
        if median_mask.any():
            median_indices = torch.where(median_mask)[0]
            n_median = median_indices.size(0)
            # Split half to left, half to right (first half to left, second half to right)
            left_half = median_indices[:n_median//2]
            right_half = median_indices[n_median//2:]
            
            # Update masks
            left_mask[left_half] = True
            right_mask[right_half] = True
            
        assert not (left_mask & right_mask).any(), "Left and right masks should not overlap"
        assert left_mask.sum() - right_mask.sum() <= 1, "Left and right masks should have the same number of elements"
        
        return left_mask, right_mask

    def _build_tree(self, X, y, X_val, y_val, train_indices=None, depth=0, avg_M=None, is_root=False):
        """
        Recursively build the tree by splitting data based on random projections.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features
        y : torch.Tensor
            Target values
        X_val : torch.Tensor
            Validation features
        y_val : torch.Tensor
            Validation target values
        depth : int
            Current depth in the tree
        avg_M : torch.Tensor, optional
            Averaged M matrix to use for generating projections
        is_final_iter : bool, default=False
            Whether this is the final iteration of tree building
            
        Returns
        -------
        dict
            A tree node (either a leaf with a model or an internal node with split information)
        """
        n_samples = X.shape[0]
        if train_indices is None:
            train_indices = torch.arange(n_samples, device=self.device)

        # Check terminal conditions
        if (n_samples <= self.min_subset_size) or (self.max_depth is not None and depth >= self.max_depth):
            if not is_root: # refill the validation set if you've split the data before
                print("Refilling validation set, because at least one split has been made.")
                X, y, X_val, y_val, train_indices = self._refill_val_set(X, y, X_val, y_val, train_indices)

            # Create and fit a TabRFM model on this subset
            model = RFM(**self.rfm_params['model'], tuning_metric=self.tuning_metric, 
                        categorical_info=self.categorical_info, device=self.device)
            
            model.fit((X, y), (X_val, y_val), **self.rfm_params['fit'], callback=self.callback)
            return {'type': 'leaf', 'model': model, 'train_indices': train_indices, 'is_root': is_root}
        
        # Generate projection vector
        if avg_M is not None and self.split_method == 'random_global_agop':
            projection = self._generate_projection_from_M(X.shape[1], avg_M)
        elif self.split_method == 'random_pca':
            Xb = X - X.mean(dim=0, keepdim=True)
            Xcov = Xb.T @ Xb
            projection = self._generate_projection_from_M(X.shape[1], Xcov)
        elif self.split_method == 'linear':
            XtX = X.T @ X
            beta = torch.linalg.solve(XtX + 1e-6*torch.eye(X.shape[1], device=self.device), X.T @ y)
            beta = beta.mean(dim=1) # probably not the best way to do this
            projection = beta / torch.norm(beta)
        elif 'agop_on_subset' in self.split_method:
            print(f"Using {self.split_method} split method")
            M = self._get_agop_on_subset(X, y)
            if self.split_method == 'top_vector_agop_on_subset':
                # Vt = torch.linalg.eigh(M)[1].T
                _, _, Vt = torch.linalg.svd(M, full_matrices=False) # more stable than eigh and should be identical for top vectors
                projection = Vt[0]
            elif self.split_method == 'random_agop_on_subset':
                projection = self._generate_projection_from_M(X.shape[1], M)
            elif self.split_method == 'top_pc_agop_on_subset':
                sqrtM = matrix_power(M, 0.5)
                XM = X @ sqrtM
                Xb = XM - XM.mean(dim=0, keepdim=True)
                # _, eig_vecs = torch.linalg.eigh(Xb.T @ Xb)
                _, _, Vt = torch.linalg.svd(Xb.T @ Xb, full_matrices=False) # do computation on Xb.T @ Xb assuming n >> d
                projection = Vt[0]
        elif self.split_method == 'fixed_vector':
            projection = self.fixed_vector
        else:
            projection = self._generate_random_projection(X.shape[1])
        
        # Project data onto the random direction
        projections = X @ projection
        
        # Find median as split point
        train_median = torch.median(projections)
        
        # Get balanced split for training set to avoid infinite recursion with repeated data
        left_mask, right_mask = self._get_balanced_split(projections, train_median)

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # Possibly imbalanced split for validation set
        projections_val = X_val @ projection
        left_mask_val = projections_val <= train_median
        right_mask_val = ~left_mask_val
        
        X_val_left, y_val_left = X_val[left_mask_val], y_val[left_mask_val]
        X_val_right, y_val_right = X_val[right_mask_val], y_val[right_mask_val] 
        
        # Build subtrees
        left_tree = self._build_tree(X_left, y_left, X_val_left, y_val_left,  
                                     train_indices=train_indices[left_mask], 
                                     depth=depth + 1,
                                     avg_M=avg_M,
                                     is_root=False)
        right_tree = self._build_tree(X_right, y_right, X_val_right, y_val_right, 
                                      train_indices=train_indices[right_mask], 
                                      depth=depth + 1, 
                                      avg_M=avg_M,
                                      is_root=False)
        
        return {
            'type': 'split',
            'split_direction': projection,
            'split_point': train_median,
            'left': left_tree,
            'right': right_tree,
            'is_root': is_root
        }
    
    def _refill_val_set(self, X, y, X_val, y_val, train_indices):
        """
        Refill the validation set with samples from the training set.
        
        This method ensures that each leaf node has a sufficient validation set
        for proper model tuning. When the validation set becomes too small after
        tree splitting, it transfers samples from the training set to the 
        validation set.
        
        Parameters
        ----------
        X : torch.Tensor
            Training features
        y : torch.Tensor
            Training targets
        X_val : torch.Tensor
            Validation features
        y_val : torch.Tensor
            Validation targets
        train_indices : torch.Tensor
            Indices of training samples in the original dataset
            
        Returns
        -------
        tuple
            Updated (X, y, X_val, y_val, train_indices) with refilled validation set
        """

        if len(X_val) <= self.min_val_size:
            n_orig_val = len(X_val)
            n_orig_train = len(X)

            num_val_to_add = self.min_val_size - len(X_val)
            num_val_to_add = min(num_val_to_add, int(len(X)*self.val_size_frac))
            shuffled_indices = torch.randperm(len(X))
            val_indices = shuffled_indices[:num_val_to_add]
            local_train_indices_to_keep = shuffled_indices[num_val_to_add:]

            X_val = torch.cat([X_val, X[val_indices]])
            y_val = torch.cat([y_val, y[val_indices]])
            X = X[local_train_indices_to_keep]
            y = y[local_train_indices_to_keep]

            train_indices = train_indices[local_train_indices_to_keep]

            assert n_orig_val + num_val_to_add == len(X_val) == len(y_val)
            assert n_orig_train - num_val_to_add == len(X) == len(y)

        return X, y, X_val, y_val, train_indices
    
    def _build_tree_with_iterations(self, X, y, X_val, y_val):
        """
        Build a tree using multiple iterations, where each iteration uses
        information from the previous iteration's leaf models.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features
        y : torch.Tensor
            Target values
        X_val : torch.Tensor
            Validation features
        y_val : torch.Tensor
            Validation target values
            
        Returns
        -------
        dict
            Final tree structure with the best validation performance
        """
        avg_M = None

        # First iteration: use random projections
        tree = self._build_tree(X, y, X_val, y_val, avg_M=None, is_root=True)
        
        # Evaluate the first tree on validation data
        best_val_score = self.score_tree(X_val, y_val, tree)
        best_tree = self.tree_copy(tree)

        val_scores = [best_val_score+0]
        
        for _ in tqdm(range(self.n_tree_iters), desc="Iterating tree"):
            
            # Later iterations: use averaged M from previous iterations
            avg_M = self._average_M_across_leaves(tree)

            del tree

            # Build new tree with improved projections
            tree = self._build_tree(X, y, X_val, y_val, avg_M=avg_M, is_root=False)

            # Evaluate this iteration's tree on validation data
            val_score = self.score_tree(X_val, y_val, tree)
            val_scores.append(val_score)

            if self.maximizing_metric and val_score > best_val_score:
                best_val_score = val_score
                best_tree = self.tree_copy(tree)
            elif not self.maximizing_metric and val_score < best_val_score:
                best_val_score = val_score
                best_tree = self.tree_copy(tree)
            
            

        print("==========================Tree iteration results==========================")
        print("Validation scores over tree iterations:", val_scores)
        print("Best validation score:", best_val_score)
        print("==========================================================================")
        return best_tree
        
    
    def fit(self, X, y, X_val, y_val):
        """
        Fit the xRFM model to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        X_val : array-like of shape (n_samples, n_features)
            Validation feature matrix
        y_val : array-like of shape (n_samples,) or (n_samples, n_targets)
            Validation target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        print(f"Fitting xRFM with {self.n_trees} trees and {self.n_tree_iters} iterations per tree")
        # Convert to torch tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        self.data_dim = X.shape[1]
        
        # Ensure y has the right shape
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        if len(y_val.shape) == 1:
            y_val = y_val.unsqueeze(-1)

        # Build multiple trees
        self.trees = []
        for _ in tqdm(range(self.n_trees), desc="Building trees"):
            if self.n_tree_iters > 0:
                tree = self._build_tree_with_iterations(X, y, X_val, y_val)
            else:
                tree = self._build_tree(X, y, X_val, y_val, is_root=True)
            self.trees.append(tree)

            if tree['type'] == 'leaf':
                print("Tree has no split, stopping training")
                break
        
        return self
    
    
    def score(self, samples, targets):
        """
        Return the score of the model on the given samples and targets on self.tuning_metric.
        
        Parameters
        ----------
        samples : array-like of shape (n_samples, n_features)
            Test samples
        targets : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for samples
            
        Returns
        -------
        float
            Score of the model on the given samples and targets
        """
        

        if self.tuning_metric == 'accuracy':
            preds = self.predict_proba(samples.to(self.device)).to(targets.device)
            preds_ = torch.argmax(preds,dim=-1)
            targets_ = torch.argmax(targets,dim=-1)
            num_classes = preds.shape[-1]
            if num_classes==2:
                return accuracy(preds_, targets_, task="binary").item()
            else:
                return accuracy(preds_, targets_, task="multiclass", num_classes=num_classes).item()
        
        elif self.tuning_metric == 'mse':
            preds = self.predict(samples.to(self.device)).to(targets.device)
            return (targets - preds).pow(2).mean()

        elif self.tuning_metric == 'auc':
            preds = self.predict_proba(samples.to(self.device)).to(targets.device)
            num_classes = preds.shape[-1]
            if num_classes==2:
                return roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy()[:,1])
            else:
                return roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy(), multi_class='ovr')
        
        else:
            raise ValueError(f"Invalid tuning metric: {self.tuning_metric}")
        
    def score_tree(self, samples, targets, tree):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        samples : array-like of shape (n_samples, n_features)
            Test samples
        targets : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for samples
        tree : dict
            Tree to use for prediction
            
        Returns
        -------
        float
            MSE loss if regression, accuracy if classification
        """
        
        num_classes = len(torch.unique(targets))
        if self.tuning_metric == 'accuracy':
            preds = self._predict_tree(samples.to(self.device), tree, proba=True).to(targets.device)
            preds_ = torch.argmax(preds,dim=-1)
            targets_ = torch.argmax(targets,dim=-1)
            num_classes = preds.shape[-1]
            if num_classes==2:
                return accuracy(preds_, targets_, task="binary").item()
            else:
                return accuracy(preds_, targets_, task="multiclass", num_classes=num_classes).item()
        
        elif self.tuning_metric == 'mse':
            preds = self._predict_tree(samples.to(self.device), tree).to(targets.device)
            return (targets - preds).pow(2).mean().item()

        elif self.tuning_metric == 'auc':
            preds = self._predict_tree(samples.to(self.device), tree, proba=True).to(targets.device)
            if num_classes <= 2:
                return roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy()[:,1])
            else:
                return roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())
        
        else:
            raise ValueError(f"Invalid tuning metric: {self.tuning_metric}")
        
    def predict(self, X):
        """
        Predict using the xRFM model by averaging predictions across all trees.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        array-like
            Returns predicted values
        """
        if self.trees is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert to torch tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        all_predictions = []
        
        # Get predictions from each tree
        for tree in self.trees:
            tree_predictions = self._predict_tree(X, tree)
            all_predictions.append(tree_predictions)
        
        # Average predictions across trees
        return torch.mean(torch.stack(all_predictions), dim=0)

    def predict_proba(self, X):
        """
        Predict class probabilities by averaging across all trees.
        Only usable if the underlying TabRFM models were fitted for classification.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        array-like
            Returns predicted probabilities
        """
        if self.trees is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert to torch tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        all_probas = []
        for tree in self.trees:
            tree_probas = self._predict_tree(X, tree, proba=True)
            all_probas.append(tree_probas)
        
        return torch.mean(torch.stack(all_probas), dim=0)
    
        
    def _predict_tree(self, X, tree, proba=False):
        """
        Make predictions for all samples using a single tree.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features
        tree : dict
            Tree to use for prediction
            
        Returns
        -------
        torch.Tensor
            Predictions for all samples
        """

        X_leaf_groups, X_leaf_group_indices, leaf_nodes = self._get_leaf_groups_and_models_on_samples(X, tree)
        predictions = []
        for X_leaf, leaf_node in zip(X_leaf_groups, leaf_nodes):
            if proba:
                preds = leaf_node['model'].predict_proba(X_leaf)
            else:
                preds = leaf_node['model'].predict(X_leaf)
            predictions.append(preds)
        
        def reorder_tensor(original_tensor, order_tensor):
            """
            Args:
                original_tensor: The tensor to be reordered
                order_tensor: A tensor containing the new positions for each element
                
            Returns:
                The reordered tensor
            """
            # Sort the indices based on the order tensor
            # This gives us the inverse permutation needed
            _, sorted_indices = torch.sort(order_tensor)
            
            # Use the sorted indices to reorder the original tensor
            return original_tensor[sorted_indices]

        order = torch.cat(X_leaf_group_indices, dim=0)
        return reorder_tensor(torch.cat(predictions, dim=0), order)
    
    def load_state_dict(self, state_dict, X_train):
        """
        Load model state from a state dictionary.
        
        This method reconstructs the model from saved parameters, including
        the tree structure and leaf model parameters. The training data is
        needed to set the centers for each leaf model.
        
        Parameters
        ----------
        state_dict : dict
            Dictionary containing model parameters from get_state_dict()
        X_train : torch.Tensor
            Training data used to set leaf model centers
        """
        self.rfm_params = state_dict['rfm_params']
        self.categorical_info = state_dict['categorical_info']

        self._build_leaf_models_from_param_trees(state_dict['param_trees'])

        # set centers for leaf models
        for tree in self.trees:
            assert tree['is_root']
            leaf_nodes = self._collect_leaf_nodes(tree)
            for leaf_node in leaf_nodes:
                leaf_model = leaf_node['model']
                leaf_center_indices = leaf_node['train_indices']
                leaf_model.centers = X_train[leaf_center_indices]
        return
    
    def _build_leaf_models_from_param_trees(self, param_trees):
        """
        Build leaf models from parameter trees during model loading.
        
        This method reconstructs the tree structure and instantiates RFM models
        at each leaf node using the saved parameters. It traverses the tree 
        structure and sets the model attributes at leaf nodes.
        
        Parameters
        ----------
        param_trees : list
            List of parameter trees from the state dictionary
        """
        self.trees = []

        def set_leaf_model_single_tree(tree):
            if tree['type'] == 'leaf':
                leaf_model = RFM(**self.rfm_params['model'], 
                                 categorical_info=self.categorical_info, 
                                 device=self.device)
                leaf_model.kernel_obj.bandwidth = tree['bandwidth']
                leaf_model.weights = tree['weights']
                leaf_model.M = tree['M']
                leaf_model.sqrtM = tree['sqrtM']
                tree['model'] = leaf_model
                return tree
            else:
                tree['left'] = set_leaf_model_single_tree(tree['left'])
                tree['right'] = set_leaf_model_single_tree(tree['right'])
                return tree
    
        for param_tree in param_trees:
            self.trees.append(set_leaf_model_single_tree(param_tree))

        return
    
    def get_state_dict(self):
        """
        Get the state dictionary containing all model parameters for serialization.
        
        The state dictionary contains the tree structure and all parameters needed
        to reconstruct the model, including individual weights, M/sqrtM matrices,
        and bandwidths for each leaf model. This enables model saving and loading.
        
        Returns
        -------
        dict
            State dictionary with keys:
            - 'rfm_params': RFM model parameters
            - 'categorical_info': Categorical feature information
            - 'param_trees': List of parameter trees containing leaf model parameters
        """
        param_trees = []
        for tree in self.trees:
            param_trees.append(get_param_tree(tree, is_root=True))
        return {
            'rfm_params': self.rfm_params,
            'categorical_info': self.categorical_info,
            'param_trees': param_trees,
        }
    

    def _get_agop_on_subset(self, X, y, subset_size=50_000):
        """
        
        This method fits a base RFM model on a subset of the data to compute the AGOP matrix,
        whose eigenvectors are used to generate projection direction for data splitting.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (n_samples, n_features)
        y : torch.Tensor
            Target values of shape (n_samples, n_targets)
        subset_size : int, default=50000
            Maximum size of the subset to use for AGOP computation
            
        Returns
        -------
        torch.Tensor
            AGOP matrix of shape (n_features, n_features)
        """
        model = RFM(**self.default_rfm_params['model'], device=self.device)

        subset_size = min(subset_size, len(X))
        subset_train_size = int(subset_size * 0.95) # 95/5 split, probably won't need the val data.

        subset_indices = torch.randperm(len(X))
        subset_train_indices = subset_indices[:subset_train_size]
        subset_val_indices = subset_indices[subset_train_size:subset_size]

        X_train = X[subset_train_indices]
        y_train = y[subset_train_indices]
        X_val = X[subset_val_indices]
        y_val = y[subset_val_indices]

        print("Getting AGOP on subset")
        print("X_train", X_train.shape, "y_train", y_train.shape, "X_val", X_val.shape, "y_val", y_val.shape)

        model.fit((X_train, y_train), (X_val, y_val), **self.default_rfm_params['fit'])
        agop = model.agop_best_model
        print("AGOP on subset", agop.shape)
        print("M", agop.diag()[:5])
        return agop

    def _get_leaf_groups_and_models_on_samples(self, X, tree):
        """
        Assign samples to leaf nodes and return grouped data with corresponding models.
        
        This method traverses the tree to determine which leaf node each sample
        belongs to, then groups the samples by leaf and returns the corresponding
        models for making predictions.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data matrix of shape (n_samples, n_features)
        tree : dict
            Tree structure with split directions and leaf models
            
        Returns
        -------
        X_leaf_groups : list of torch.Tensor
            List of data tensors, one for each leaf node containing the samples
            that belong to that leaf
        X_leaf_group_indices : list of torch.Tensor
            List of tensors containing the original indices of samples in each 
            leaf group, used for reordering predictions
        leaf_nodes : list of dict
            List of leaf node dictionaries containing the trained models
        """
        # Initialize results lists
        X_leaf_groups = []
        X_leaf_group_indices = []
        leaf_nodes = []
        
        # Initialize stack with the root node and all sample indices
        sample_indices = torch.arange(X.shape[0], device=self.device)
        stack = [(X, sample_indices, tree)]
        
        # Iterative traversal of the tree
        while stack:
            current_X, current_indices, current_node = stack.pop()
            
            # If we've reached a leaf node, store the results
            if current_node['type'] == 'leaf':
                X_leaf_groups.append(current_X)
                X_leaf_group_indices.append(current_indices)
                leaf_nodes.append(current_node)
                continue
            
            # Compute projections for all samples in current_X
            projections = current_X @ current_node['split_direction']
            
            # Split samples based on projection values
            left_mask = projections <= current_node['split_point']
            right_mask = ~left_mask
            
            # Add right child to stack (will be processed first since we're using pop())
            if right_mask.sum() > 0:
                stack.append((
                    current_X[right_mask], 
                    current_indices[right_mask], 
                    current_node['right']
                ))
            
            # Add left child to stack
            if left_mask.sum() > 0:
                stack.append((
                    current_X[left_mask], 
                    current_indices[left_mask], 
                    current_node['left']
                ))
        
        return X_leaf_groups, X_leaf_group_indices, leaf_nodes
    