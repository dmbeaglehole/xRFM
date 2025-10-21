from typing import Optional, Union, Tuple

import torch

try:
    import kermac
except ImportError:
    kermac = None

def get_sub_matrix(mat: Union[torch.Tensor, None], indices: torch.Tensor) -> Union[torch.Tensor, None]:
    """
    Get a submatrix of mat based on the indices.
    """
    if mat is None:
        return None
    if len(mat.shape) == 1:
        return mat[indices]
    else:
        return mat[indices][:, indices]

class Kernel:
    def __init__(self):
        self.is_adaptive_bandwidth = True
        self.handle_categorical = False
        self.use_sqrtM = True
        
    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def _transform_m(self, x: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies the given transformation matrix to x.
        :param x: Points of shape (n, d_in).
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in,) or None.
            A vector will be interpreted as a diagonal matrix, and None as the identity matrix.
        :return: Tensor of shape (n, d_out), where d_out=d_in in case mat is a vector or None.
        """
        if mat is not None:
            if len(mat.shape) == 1:
                # diagonal
                x = x * mat[None, :].to(dtype=x.dtype)
            elif len(mat.shape) == 2:
                x = x @ mat.to(dtype=x.dtype)
            else:
                raise ValueError(f'm_matrix should have one or two dimensions, but got shape {mat.shape}')
        return x

    def _reset_adaptive_bandwidth(self):
        self.is_adaptive_bandwidth = False
        return

    def _adapt_bandwidth(self, kernel_mat: torch.Tensor, adapt_mode='median', sub_mat_size=5000):
        """
        Input is distance matrix with entries D(x,z)^p for exponent p.
        """
        n, m = kernel_mat.shape
        assert n <= m, "Kernel matrix must be wider than it is tall"

        # Subsample the kernel matrix to avoid OOM for mean/median calculation
        sub_mat_size = min(sub_mat_size, n)
        sample_indices = torch.randperm(n)[:sub_mat_size]
        sample_matrix = kernel_mat[sample_indices][:, sample_indices]

        # We need to take element-wise root of entries
        if self.exponent != 1.0:
            sample_matrix = sample_matrix**(1/self.exponent)

        # mask for off-diagonal elements
        mask = ~torch.eye(sub_mat_size, dtype=bool, device=kernel_mat.device)

        # Get mean/median of off-diagonal elements only
        if adapt_mode == 'median':
            bandwidth_multiplier = torch.median(sample_matrix[mask])
        elif adapt_mode == 'mean':
            bandwidth_multiplier = torch.mean(sample_matrix[mask])
        else:
            raise ValueError(f"Invalid adapt_mode: {adapt_mode}")
        self.bandwidth = self.base_bandwidth * bandwidth_multiplier.item()
        self.is_adaptive_bandwidth = True
        return

    
    def set_categorical_indices(self, numerical_indices, categorical_indices, categorical_vectors, device='cuda'):
        print("Setting categorical indices")
        self.numerical_indices = numerical_indices.to(device)
        self.categorical_indices = [categorical_indices[i].to(device) for i in range(len(categorical_indices))]
        self.categorical_vectors = [categorical_vectors[i].to(device) for i in range(len(categorical_vectors))]
        self.handle_categorical = True
        return
    
    def get_agop(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                 mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        if self.handle_categorical:
            return self.get_agop_categorical(x, z, coefs, mat, center_grads)
        else:
            # see get_function_grads
            f_grads = self.get_function_grads(x, z, coefs, mat)
            # merge output and n_z dims
            f_grads = f_grads.reshape(-1, f_grads.shape[-1])
            if center_grads:
                f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
            return f_grads.transpose(-1, -2) @ f_grads
    
    def get_agop_categorical(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                 mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        
        numerical_indices = self.numerical_indices
        categorical_indices = self.categorical_indices

        # see get_function_grads
        f_grads = self.get_function_grads(x, z, coefs, mat)
        # merge output and n_z dims
        f_grads = f_grads.reshape(-1, f_grads.shape[-1])
        if center_grads:
            f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
        
        # Initialize the final AGOP matrix with zeros
        d = x.shape[1]
        agop = torch.zeros((d, d), device=x.device, dtype=x.dtype)
        
        # Place numerical block if it exists
        if numerical_indices is not None and len(numerical_indices) > 0:
            agop[numerical_indices[:, None], numerical_indices] = (
                f_grads[:, numerical_indices].T @ f_grads[:, numerical_indices]
            )
        
        # Place categorical blocks
        if categorical_indices is not None:
            for cat_idx in categorical_indices:
                agop[cat_idx[:, None], cat_idx] = (
                    f_grads[:, cat_idx].T @ f_grads[:, cat_idx]
                )
        
        return agop
        
    def get_kernel_matrix(self, x: torch.Tensor, z: torch.Tensor,
                          mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the kernel matrix (k(x[i, :], z[j, :]))_{i,j}
        :param x: Points of shape (n_x, d_in).
        :param z: Points of shape (n_z, d_in).
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in,) or None. This will be applied to x and z.
        Corresponds to sqrtM in RFM.    
        :return: The kernel matrix of shape (n_x, n_z).
        """
        if self.handle_categorical:
            return self._get_kernel_matrix_categorical_impl(x, z, mat)
        else:
            return self._get_kernel_matrix_impl(x, z, mat)
    
    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor,
                          mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def get_kernel_matrix_symm(self, x: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # todo: only compute certain blocks?
        return self.get_kernel_matrix(x, x, mat)

    def get_function_grads(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                           mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the matrix of function gradients at points z.
        The function is given by f_l(\cdot) = \sum_i coefs[l, i] * k(x[i], \cdot).
        :param x: Matrix of shape (n_x, d_in)
        :param z: Matrix of shape (n_z, d_in)
        :param coefs: Vector of shape (f, n_x) where f is the number of functions
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in)
        :return: Should return a tensor of shape (f, n_z, d_in).
        """
        grads = self._get_function_grad_impl(x, z, coefs, mat) # get grad_Mx k(Mx, Mz)
        return self._transform_m(grads, mat) # use that grad_x k(Mx, Mz) = M grad_Mx k(Mx, Mz)
    
    def get_agop_diag(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                      mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        # see get_function_grads
        f_grads = self.get_function_grads(x, z, coefs, mat)
        # merge output and n_z dims
        f_grads = f_grads.reshape(-1, f_grads.shape[-1])
        if center_grads:
            f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
        return f_grads.square().sum(dim=-2)

class LightLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps
        self.use_sqrtM = False

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)

        xm_norm_sqr = (xm*x).sum(dim=-1)
        zm_norm_sqr = (zm*z).sum(dim=-1)

        kernel_mat = xm_norm_sqr[:, None] - 2*xm@z.T + zm_norm_sqr[None, :]
        kernel_mat.clamp_(min=0)
        kernel_mat.sqrt_()

        if self.exponent != 1.0:
            kernel_mat.pow_(self.exponent)
            
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)

        kernel_mat.mul_(-1. / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def get_function_grads(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, 
                           mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)

        xm_norm_sqr = (xm*x).sum(dim=-1)
        zm_norm_sqr = (zm*z).sum(dim=-1)

        dists = xm_norm_sqr[:, None] - 2*xm@z.T + zm_norm_sqr[None, :]
        dists.clamp_(min=0)
        dists.sqrt_()

        kernel_mat = dists ** self.exponent
        kernel_mat.mul_(-1/(self.bandwidth ** self.exponent))
        kernel_mat.exp_()

        # now compute M
        mask = dists >= self.eps
        dists.clamp_(min=self.eps)
        dists.pow_(self.exponent - 2)
        kernel_mat.mul_(dists)
        kernel_mat.mul_(mask)  # this is very important for numerical stability
        kernel_mat.mul_(-self.exponent / (self.bandwidth ** self.exponent))

        # now we want result[l, j, d] = \sum_i coefs[l, i] M[i, j] (z[j, d] - x[i, d])
        return torch.einsum('li,ij,jd->ljd', coefs, kernel_mat, zm) - torch.einsum('li,ij,id->ljd', coefs, kernel_mat, xm)
        
class LaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps  # this one is for numerical stability

    def get_sample_batch_size(self, n: int, d: int, scalar_size: int = 4, mem_constant: float = 20) -> int:
        if torch.cuda.is_available():
            total_memory_possible = torch.cuda.get_device_properties(torch.device('cuda')).total_memory
            curr_mem_use = torch.cuda.memory_allocated()
            available_memory = total_memory_possible - curr_mem_use
            return int(available_memory / (mem_constant*n*scalar_size))
        else:
            # hard code here
            return 5_000

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        kernel_mat = torch.cdist(self._transform_m(x, mat), self._transform_m(z, mat))
        kernel_mat.clamp_(min=0)
        if self.exponent != 1.0:
            kernel_mat.pow_(self.exponent)

        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)

        kernel_mat.mul_(-1. / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        numerical_indices = self.numerical_indices # (n_num,)
        categorical_indices = self.categorical_indices # List of (d_cat_i,) for each categorical feature
        categorical_vectors = self.categorical_vectors # List of (d_cat_i, d_cat_i) for each categorical feature

        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        
        def squared_dist_fn(x, z):
            kernel_mat = torch.cdist(x, z)**2
            kernel_mat.clamp_(min=0)
            return kernel_mat

        
        mat_num = get_sub_matrix(mat, numerical_indices)
        xnum = self._transform_m(x[:, numerical_indices], mat_num)
        znum = self._transform_m(z[:, numerical_indices], mat_num)

        # Initialize kernel matrix with squared distances of numerical features
        dist_mat = squared_dist_fn(xnum, znum)

        # Add to running total of squared distances for categorical features
        batch_size = self.get_sample_batch_size(znum.shape[0], znum.shape[1])
        for i, (cat_idx, cat_vecs) in enumerate(zip(categorical_indices, categorical_vectors)):
            x_cat = x[:, cat_idx].argmax(dim=-1)
            z_cat = z[:, cat_idx].argmax(dim=-1)

            # Get the kernel matrix for this categorical feature's embeddings
            mat_cat = get_sub_matrix(mat, cat_idx)
            cat_vecs_transformed = self._transform_m(cat_vecs, mat_cat)
            cat_embedding_kernel = squared_dist_fn(cat_vecs_transformed, cat_vecs_transformed)

            # Index into the kernel matrix using the categorical indices
            for i in range(0, x.shape[0], batch_size):
                dist_mat[i:i+batch_size].add_(cat_embedding_kernel[x_cat[i:i+batch_size, None], z_cat[None, :]])

        # Take square root of squared distances to get 2-norm distances -> ||x-z||_2
        dist_mat.sqrt_()

        # Apply exponent -> ||x-z||^p_2
        dist_mat.pow_(self.exponent)

        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(dist_mat)

        # Apply bandwidth and exponent -> exp(-||x-z||^p_2 / L^p)
        dist_mat.mul_(-1./(self.bandwidth**self.exponent))
        dist_mat.exp_()
        return dist_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:

        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)
        dists = torch.cdist(xm, zm)
        dists.clamp_(min=0)

        # gradient of k(x, z) = exp(-\gamma \|x - z\|^\beta) wrt z  (where \beta = self.exponent)
        # is -\gamma k(x, z) \beta \|x - z\|^{\beta - 1} (z-x)/\|x-z\| = -\gamma \beta k(x, z) \|x - z\|^{\beta-2} (z-x)
        # therefore, setting f_l (z) = \sum_i coefs[l, i] k(x[i], z), we have
        # \grad f_l(z[j]) = \sum_i coefs[l, i] M[i, j] (z[j] - x[i]),
        # where M[i, j] = -\gamma \beta k(x[i], z[j]) \|x[i] - z[j]\|^{\beta - 2}
        kernel_mat = dists ** self.exponent
        kernel_mat.mul_(-1 / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()

        # now compute M
        mask = dists >= self.eps
        dists.clamp_(min=self.eps)
        dists.pow_(self.exponent - 2)
        kernel_mat.mul_(dists)
        kernel_mat.mul_(mask)  # this is very important for numerical stability
        kernel_mat.mul_(-self.exponent / (self.bandwidth ** self.exponent))

        # now we want result[l, j, d] = \sum_i coefs[l, i] M[i, j] (z[j, d] - x[i, d])
        return torch.einsum('li,ij,jd->ljd', coefs, kernel_mat, zm) - torch.einsum('li,ij,id->ljd', coefs, kernel_mat, xm)

class ProductLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps  # this one is for numerical stability

    def get_sample_batch_size(self, n: int, d: int, scalar_size: int = 4, mem_constant: float = 20) -> int:
        total_memory_possible = torch.cuda.get_device_properties(torch.device('cuda')).total_memory
        curr_mem_use = torch.cuda.memory_allocated()
        available_memory = total_memory_possible - curr_mem_use
        return int(available_memory / (mem_constant*n*scalar_size))

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None, kernel_batch_size=20_000) -> torch.Tensor:
        n, d = z.shape
        if x.shape[0] <= kernel_batch_size:
            kernel_mat = torch.cdist(self._transform_m(x, mat), self._transform_m(z, mat), p=self.exponent)
        else:
            kernel_mat = torch.empty((x.shape[0], z.shape[0]), device=x.device, dtype=x.dtype)
            for i in range(0, x.shape[0], kernel_batch_size):
                kernel_mat[i:i+kernel_batch_size] = torch.cdist(self._transform_m(x[i:i+kernel_batch_size], mat), self._transform_m(z, mat), p=self.exponent)

        kernel_mat.clamp_(min=0)

        kernel_mat.pow_(self.exponent)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)

        kernel_mat.mul_(-1./(self.bandwidth**self.exponent))
        kernel_mat.exp_()
        return kernel_mat
    
    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        numerical_indices = self.numerical_indices # (n_num,)
        categorical_indices = self.categorical_indices # List of (d_cat_i,) for each categorical feature
        categorical_vectors = self.categorical_vectors # List of (d_cat_i, d_cat_i) for each categorical feature

        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        
        def dist_fn(x, z):
            kernel_mat = torch.cdist(x, z, p=self.exponent)
            kernel_mat.clamp_(min=0)
            if self.exponent != 1.0:
                kernel_mat.pow_(self.exponent)
            return kernel_mat
        
        mat_num = get_sub_matrix(mat, numerical_indices)
        xnum = self._transform_m(x[:, numerical_indices], mat_num)
        znum = self._transform_m(z[:, numerical_indices], mat_num)

        batch_size = self.get_sample_batch_size(znum.shape[0], znum.shape[1])
        kernel_mat = torch.zeros((xnum.shape[0], znum.shape[0]), device=xnum.device, dtype=xnum.dtype)
        num_batch_size = 2*batch_size

        for i in range(0, xnum.shape[0], num_batch_size):
            kernel_mat[i:i+num_batch_size, :] = dist_fn(xnum[i:i+num_batch_size], znum)
        for cat_idx, cat_vecs in zip(categorical_indices, categorical_vectors):

            x_cat = x[:, cat_idx].argmax(dim=-1)
            z_cat = z[:, cat_idx].argmax(dim=-1)

            # Get the kernel matrix for this categorical feature's embeddings
            mat_cat = get_sub_matrix(mat, cat_idx)
            cat_vecs_transformed = self._transform_m(cat_vecs, mat_cat)
            cat_embedding_kernel = dist_fn(cat_vecs_transformed, cat_vecs_transformed)

            # Index into the kernel matrix using the categorical indices
            for i in range(0, x.shape[0], batch_size):
                kernel_mat[i:i+batch_size].add_(cat_embedding_kernel[x_cat[i:i+batch_size, None], z_cat[None, :]])

        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)

        kernel_mat.mul_(-1./(self.bandwidth**self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)
        def forward_func(zm_):
            dists = torch.cdist(xm, zm_, p=self.exponent) ** self.exponent
            factor = -((1. / self.bandwidth) ** self.exponent)
            # this is \sum_j f(z_j), so the derivative wrt z will be jacobian(f)(z_j) for all z_j
            return coefs @ torch.exp(factor * (dists * (dists >= self.eps))).sum(dim=1)

        return torch.func.jacrev(forward_func)(zm)


class LpqLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, p: float, q: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert 0 < p <= 2
        assert 0 < q <= p
        assert eps > 0
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.p = p
        self.exponent = q
        self.eps = eps  # this one is for numerical stability
        self.bandwidth_mode = bandwidth_mode

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        kernel_mat = torch.cdist(self._transform_m(x, mat), self._transform_m(z, mat), p=self.p)
        kernel_mat.clamp_(min=0)
        
        kernel_mat.pow_(self.exponent)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)

        kernel_mat.mul_(-1. / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)

        def forward_func(z_transformed: torch.Tensor) -> torch.Tensor:
            base_dists = torch.cdist(xm, z_transformed, p=self.p)
            base_mask = base_dists >= self.eps

            dist_pow_q = torch.where(
                base_mask,
                base_dists.clamp_min(self.eps).pow(self.exponent),
                torch.zeros_like(base_dists),
            )

            factor = -((1.0 / self.bandwidth) ** self.exponent)
            kernel_vals = torch.exp(factor * dist_pow_q)
            return coefs @ kernel_vals.sum(dim=1)

        return torch.func.jacrev(forward_func)(zm)

class SumPowerLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, const_mix: float = 0.0,
                 power: int = 2,
                 bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        assert 0 <= const_mix < 1
        assert bandwidth_mode == 'constant', 'Adaptive bandwidth currently not supported'
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.exponent = exponent
        self.const_mix = const_mix
        self.power = power
        self.eps = eps  # this one is for numerical stability
        self.bandwidth_mode = bandwidth_mode

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._transform_m(x, mat)
        z = self._transform_m(z, mat)
        
        diffs = x[:, None, :] - z[None, :, :]
        diffs.abs_()
        diffs.pow_(self.exponent)
        diffs.mul_(-1. / (self.bandwidth ** self.exponent))
        diffs.exp_()
        sum = diffs.sum(dim=-1)
        sum.mul_((1.0 - self.const_mix) / x.shape[1])  # normalize so the max sum is 1
        sum.add_(self.const_mix)
        sum.pow_(self.power)
        return sum

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        xm = self._transform_m(x, mat)  # (n_x, d)
        zm = self._transform_m(z, mat)  # (n_z, d)

        xm = self._transform_m(x, mat)  # (n_x, d)
        zm = self._transform_m(z, mat)  # (n_z, d)

        coefs = coefs.to(dtype=xm.dtype)

        def forward_func(z_tensor: torch.Tensor) -> torch.Tensor:
            diffs = torch.abs(xm[:, None, :] - z_tensor[None, :, :]).pow(self.exponent)
            diffs = torch.exp((-1.0 / (self.bandwidth ** self.exponent)) * diffs)
            summed = (1.0 - self.const_mix) * (diffs.sum(dim=-1) / float(xm.shape[-1])) + self.const_mix
            powered = summed.pow(self.power)
            # Sum across z dimension to get shape (n_x,)
            agg = powered.sum(dim=-1)
            return coefs @ agg

        grads = torch.func.jacrev(forward_func)(zm)
        return grads

class KermacProductLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-8, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert 0 < exponent <= 2
        assert eps > 0
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps  # this one is for numerical stability
        self.bandwidth_mode = bandwidth_mode

        if kermac is None:
            raise ImportError("kermac is required for KermacProductLaplaceKernel.")
        
    def get_sample_batch_size(self, n: int, d: int, scalar_size: int = 4, mem_constant: float = 20) -> int:
        total_memory_possible = torch.cuda.get_device_properties(torch.device('cuda')).total_memory
        curr_mem_use = torch.cuda.memory_allocated()
        available_memory = total_memory_possible - curr_mem_use
        return int(available_memory / (mem_constant*n*scalar_size))

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, 
                                mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        kernel_mat = kermac.cdist(self._transform_m(x, mat), self._transform_m(z, mat), p=self.exponent).squeeze(0)
        kernel_mat.clamp_(min=0)

        kernel_mat.pow_(self.exponent)

        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)

        kernel_mat.mul_(-1./(self.bandwidth**self.exponent))
        kernel_mat.exp_()

        return kernel_mat
    
    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        numerical_indices = self.numerical_indices # (n_num,)
        categorical_indices = self.categorical_indices # List of (d_cat_i,) for each categorical feature
        categorical_vectors = self.categorical_vectors # List of (d_cat_i, d_cat_i) for each categorical feature

        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        
        def dist_fn(x_, z_):
            kernel_mat = kermac.cdist(x_, z_, p=self.exponent).squeeze(0)
            kernel_mat.clamp_(min=0)
            if self.exponent != 1.0:
                kernel_mat.pow_(self.exponent)
            return kernel_mat

        
        mat_num = get_sub_matrix(mat, numerical_indices)
        xnum = self._transform_m(x[:, numerical_indices], mat_num)
        znum = self._transform_m(z[:, numerical_indices], mat_num)

        kernel_mat = dist_fn(xnum, znum)

        batch_size = self.get_sample_batch_size(znum.shape[0], znum.shape[1])
        for cat_idx, cat_vecs in zip(categorical_indices, categorical_vectors):

            x_cat = x[:, cat_idx].argmax(dim=-1)
            z_cat = z[:, cat_idx].argmax(dim=-1)

            # Get the kernel matrix for this categorical feature's embeddings
            mat_cat = get_sub_matrix(mat, cat_idx)
            cat_vecs_transformed = self._transform_m(cat_vecs, mat_cat)
            cat_embedding_kernel = dist_fn(cat_vecs_transformed, cat_vecs_transformed)

            # Index into the kernel matrix using the categorical indices
            for i in range(0, x.shape[0], batch_size):
                kernel_mat[i:i+batch_size].add_(cat_embedding_kernel[x_cat[i:i+batch_size, None], z_cat[None, :]])

        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        kernel_mat.mul_(-1./(self.bandwidth**self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.handle_categorical:
            kernel_mat = self._get_kernel_matrix_categorical_impl(x, z, mat)
        else:
            kernel_mat = self._get_kernel_matrix_impl(x, z, mat)

        a_mat = -kernel_mat * self.exponent / (self.bandwidth ** self.exponent)

        mask = (kernel_mat < (1-self.eps)).float()
        a_mat = a_mat * mask

        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)
        xm = xm.T.contiguous()
        zm = zm.T.contiguous()
        out = kermac.cdist_grad(a_mat, xm, coefs, zm, p=self.exponent)
        return out.transpose(-2, -1).float()
    

class KermacLpqLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, p: float, q: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert 0 < p <= 2
        assert 0 < q <= p
        assert eps > 0
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.p = p
        self.exponent = q
        self.eps = eps  # this one is for numerical stability
        self.bandwidth_mode = bandwidth_mode

    def get_sample_batch_size(self, n: int, d: int, scalar_size: int = 4, mem_constant: float = 20) -> int:
        total_memory_possible = torch.cuda.get_device_properties(torch.device('cuda')).total_memory
        curr_mem_use = torch.cuda.memory_allocated()
        available_memory = total_memory_possible - curr_mem_use
        return int(available_memory / (mem_constant*n*scalar_size))

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None, 
                                dist_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dist_mat is None:
            kernel_mat = kermac.cdist(self._transform_m(x, mat), self._transform_m(z, mat), p=self.p).squeeze(0)
        else:
            kernel_mat = dist_mat.clone()
        kernel_mat.clamp_(min=0)
        
        kernel_mat.pow_(self.exponent)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)

        kernel_mat.mul_(-1. / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()
        return kernel_mat
    
    def _get_dist_matrix_categorical(self, x, z, mat):
        numerical_indices = self.numerical_indices # (n_num,)
        categorical_indices = self.categorical_indices # List of (d_cat_i,) for each categorical feature
        categorical_vectors = self.categorical_vectors # List of (d_cat_i, d_cat_i) for each categorical feature

        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        
        def p_power_dist_fn(x_, z_):
            kernel_mat = kermac.cdist(x_, z_, p=self.p).squeeze(0)
            kernel_mat.clamp_(min=0)
            if self.p != 1.0:
                kernel_mat.pow_(self.p)
            return kernel_mat

        mat_num = get_sub_matrix(mat, numerical_indices)
        xnum = self._transform_m(x[:, numerical_indices], mat_num)
        znum = self._transform_m(z[:, numerical_indices], mat_num)

        kernel_mat = p_power_dist_fn(xnum, znum)

        batch_size = self.get_sample_batch_size(znum.shape[0], znum.shape[1])
        for cat_idx, cat_vecs in zip(categorical_indices, categorical_vectors):

            x_cat = x[:, cat_idx].argmax(dim=-1)
            z_cat = z[:, cat_idx].argmax(dim=-1)

            # Get the kernel matrix for this categorical feature's embeddings
            mat_cat = get_sub_matrix(mat, cat_idx)
            cat_vecs_transformed = self._transform_m(cat_vecs, mat_cat)
            cat_embedding_kernel = p_power_dist_fn(cat_vecs_transformed, cat_vecs_transformed)

            # Index into the kernel matrix using the categorical indices
            for i in range(0, x.shape[0], batch_size):
                kernel_mat[i:i+batch_size].add_(cat_embedding_kernel[x_cat[i:i+batch_size, None], z_cat[None, :]])

        return kernel_mat.clamp_(min=0).pow_(1 / self.p)
        
    
    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None, 
                                            dist_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if dist_mat is None:
            kernel_mat = self._get_dist_matrix_categorical(x, z, mat)
        else:
            kernel_mat = dist_mat.clone()
        kernel_mat.pow_(self.exponent) # exponentiate for bandwidth adaptation

        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        kernel_mat.mul_(-1./(self.bandwidth**self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)

        if self.handle_categorical:
            dist_mat = self._get_dist_matrix_categorical(x, z, mat)
            kernel_mat = self._get_kernel_matrix_categorical_impl(x, z, mat, dist_mat)
        else:
            dist_mat = kermac.cdist(xm, zm, p=self.p).squeeze(0)
            kernel_mat = self._get_kernel_matrix_impl(x, z, mat, dist_mat)
        
        mask = (dist_mat >= self.eps).float()   
        dist_mat = torch.clamp(dist_mat, min=self.eps)
        a_mat = -kernel_mat * self.exponent * dist_mat.pow(self.exponent - self.p) / (self.bandwidth ** self.exponent)
        a_mat = a_mat * mask

        del kernel_mat, dist_mat, mask

        xm = xm.T.contiguous()
        zm = zm.T.contiguous()
        out = kermac.cdist_grad(a_mat, xm, coefs, zm, p=self.p)
        return out.transpose(-2, -1).float()

class KermacSumPowerLaplaceKernel(Kernel):
    def __init__(
        self,
        bandwidth: float,
        p: float,
        q: float,
        const_mix: float = 0.0,
        eps: float = 1e-10,
        bandwidth_mode: str = 'constant',
    ):
        super().__init__()
        assert bandwidth > 0
        assert 0 < p <= 2
        assert 0 < q <= p
        assert eps > 0
        assert 0.0 <= const_mix < 1.0
        assert bandwidth_mode == 'constant', "Adaptive bandwidth currently not supported"
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.p = p  # keep attribute for backwards compatibility
        self.power = p
        self.exponent = q
        self.q = q
        self.const_mix = const_mix
        self.eps = eps  # this one is for numerical stability
        self.bandwidth_mode = bandwidth_mode

    def get_sample_batch_size(self, n: int, d: int, scalar_size: int = 4, mem_constant: float = 20) -> int:
        total_memory_possible = torch.cuda.get_device_properties(torch.device('cuda')).total_memory
        curr_mem_use = torch.cuda.memory_allocated()
        available_memory = total_memory_possible - curr_mem_use
        return int(available_memory / (mem_constant*n*scalar_size))

    def _sum_exp_pairwise(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if x.numel() == 0 or z.numel() == 0:
            empty = torch.zeros(
                (x.shape[0], z.shape[0]),
                dtype=x.dtype,
                device=x.device,
            )
            return empty, 0

        dim_count = x.shape[-1]
        sum_exp = kermac.cdist_expadd(
            x.contiguous(),
            z.contiguous(),
            q=float(self.exponent),
            ell=float(self.bandwidth),
        )

        if sum_exp.dim() == 3 and sum_exp.size(0) == 1:
            sum_exp = sum_exp.squeeze(0)
        if sum_exp.dim() == 2 and sum_exp.shape == (z.shape[0], x.shape[0]):
            sum_exp = sum_exp.transpose(0, 1).contiguous()

        return sum_exp, dim_count

    def _finalize_kernel(self, sum_exp: torch.Tensor, dim_count: int) -> torch.Tensor:
        if dim_count <= 0:
            raise ValueError("SumPower kernel requires at least one feature dimension.")
        mix = (1.0 - self.const_mix) / float(dim_count)
        kernel = self.const_mix + mix * sum_exp
        return kernel.clamp_min(0.0).pow(self.power)

    def _get_kernel_matrix_impl(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mat: Optional[torch.Tensor] = None,
        dist_mat: Optional[Tuple[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        if dist_mat is not None:
            sum_exp, dim_count = dist_mat
        else:
            xm = self._transform_m(x, mat)
            zm = self._transform_m(z, mat)
            sum_exp, dim_count = self._sum_exp_pairwise(xm, zm)
        return self._finalize_kernel(sum_exp, dim_count)

    def _get_dist_matrix_categorical(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mat: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        numerical_indices = getattr(self, "numerical_indices", None)
        categorical_indices = getattr(self, "categorical_indices", None)
        categorical_vectors = getattr(self, "categorical_vectors", None)

        if (numerical_indices is None or len(numerical_indices) == 0) and (
            categorical_indices is None or len(categorical_indices) == 0
        ):
            raise ValueError("No numerical or categorical features provided.")

        sum_exp = None
        dim_count = 0

        if numerical_indices is not None and len(numerical_indices) > 0:
            mat_num = get_sub_matrix(mat, numerical_indices)
            x_num = self._transform_m(x[:, numerical_indices], mat_num)
            z_num = self._transform_m(z[:, numerical_indices], mat_num)
            numeric_sum, numeric_dims = self._sum_exp_pairwise(x_num, z_num)
            sum_exp = numeric_sum
            dim_count += numeric_dims

        if categorical_indices is not None and categorical_vectors is not None:
            if len(categorical_indices) != len(categorical_vectors):
                raise ValueError("Number of categorical index and vector groups must match")
            for cat_idx, cat_vecs in zip(categorical_indices, categorical_vectors):
                mat_cat = get_sub_matrix(mat, cat_idx)
                transformed_vecs = self._transform_m(cat_vecs, mat_cat)
                cat_sum, cat_dims = self._sum_exp_pairwise(transformed_vecs, transformed_vecs)
                if cat_dims == 0:
                    continue
                x_cat = x[:, cat_idx].argmax(dim=-1)
                z_cat = z[:, cat_idx].argmax(dim=-1)
                cat_contrib = cat_sum[x_cat[:, None], z_cat[None, :]]
                if sum_exp is None:
                    sum_exp = cat_contrib.clone()
                else:
                    sum_exp = sum_exp + cat_contrib
                dim_count += cat_dims

        if sum_exp is None:
            sum_exp = torch.zeros(
                (x.shape[0], z.shape[0]),
                dtype=x.dtype,
                device=x.device,
            )
        return sum_exp, dim_count

    def _get_kernel_matrix_categorical_impl(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mat: Optional[torch.Tensor] = None,
        dist_mat: Optional[Tuple[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        if dist_mat is None:
            sum_exp, dim_count = self._get_dist_matrix_categorical(x, z, mat)
        else:
            sum_exp, dim_count = dist_mat
        return self._finalize_kernel(sum_exp, dim_count)
 

    def _get_function_grad_impl(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        coefs: torch.Tensor,
        mat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gradient of the kernel
            k(x,z) = ( const_mix + ((1-const_mix)/d) * sum_j exp(-( |x_j - z_j| / ell )^q) )^p
        w.r.t. x, accumulated with coefficients 'coefs'.

        Returns: tensor with the same layout as the Lpq case
                (matches your current caller expectations).
        """

        # 1) Optional linear transform (same as in Lpq)
        xm = self._transform_m(x, mat)  # [K, N]
        zm = self._transform_m(z, mat)  # [M, N]

        # 2) Build u_{k,m} = const_mix + ((1-cmix)/d) * sum_j exp(-(|diff|/ell)^q)
        #    We compute it in PyTorch for the pairwise (k,m) matrix.
        #    Shapes: xm[:,None,:] -> [K,1,N]; zm[None,:,:] -> [1,M,N]; broadcast -> [K,M,N]
        diff = xm[:, None, :] - zm[None, :, :]                      # [K, M, N]
        r = diff.abs()                                              # [K, M, N]

        ell = float(self.bandwidth)
        q = float(self.exponent)
        p = float(self.power)

        exp_terms = torch.exp(-(r / ell).pow(q))                    # [K, M, N]
        sum_exp = exp_terms.sum(dim=-1)                             # [K, M]

        d = xm.shape[-1]                                            # feature dimension
        cmix = self.const_mix

        u = cmix + (1.0 - cmix) * (sum_exp / float(d))              # [K, M]

        # 3) Pairwise prefactor a_{k,m} (no coordinate dependence)
        #    a[k,m] = - p * u^{p-1} * (1-cmix)/d * q / ell^q
        a_mat = (
            -p
            * u.clamp_min(1e-24).pow(p - 1)
            * (1.0 - cmix)
            / float(d)
            * (q / (ell ** q))
        )  # [K, M]

        # 4) Call the specialized grad op that handles per-coordinate factor:
        #       exp(-(|diff|/ell)^q) * |diff|^{q-1} * sign(diff)
        #    Shapes expected by the low-level op match your cdist_grad:
        #       a[K,M] = a_mat
        #       b[N,K] = xm^T
        #       c[O,K] = coefs
        #       d[N,M] = zm^T
        #    It also receives q, ell, and eps for numerical stability.
        b = xm.T.contiguous()  # [N, K]
        d_ = zm.T.contiguous() # [N, M]

        out = kermac.cdist_grad_expadd(
            a_mat.contiguous(),   # [K, M]
            b,                    # [N, K]
            coefs,                # [O, K]
            d_,                   # [N, M]
            q=float(q),
            ell=float(ell),
            eps=float(getattr(self, "eps", 0.0)),
        )  # -> [O, N, M]

        return out.transpose(-2, -1).float()  # [O, M, N] -> match Lpq caller


    
if __name__ == '__main__':
    # kernel = LaplaceKernel(bandwidth=2.0, exponent=1.0)
    kernel = KermacProductLaplaceKernel(bandwidth=2.0, exponent=1.2)

    n_samples = 2000
    n_features = 100
    x = torch.rand(n_samples, n_features)
    coefs = torch.rand(1, n_samples)
    kernel.get_agop(x, x, coefs)

    print('here')

    import matplotlib.pyplot as plt

    x = torch.linspace(-2.0, 2.0, 5)[:, None]
    z = torch.linspace(-4.0, 4.0, 500)[:, None]
    coefs = torch.as_tensor([[1.0, 0.8, 0.4, -0.5, -2.0], [0.1, 0.2, 0.3, 0.4, 0.5]])
    # mat = None
    mat = torch.as_tensor([0.5])
    # mat = torch.as_tensor([[0.5]])
    f_values = coefs[0, :] @ kernel.get_kernel_matrix(x, z, mat)
    plt.plot(z[:, 0], f_values, 'tab:blue', label='function')
    plt.plot(z, kernel.get_function_grads(x, z, coefs, mat)[0], 'tab:orange', label='gradient')
    plt.plot(0.5 * (z[1:, 0] + z[:-1, 0]), (f_values[1:] - f_values[:-1]) / (z[1:, 0] - z[:-1, 0]), color='tab:green',
             linestyle='--', label='finite diff')
    plt.legend()
    plt.show()
