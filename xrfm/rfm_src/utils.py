'''Helper functions.'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm, fractional_matrix_power

class SmoothClampedReLU(nn.Module):
    def __init__(self, beta=50):
        super(SmoothClampedReLU, self).__init__()
        self.beta = beta
        
    def forward(self, x):
        # Smooth transition at x=0 (using softplus with high beta)
        activated = F.softplus(x, beta=self.beta)
        # Smooth transition at x=1 (using sigmoid scaled and shifted)
        # As x approaches infinity, this approaches 1
        clamped = activated - F.softplus(activated - 1, beta=self.beta)
        
        return clamped

def f1_score(preds, targets, num_classes, min_float=1e-8):
    
    # Calculate F1 score components
    if num_classes == 2:
        preds = (preds >= 0.5).int()
        targets = targets.int()
        
        # get F1 on positive class
        tp = ((preds[:,1] == 1) & (targets[:,1] == 1)).sum().float()
        fp = ((preds[:,1] == 1) & (targets[:,1] == 0)).sum().float()
        fn = ((preds[:,1] == 0) & (targets[:,1] == 1)).sum().float()
    else:
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)
        
        for c in range(num_classes):
            tp[c] = ((preds == c) & (targets == c)).sum().float()
            fp[c] = ((preds == c) & (targets != c)).sum().float()
            fn[c] = ((preds != c) & (targets == c)).sum().float()
    
    # Avoid division by zero
    precision = tp / (tp + fp + min_float)
    recall = tp / (tp + fn + min_float)
    
    f1 = 2 * precision * recall / (precision + recall + min_float)
    
    # Return mean F1 score across all classes
    return f1.mean()

def float_x(data):
    '''Set data array precision.'''
    return np.float32(data)

def matrix_power(M, power):
    """
    Compute the power of a matrix.
    :param M: Matrix to power.
    :param power: Power to raise the matrix to.
    :return: Matrix raised to the power - M^{power}.
    """
    return stable_matrix_power(M, power)
    # if len(M.shape) == 2:
    #     assert M.shape[0] == M.shape[1], "Matrix must be square"

    #     # gpu square root
    #     S, U = torch.linalg.eigh(M)
    #     S[S<0] = 0.
    #     return U @ torch.diag(S**power) @ U.T
    # elif len(M.shape) == 1:
    #     assert M.shape[0] > 0, "Vector must be non-empty"
    #     M[M<0] = 0.
    #     return M**power
    # else:
    #     raise ValueError(f"Invalid matrix shape for square root: {M.shape}")
    
def stable_matrix_power(M, power):
    """
    Compute the power of a matrix.
    :param M: Matrix to power.
    :param power: Power to raise the matrix to.
    :return: Matrix raised to the power - M^{power}.
    """
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"
        if M.shape[0] < 700:
            M_cpu = M.cpu().float()
            M_cpu.diagonal().add_(1e-8)
            U, S, _ = torch.linalg.svd(M_cpu)
            S[S<0] = 0.
            return (U @ torch.diag(S**power) @ U.T).to(device=M.device, dtype=M.dtype)
        else:
            M.diagonal().add_(1e-8)
            S, U = torch.linalg.eigh(M)
            S[S<0] = 0.
            return (U @ torch.diag(S**power) @ U.T).to(device=M.device, dtype=M.dtype)

    elif len(M.shape) == 1:
        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")

