'''Helper functions.'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def device_from_str(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

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
    
def stable_matrix_power(M, power, MAX_DIMENSIONS_FOR_SVD=5000):
    """
    Compute the power of a matrix.
    :param M: Matrix to power.
    :param power: Power to raise the matrix to.
    :return: Matrix raised to the power - M^{power}.
    """
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"

        # Handle NaNs
        if torch.isnan(M).all():
            print("All NaNs in matrix, returning identity")
            return torch.eye(M.shape[0], device=M.device, dtype=M.dtype)

        if torch.isnan(M).any():
            print("Some NaNs in matrix, replacing with 0")
            M = torch.nan_to_num(M, nan=0.0, posinf=1e12, neginf=-1e12)
            # Optional: scale to a reasonable magnitude
            scale = M.abs().max()
            if scale > 0:
                M = M / scale

        M.diagonal().add_(1e-8)
        if M.shape[0] < MAX_DIMENSIONS_FOR_SVD:
            print("Using SVD")
            U, S, _ = torch.linalg.svd(M)
        else:
            print("Using SVD lowrank with q=", MAX_DIMENSIONS_FOR_SVD)
            print("M.shape", M.shape)
            start_time = time.time()
            U, S, _ = torch.svd_lowrank(M, q=MAX_DIMENSIONS_FOR_SVD)
            end_time = time.time()
            print(f"Time taken for SVD lowrank: {end_time - start_time} seconds")

        S[S<0] = 0.
        return (U @ torch.diag(S**power) @ U.T).to(device=M.device, dtype=M.dtype)

    elif len(M.shape) == 1:
        # Handle NaNs
        if torch.isnan(M).all():
            print("All NaNs in vector, returning all ones")
            return torch.ones(M.shape[0], device=M.device, dtype=M.dtype)

        if torch.isnan(M).any():
            print("Some NaNs in vector, replacing with 0")
            M = torch.nan_to_num(M, nan=0.0, posinf=1e12, neginf=-1e12)
            # Optional: scale to a reasonable magnitude
            scale = M.abs().max()
            if scale > 0:
                M = M / scale

        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")
