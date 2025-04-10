import numpy as np
import torch
from rfm import LaplaceRFM, GeneralizedLaplaceRFM
from rfm.generic_kernels import LaplaceKernel, ProductLaplaceKernel, LpqLaplaceKernel
from rfm.generic_kernels import SumPowerLaplaceKernel
from rfm.recursive_feature_machine import GenericRFM
import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 256


n = 2000 # samples
d = 10  # dimension
n_cats = [200, 200]

def fstar(X):
    return (X[:,0]**2)[:,None].float()


bw = 20.
reg = 1e-4
iters = 3
num_to_sample = 128

X_train = torch.randn(n, d).cuda()
X_test = torch.randn(n, d).cuda()

train_cats = [torch.randint(0, n_cat, (n,)) for n_cat in n_cats]
test_cats = [torch.randint(0, n_cat, (n,)) for n_cat in n_cats]

X_train = torch.cat([X_train, torch.zeros(n, sum(n_cats), device=X_train.device)], dim=1)
X_test = torch.cat([X_test, torch.zeros(n, sum(n_cats), device=X_test.device)], dim=1)
for i, (train_cat, test_cat) in enumerate(zip(train_cats, test_cats)):
    X_train[torch.arange(n), train_cat + d + sum(n_cats[:i])] = 1
    X_test[torch.arange(n), test_cat + d + sum(n_cats[:i])] = 1

# print(f'X_train.shape: {X_train.shape}')
# print(f'X_test.shape: {X_test.shape}')

# print(f'X_train: {X_train[:2,-5:]}')
y_train = fstar(X_train).cuda()
y_test = fstar(X_test).cuda()

numerical_indices = torch.arange(d, device=X_train.device)
categorical_vectors = [torch.eye(n_cat, device=X_train.device)-1/n_cat for n_cat in n_cats]
categorical_indices = []
for n_cat in n_cats:
    categorical_indices.append(torch.arange(n_cat, device=X_train.device) + d)
    d += n_cat

# print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
d=700
a = torch.randn(d, d)
a = a@a.T
a = a.float()

from scipy.linalg import sqrtm, fractional_matrix_power
M_cpu = torch.randn(d, d)
M_cpu = M_cpu@M_cpu.T
M_cpu = M_cpu.float()
start_time = time.time()
M_cpu = torch.from_numpy(fractional_matrix_power(M_cpu, 0.67))
tot = time.time()-start_time
print(f'Time taken: {tot:g} s')
print(M_cpu)
exit()
# import adit_rfm
# start_time = time.time()
# model = adit_rfm.rfm(
#     (X_train, y_train), 
#     (X_test, y_test), 
#     L=bw, 
#     reg=reg, 
#     num_iters=iters
# )
# print(f'adit_rfm time: {time.time()-start_time:g} s')


# model = GenericRFM(LaplaceKernel(bandwidth=bw, exponent=1.0), reg=reg, device='cuda')
# # model = GenericRFM(ProductLaplaceKernel(bandwidth=bw, exponent=1.0), diag=False, reg=reg, device='cuda', centering=True)
# model.set_categorical_indices(numerical_indices, categorical_indices, categorical_vectors)
# start_time = time.time()
# model.fit(
#     (X_train, y_train), 
#     (X_test, y_test), 
#     iters=iters,
#     classification=False,
#     M_batch_size=len(X_train),
#     total_points_to_sample=5000
# )
# print(f'Generic time: {time.time()-start_time:g} s')


# model = LaplaceRFM(bandwidth=bw, reg=reg, device='cuda')
# start_time = time.time()
# model.fit(
#     (X_train, y_train), 
#     (X_test, y_test), 
#     iters=iters,
#     classification=False,
#     M_batch_size=len(X_train),
#     total_points_to_sample=5000,
#     verbose=False
# )
# print(f'Laplace time: {time.time()-start_time:g} s')

# model = GenericRFM(LaplaceKernel(bandwidth=bw, exponent=1.0), reg=reg, device='cuda')
model = GenericRFM(ProductLaplaceKernel(bandwidth=bw, exponent=1.0), diag=True, reg=reg, device='cuda')
model.set_categorical_indices(numerical_indices, categorical_indices, categorical_vectors)
start_time = time.time()
model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=iters,
    classification=False,
    M_batch_size=128,
    total_points_to_sample=num_to_sample,
    verbose=True
)
# print("M matrix:")
# print(model.M)
print(f'Generic time new: {time.time()-start_time:g} s')

print("="*100)
model = GenericRFM(ProductLaplaceKernel(bandwidth=bw, exponent=1.0), diag=True, reg=reg, device='cuda')
start_time = time.time()
model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=iters,
    classification=False,
    M_batch_size=128,
    total_points_to_sample=num_to_sample,
    verbose=True
)
# print("M matrix:")
# print(model.M)
print(f'Generic time: {time.time()-start_time:g} s')