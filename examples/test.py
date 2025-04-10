import numpy as np
import torch
from tabrfm import TabRFM
from tabrfm.kernels import LaplaceKernel
from tabrfm.experimental.rp_rfm import RP_RFM
import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 256

def fstar(X):
    return (X[:, 0] ** 2 + 0.5*X[:, 1])[:,None].float()

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

n = 125_000 # samples
ntest = 5000
d = 500  # dimension

bw = 20.
reg = 1e-8
iters = 3

X_train = torch.randn(n, d).cuda()
X_test = torch.randn(ntest, d).cuda()
y_train = fstar(X_train).cuda()
y_test = fstar(X_test).cuda()

print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

DEVICE = torch.device("cuda")
rfm_params = {
    "kernel": LaplaceKernel(bandwidth=bw, exponent=1.0),
    "diag": False,
    "reg": reg,
    "iters": iters,
    "M_batch_size": len(X_train),
    "verbose": False,
    "early_stop_rfm": True,
}
model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=20_000, n_trees=1, n_tree_iters=1, tuning_metric='mse')

start_time = time.time()

model.fit(X_train, y_train, X_test, y_test)

end_time = time.time()

y_pred = model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'RP-RFM time: {end_time-start_time:g} s, loss: {loss.item():g}')

print("--------------------------------")
model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=20_000, n_trees=1, n_tree_iters=0, tuning_metric='mse')

start_time = time.time()

model.fit(X_train, y_train, X_test, y_test)

end_time = time.time()

y_pred = model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'RP-RFM time 0 iters: {end_time-start_time:g} s, loss: {loss.item():g}')
print("--------------------------------")



DEVICE = torch.device("cuda")
model = TabRFM(LaplaceKernel(bandwidth=bw, exponent=1.0), diag=False, device=DEVICE, early_stop_rfm=True)

start_time = time.time()

model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=iters,
    classification=False,
    M_batch_size=len(X_train),
    reg=reg,
)

end_time = time.time()

y_pred = model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'Generic time: {end_time-start_time:g} s, loss: {loss.item():g}')

