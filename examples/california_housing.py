import numpy as np
import torch
from tabrfm import TabRFM
from tabrfm.kernels import LaplaceKernel, ProductLaplaceKernel
from tabrfm.experimental.rp_rfm import RP_RFM
import time
import sklearn
from sklearn.metrics import accuracy_score

np.random.seed(0)
torch.manual_seed(0)

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

def accuracy(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).float().mean()

def one_hot_encoding(y, device='cuda'):
    y = torch.from_numpy(y).long()-1
    return torch.zeros(len(y), 7).scatter_(1, y.unsqueeze(1), 1).to(device)

X, y = sklearn.datasets.fetch_california_housing(data_home='/projects/bbjr/dbeaglehole/', return_X_y=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_test, y_test, test_size=0.2, random_state=0)


max_n_train = 70_000
max_n_val = 20_000

X_train = torch.from_numpy(X_train[:max_n_train]).float().cuda()
y_train = torch.from_numpy(y_train[:max_n_train]).float().cuda().unsqueeze(1)

X_val = torch.from_numpy(X_val[:max_n_val]).float().cuda()
y_val = torch.from_numpy(y_val[:max_n_val]).float().cuda().unsqueeze(1)

X_test = torch.from_numpy(X_test).float().cuda()
y_test = torch.from_numpy(y_test).float().cuda().unsqueeze(1)

print(f'X_train.shape: {X_train.shape}')
print(f'X_val.shape: {X_val.shape}')
print(f'X_test.shape: {X_test.shape}')

print(f'y_train.shape: {y_train.shape}')
print(f'y_val.shape: {y_val.shape}')
print(f'y_test.shape: {y_test.shape}')

############################ RP-RFM #############################
DEVICE = torch.device("cuda")
bw = 5.
reg = 1e-3
iters = 3
kernel_obj = ProductLaplaceKernel(bandwidth=bw, exponent=1.0)
rfm_params = {
    "kernel": kernel_obj,
    "diag": False,
    "reg": reg,
    "iters": iters,
    "M_batch_size": len(X_train),
    "verbose": True,
}
model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=15_000, n_trees=3, n_tree_iters=0, tuning_metric='mse', split_method='linear')

start_time = time.time()

model.fit(X_train, y_train, X_val, y_val)

end_time = time.time()

print("Predicting on test set")
y_pred = model.predict_proba(X_test)
print(y_pred.shape)
print(y_test.shape)
mse = mse_loss(y_pred, y_test)
print(f'RP-RFM time: {end_time-start_time:g} s, mse: {mse.item():g}')


############################ TABRFM #############################
DEVICE = torch.device("cuda")
model = TabRFM(kernel_obj, diag=False, device=DEVICE, tuning_metric='mse')

# X_train_subset = X_train[:70_000]
# y_train_subset = y_train[:70_000]
# X_val_subset = X_val[:20_000]
# y_val_subset = y_val[:20_000]
X_train_subset = X_train
y_train_subset = y_train
X_val_subset = X_val
y_val_subset = y_val

start_time = time.time()

model.fit(
    (X_train_subset, y_train_subset), 
    (X_val_subset, y_val_subset), 
    iters=iters,
    M_batch_size=len(X_train_subset),
    reg=reg,
)

end_time = time.time()

y_pred = model.predict_proba(X_test)
mse = mse_loss(y_pred, y_test)
print(f'Generic time: {end_time-start_time:g} s, mse: {mse.item():g}')
