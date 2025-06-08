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


DEVICE = torch.device("cuda")
xrfm_params = {
    'model': {
        'kernel': "l1",
        'bandwidth': bw,
        'exponent': 1.0,
        'diag': False,
        'bandwidth_mode': "constant"
    },
    'fit': {
        'reg': reg,
        'iters': iters,
        'M_batch_size': len(X_train),
        'verbose': False,
        'early_stop_rfm': True,
    }
}
default_rfm_params = {
    'model': {
        "kernel": 'l2_high_dim',
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
xrfm_model = xRFM(xrfm_params, device=DEVICE, min_subset_size=min_subset_size, tuning_metric='mse', 
                  default_rfm_params=default_rfm_params, 
                  split_method='fixed_vector', 
                  fixed_vector=torch.ones(d).cuda().to(X_train.dtype))



start_time = time.time()
xrfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = xrfm_model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'xRFM time: {end_time-start_time:g} s, loss: {loss.item():g}')
print('-'*150)

rfm_params = {**xrfm_params['model'], **xrfm_params['fit']}
rp_rfm_model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=min_subset_size, 
                      tuning_metric='mse', n_tree_iters=0, n_trees=1,
                      split_method='fixed_vector', 
                      fixed_vector=torch.ones(d).cuda().to(X_train.dtype))

start_time = time.time()
rp_rfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = rp_rfm_model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'RP-RFM time: {end_time-start_time:g} s, loss: {loss.item():g}')
print('-'*150)