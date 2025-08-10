import numpy as np
import torch

from xrfm import xRFM

import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

np.random.seed(0)
torch.manual_seed(0)

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

def accuracy(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).float().mean()

def one_hot_encoding(y, device='cuda'):
    y = torch.from_numpy(y).long()-1
    return torch.zeros(len(y), 7).scatter_(1, y.unsqueeze(1), 1).to(device)

X, y = fetch_covtype(data_home='/projects/bbjr/dbeaglehole/', return_X_y=True, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=0)


max_n_train = 100_000
max_n_val = 50_000
min_subset_size = 25_000

X_train = torch.from_numpy(X_train[:max_n_train]).float().cuda()
y_train = one_hot_encoding(y_train[:max_n_train])

X_val = torch.from_numpy(X_val[:max_n_val]).float().cuda()
y_val = one_hot_encoding(y_val[:max_n_val])

X_test = torch.from_numpy(X_test).float().cuda()
y_test = one_hot_encoding(y_test)

print(f'X_train.shape: {X_train.shape}')
print(f'X_val.shape: {X_val.shape}')
print(f'X_test.shape: {X_test.shape}')

print(f'y_train.shape: {y_train.shape}')
print(f'y_val.shape: {y_val.shape}')
print(f'y_test.shape: {y_test.shape}')

DEVICE = torch.device("cuda")
bw = 5.
reg = 1e-3
iters = 1

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
xrfm_model = xRFM(xrfm_params, device=DEVICE, min_subset_size=min_subset_size, tuning_metric='accuracy', 
                  default_rfm_params=default_rfm_params, 
                  split_method='top_vector_agop_on_subset')



start_time = time.time()
xrfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = xrfm_model.predict_proba(X_test)
acc = accuracy(y_pred, y_test)
print(f'xRFM time: {end_time-start_time:g} s, acc: {acc:g}')
print('-'*150)