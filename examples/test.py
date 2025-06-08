import numpy as np
import torch

import sys
sys.path.append('../xrfm')
from xrfm import xRFM

import sys
sys.path.insert(0, '/u/dbeaglehole/tabrfm')
from tabrfm.experimental.rp_rfm import RP_RFM

import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 256

def fstar(X):
    return (X[:, 0] ** 2 + 0.5*X[:, 1])[:,None].float()

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

n = 2_000 # samples
ntest = 2_000
d = 50  # dimension

bw = 8.516821304578539
reg = 5.438790710761095e-05
iters = 5
min_subset_size = 500
exponent = 1.0385674510481542

X_train = torch.randn(n, d).cuda()
X_test = torch.randn(ntest, d).cuda()
y_train = fstar(X_train).cuda()
y_test = fstar(X_test).cuda()

# print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
# print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

DEVICE = torch.device("cuda")
xrfm_params = {
    'model': {
        'kernel': "l1",
        'bandwidth': bw,
        'exponent': exponent,
        'diag': True,
        'bandwidth_mode': "adaptive"
    },
    'fit': {
        'reg': reg,
        'iters': iters,
        'M_batch_size': len(X_train),
        'verbose': False,
        'early_stop_rfm': True,
    }
}
# default_rfm_params = {
#     'model': {
#         "kernel": 'l2_high_dim',
#         "exponent": 1.0,
#         "bandwidth": 10.0,
#         "diag": False,
#         "bandwidth_mode": "constant"
#     },
#     'fit' : {
#         "get_agop_best_model": True,
#         "return_best_params": False,
#         "reg": 1e-3,
#         "iters": 0,
#         "early_stop_rfm": False,
#         "verbose": False
#     }
# }
rfm_params = {**xrfm_params['model'], **xrfm_params['fit']}
rp_rfm_model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=min_subset_size, 
                      tuning_metric='mse', n_tree_iters=0, n_trees=1,
                      split_method='top_vector_agop_on_subset')
                    #   split_method='fixed_vector', 
                    #   fixed_vector=torch.ones(d).cuda().to(X_train.dtype))




start_time = time.time()
rp_rfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = rp_rfm_model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'RP-RFM time: {end_time-start_time:g} s, loss: {loss.item():g}')
print('-'*150)

xrfm_model = xRFM(xrfm_params, device=DEVICE, min_subset_size=min_subset_size, tuning_metric='mse', 
                  split_method='top_vector_agop_on_subset')
                    # default_rfm_params=default_rfm_params, 
                  #split_method='fixed_vector', 
                  #fixed_vector=torch.ones(d).cuda().to(X_train.dtype))

start_time = time.time()
xrfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = xrfm_model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'xRFM time: {end_time-start_time:g} s, loss: {loss.item():g}')
print('-'*150)




# rfm_params = {**xrfm_params['model'], **xrfm_params['fit']}
# rp_rfm_model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=min_subset_size, 
#                       tuning_metric='mse', n_tree_iters=0, n_trees=1,
#                       split_method='fixed_vector', 
#                       fixed_vector=torch.ones(d).cuda().to(X_train.dtype))

# start_time = time.time()
# rp_rfm_model.fit(X_train, y_train, X_test, y_test)
# end_time = time.time()

# y_pred = rp_rfm_model.predict(X_test)
# loss = mse_loss(y_pred, y_test)
# print(f'RP-RFM 2 time: {end_time-start_time:g} s, loss: {loss.item():g}')
# print('-'*150)

# xrfm_model = xRFM(xrfm_params, device=DEVICE, min_subset_size=min_subset_size, tuning_metric='mse', 
#                   default_rfm_params=default_rfm_params, 
#                   split_method='fixed_vector', 
#                   fixed_vector=torch.ones(d).cuda().to(X_train.dtype))

# start_time = time.time()
# xrfm_model.fit(X_train, y_train, X_test, y_test)
# end_time = time.time()

# y_pred = xrfm_model.predict(X_test)
# loss = mse_loss(y_pred, y_test)
# print(f'xRFM 2 time: {end_time-start_time:g} s, loss: {loss.item():g}')
# print('-'*150)

