import numpy as np
import torch
import time

np.random.seed(0)
torch.manual_seed(0)

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

M_batch_size = 256


n = 4000 # samples
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



from xrfm import xRFM

import sys
sys.path.insert(0, '/u/dbeaglehole/tabrfm')
from tabrfm.experimental.rp_rfm import RP_RFM


DEVICE = torch.device("cuda")
bw = 5.
reg = 1e-3
iters = 1
min_subset_size = 1000

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

categorical_info = {
    'categorical_indices': categorical_indices,
    'categorical_vectors': categorical_vectors,
    'numerical_indices': numerical_indices,
}
xrfm_model = xRFM(xrfm_params, device=DEVICE, min_subset_size=min_subset_size, 
                  default_rfm_params=default_rfm_params, 
                  categorical_info=categorical_info,
                  split_method='top_vector_agop_on_subset')





rfm_params = {**xrfm_params['model'], **xrfm_params['fit']}
rp_rfm_model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=min_subset_size, 
                      n_tree_iters=0, n_trees=1,
                      categorical_info = categorical_info,
                      split_method='top_vector_agop_on_subset')

start_time = time.time()
rp_rfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = rp_rfm_model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'RP-RFM time: {end_time-start_time:g} s, loss: {loss.item():g}')
print('-'*150)

start_time = time.time()
xrfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = xrfm_model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'xRFM time: {end_time-start_time:g} s, loss: {loss.item():g}')
print('-'*150)