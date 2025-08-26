import numpy as np
import torch

from xrfm import xRFM

import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 2187

def fstar(X):
    return (X[:, 0] ** 2 + 0.5*X[:, 1])[:,None].float()

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

n = 2187 #19115 # samples
ntest = 313
d = 50 #8036  # dimension

bw = 10.
reg = 1e-3
iters = 3
min_subset_size = 19115
exponent = 1.1
p = 1.1
assert 0 < exponent <= p

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.randn(n, d).to(DEVICE)
X_test = torch.randn(ntest, d).to(DEVICE)
y_train = fstar(X_train).to(DEVICE)
y_test = fstar(X_test).to(DEVICE)

# print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
# print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

xrfm_params = {
    'model': {
        'kernel': "lpq_kermac", #l1_kermac", #"lpq_kermac", #"l1",
        'bandwidth': bw,
        'exponent': exponent,
        'norm_p': p,
        'diag': False,
        'bandwidth_mode': "constant"
    },
    'fit': {
        'reg': reg,
        'iters': iters,
        'M_batch_size': M_batch_size,
        'verbose': True,
        'early_stop_rfm': True,
    }
}


xrfm_model = xRFM(xrfm_params, device=DEVICE, min_subset_size=min_subset_size, tuning_metric='mse', 
                  split_method='top_vector_agop_on_subset')

start_time = time.time()
xrfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = torch.from_numpy(xrfm_model.predict(X_test.detach().cpu().numpy())).to(DEVICE)
loss = mse_loss(y_pred, y_test)
print(f'xRFM time: {end_time-start_time:g} s, loss: {loss.item():g}')
print('-'*150)
