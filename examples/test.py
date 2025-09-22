import numpy as np
import torch

from xrfm import xRFM

import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 2187

def fstar(X):
    return (torch.pow(X[:, 0], 2) > 0.5)[:,None].float()

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

n = 1000 #19115 # samples
ntest = 313
d = 20 #8036  # dimension

tuning_metric = 'accuracy'
bw = 10.
reg = 1e-3
iters = 4
min_subset_size = 60000
exponent = 1.
p = 2.
assert 0 < exponent <= p

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.randn(n, d).to(DEVICE)
X_test = torch.randn(ntest, d).to(DEVICE)
y_train_clean = fstar(X_train).reshape(-1).cpu().numpy()
y_test_clean = fstar(X_test).reshape(-1).cpu().numpy()

# Add Bernoulli noise
noise_prob = 0.05  # 10% chance of flipping label
y_train = np.where(np.random.rand(n) < noise_prob, 
                      1 - y_train_clean, y_train_clean)
y_test = np.where(np.random.rand(ntest) < noise_prob, 
                     1 - y_test_clean, y_test_clean)         

print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')


y_train = np.expand_dims(y_train, -1)
y_test = np.expand_dims(y_test, -1)

xrfm_params = {
    'model': {
        'kernel': "l2", #l1_kermac", #"lpq_kermac", #"l1",
        'bandwidth': bw,
        'exponent': exponent,
        'norm_p': p,
        'diag': False,
        'bandwidth_mode': "constant",
    },
    'fit': {
        'solver': 'solve', #'log_reg', 'solve
        'reg': reg,
        # 'manual_reg_tuning': True,
        # 'manual_reg_tuning_range': [1e-6, 1e1],
        # 'manual_reg_tuning_steps': 10,
        'iters': iters,
        'M_batch_size': M_batch_size,
        'verbose': True,
        'early_stop_rfm': True,
    }
}


xrfm_model = xRFM(xrfm_params, device=DEVICE, min_subset_size=min_subset_size, 
                  tuning_metric=tuning_metric, 
                  split_method='top_vector_agop_on_subset')

start_time = time.time()
xrfm_model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_test = y_test.reshape(-1)
y_pred = xrfm_model.predict(X_test).reshape(-1)
print("y_pred[:5]", y_pred[:5])
acc = np.mean(y_pred == y_test)
print(f'xRFM time: {end_time-start_time:g} s, acc: {acc}')
print('-'*150)

y_pred = xrfm_model.predict_proba(X_test).argmax(axis=-1)
acc = np.mean(y_pred == y_test)
print(f'xRFM 2 acc: {acc:.4f}')
print('-'*150)

