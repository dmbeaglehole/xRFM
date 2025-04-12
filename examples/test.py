import numpy as np
import torch
from xrfm import xRFM
import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 256

def fstar(X):
    return (X[:, 0] ** 2 + 0.5*X[:, 1])[:,None].float()

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()

n = 20_000 # samples
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
    "kernel": 'l1',
    "bandwidth": bw,
    "exponent": 1.0,
    "diag": False,
    "reg": reg,
    "iters": iters,
    "M_batch_size": len(X_train),
    "verbose": False,
    "early_stop_rfm": True,
    'bandwidth_mode': 'adaptive'
}
model = xRFM(rfm_params, device=DEVICE, min_subset_size=2_300, tuning_metric='mse')

start_time = time.time()
model.fit(X_train, y_train, X_test, y_test)
end_time = time.time()

y_pred = model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'xRFM time: {end_time-start_time:g} s, loss: {loss.item():g}')


state_dict = model.get_state_dict()

# print(state_dict['param_trees'])
def get_tree_attributes(tree, attr='train_indices'):
    if tree['type'] == 'leaf':
        return print(tree[attr].shape)
    else:
        get_tree_attributes(tree['left'], attr)
        get_tree_attributes(tree['right'], attr)
get_tree_attributes(state_dict['param_trees'][0])

new_model = xRFM()
new_model.load_state_dict(state_dict, X_train)

y_pred = new_model.predict(X_test)
loss = mse_loss(y_pred, y_test)
print(f'Loaded xRFM loss: {loss.item():g}')

