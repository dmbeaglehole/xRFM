import numpy as np
import torch
from tabrfm import TabRFM
from tabrfm.kernels import LaplaceKernel, ProductLaplaceKernel
from tabrfm.experimental.rp_rfm import RP_RFM
import time
import sklearn
from sklearn.metrics import accuracy_score
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


max_n_train = 600_000
max_n_val = 200_000

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

############################ RP-RFM #############################
DEVICE = torch.device("cuda")
bw = 5.
reg = 1e-3
iters = 0
rfm_params = {
    "kernel": LaplaceKernel(bandwidth=bw, exponent=1.0),
    "diag": False,
    "reg": reg,
    "iters": iters,
    "M_batch_size": len(X_train),
    "verbose": True,
}
model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=20_000, n_trees=1, n_tree_iters=0, tuning_metric='accuracy', split_method='top_vector_agop_on_subset')

start_time = time.time()
model.fit(X_train, y_train, X_val, y_val)
end_time = time.time()

print("Predicting on test set")
y_pred = model.predict_proba(X_test)
print(y_pred.shape)
print(y_test.shape)
acc = accuracy(y_pred, y_test)
print(f'RP-RFM, top vector AGOP splitting time: {end_time-start_time:g} s, acc: {acc.item():g}')
print("--------------------------------")
rfm_params = {
    "kernel": LaplaceKernel(bandwidth=bw, exponent=1.0),
    "diag": False,
    "reg": reg,
    "iters": iters,
    "M_batch_size": len(X_train),
    "verbose": False,
    "early_stop_rfm": True,
}
model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=20_000, n_trees=1, n_tree_iters=0, tuning_metric='accuracy', split_method='top_pc_agop_on_subset')#'top_vector_agop_on_subset')

start_time = time.time()
model.fit(X_train, y_train, X_val, y_val)
end_time = time.time()

y_pred = model.predict(X_test)
acc = accuracy(y_pred, y_test)
print(f'Deterministic recursive AGOP splitting RP-RFM, top PC AGOP splitting time 0 iters: {end_time-start_time:g} s, acc: {acc.item():g}')
print("--------------------------------")

# ############################ TABRFM #############################
# DEVICE = torch.device("cuda")
# model = TabRFM(LaplaceKernel(bandwidth=bw, exponent=1.0), diag=False, device=DEVICE, tuning_metric='accuracy')

# X_train_subset = X_train[:70_000]
# y_train_subset = y_train[:70_000]
# X_val_subset = X_val[:20_000]
# y_val_subset = y_val[:20_000]
# # X_train_subset = X_train
# # y_train_subset = y_train
# # X_val_subset = X_val
# # y_val_subset = y_val

# start_time = time.time()

# model.fit(
#     (X_train_subset, y_train_subset), 
#     (X_val_subset, y_val_subset), 
#     iters=iters,
#     M_batch_size=len(X_train_subset),
#     reg=reg,
# )

# end_time = time.time()

# y_pred = model.predict_proba(X_test)
# acc = accuracy(y_pred, y_test)
# print(f'Generic time: {end_time-start_time:g} s, acc: {acc.item():g}')


############################ XGBOOST #############################
# import xgboost as xgb
# # Initialize the XGBoost classifier with verbose parameters
# model = xgb.XGBClassifier(
#     n_estimators=100,            # Number of gradient boosted trees
#     learning_rate=0.1,           # Step size shrinkage to prevent overfitting
#     max_depth=3,                 # Maximum tree depth
#     min_child_weight=1,          # Minimum sum of instance weight needed in a child
#     subsample=1,                 # Subsample ratio of training instances
#     colsample_bytree=1,          # Subsample ratio of columns when constructing each tree
#     objective='multi:softprob',  # Multiclass probability output
#     verbosity=2                  # Verbose output (0=silent, 1=warning, 2=info, 3=debug)
# )

# # Fit the model with verbose output
# print("Starting model training...")
# eval_set = [(X_val.cpu().numpy(), y_val.argmax(dim=1).cpu().numpy())]

# model.fit(
#     X_train.cpu().numpy(),
#     y_train.argmax(dim=1).cpu().numpy(),
#     eval_set=eval_set,                   # Validation data for evaluation
#     verbose=True,                        # Print progress messages
# )

# # Make predictions with verbose output
# print("\nMaking predictions...")
# y_pred = model.predict(X_test.cpu().numpy())

# # Evaluate model performance
# accuracy = accuracy_score(y_test.argmax(dim=1).cpu().numpy(), y_pred)
# print(f"Accuracy: {accuracy:.4f}")
