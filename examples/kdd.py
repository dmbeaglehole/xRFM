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

def one_hot_encoding(y, num_classes, device='cuda'):
    y = torch.from_numpy(y).long()
    return torch.zeros(len(y), num_classes).scatter_(1, y.unsqueeze(1), 1).to(device)


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
import numpy as np
from collections import Counter

def prune_identical_samples(X, y):
    # Get unique indices while preserving order
    unique_indices = []
    seen_samples = set()
    
    for i, sample in enumerate(X):
        # Convert sample to a hashable type (tuple)
        sample_tuple = tuple(sample)
        if sample_tuple not in seen_samples:
            unique_indices.append(i)
            seen_samples.add(sample_tuple)
    
    # Return only unique samples
    return X[unique_indices], y[unique_indices]

# Load the KDDCup99 dataset
X, y = sklearn.datasets.fetch_kddcup99(data_home='/projects/bbjr/dbeaglehole/', return_X_y=True, shuffle=True, percent10=True)
# X, y = prune_identical_samples(X, y)

num_classes = len(np.unique(y))
# Print key information about the dataset
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {num_classes}")
print(f"Class Distributions: {Counter(y)}")

# Encode categorical variables
nominal = [1, 2, 3]
transformer = ColumnTransformer(transformers=[('ordinal', OrdinalEncoder(), nominal)], remainder='passthrough')
# Perform ordinal encoding
X = transformer.fit_transform(X)


normalizer = StandardScaler()
X = normalizer.fit_transform(X)

# Encode target variable
y = LabelEncoder().fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.2)

max_n_train = 1_000_000
max_n_val = 200_000

X_train = torch.from_numpy(X_train[:max_n_train].astype(np.float32)).float().cuda()
y_train = one_hot_encoding(y_train[:max_n_train], num_classes).float().cuda()


X_val = torch.from_numpy(X_val[:max_n_val].astype(np.float32)).float().cuda()
y_val = one_hot_encoding(y_val[:max_n_val], num_classes).float().cuda()

X_test = torch.from_numpy(X_test.astype(np.float32)).float().cuda()
y_test = one_hot_encoding(y_test, num_classes).float().cuda()

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
kernel_obj = LaplaceKernel(bandwidth=bw, exponent=1.0)
metric = 'accuracy'
rfm_params = {
    "kernel": kernel_obj,
    "diag": False,
    "reg": reg,
    "iters": iters,
    "M_batch_size": len(X_train),
    "verbose": True,
}
model = RP_RFM(rfm_params, device=DEVICE, min_subset_size=20_000, n_trees=2, n_tree_iters=0, tuning_metric=metric)

start_time = time.time()

model.fit(X_train, y_train, X_val, y_val)

end_time = time.time()

print("Predicting on test set")
y_pred = model.predict_proba(X_test)
print(y_pred.shape)
print(y_test.shape)
mse = mse_loss(y_pred, y_test)
acc = accuracy(y_pred, y_test)
print(f'RP-RFM time: {end_time-start_time:g} s, mse: {mse.item():g}, acc: {acc.item():g}')


############################ TABRFM #############################
DEVICE = torch.device("cuda")
model = TabRFM(kernel_obj, diag=False, device=DEVICE, tuning_metric=metric)

X_train_subset = X_train[:70_000]
y_train_subset = y_train[:70_000]
X_val_subset = X_val[:20_000]
y_val_subset = y_val[:20_000]
# X_train_subset = X_train
# y_train_subset = y_train
# X_val_subset = X_val
# y_val_subset = y_val

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
acc = accuracy(y_pred, y_test)
print(f'Generic time: {end_time-start_time:g} s, acc: {acc.item():g}, mse: {mse.item():g}')


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
