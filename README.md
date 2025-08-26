# xRFM - Recursive Feature Machines optimized for tabular data


**xRFM** is a scalable implementation of Recursive Feature Machines (RFMs) optimized for tabular data. This library provides both the core RFM algorithm and a tree-based extension (xRFM) that enables efficient processing of large datasets through recursive data splitting.

## Core Components

```
xRFM/
├── xrfm/
│   ├── xrfm.py              # Main xRFM class (tree-based)
│   ├── tree_utils.py        # Tree manipulation utilities
│   └── rfm_src/
│       ├── recursive_feature_machine.py  # Base RFM class
│       ├── kernels.py       # Kernel implementations
│       ├── eigenpro.py      # EigenPro optimization
│       ├── utils.py         # Utility functions
│       ├── svd.py           # SVD operations
│       └── gpu_utils.py     # GPU memory management
├── examples/                # Usage examples
└── setup.py                # Package configuration
```

## Installation

```bash
pip install xrfm
```

Or to use the KermacProductLaplaceKernel, with CUDA-11 or CUDA-12:

```bash
pip install xrfm[cu11]
```

or 

```bash
pip install xrfm[cu12]
```

### Development Installation

```bash
git clone https://github.com/dmbeaglehole/xRFM.git
cd xRFM
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from xrfm import xRFM
from sklearn.model_selection import train_test_split

# Create synthetic data
def target_function(X):
    return torch.cat([
        (X[:, 0] > 0)[:, None], 
        (X[:, 1] < 0.5)[:, None]
    ], dim=1).float()

# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = xRFM(device=device, tuning_metric='mse')

# Generate data
n_samples = 2000
n_features = 100
X = torch.randn(n_samples, n_features, device=device)
y = target_function(X)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=0)

model.fit(X_train, y_train, X_val, y_val)
y_pred_test = model.predict(X_test)
```

### Custom Configuration

```python
# Custom RFM parameters
rfm_params = {
    'model': {
        'kernel': 'l2',           # Kernel type
        'bandwidth': 5.0,         # Kernel bandwidth
        'exponent': 1.0,          # Kernel exponent
        'diag': False,            # Diagonal Mahalanobis matrix
        'bandwidth_mode': 'constant'
    },
    'fit': {
        'reg': 1e-3,              # Regularization parameter
        'iters': 5,               # Number of iterations
        'M_batch_size': 1000,     # Batch size for AGOP
        'verbose': True,          # Verbose output
        'early_stop_rfm': True    # Early stopping
    }
}

# Initialize model with custom parameters
model = xRFM(
    rfm_params=rfm_params,
    device=device,
    min_subset_size=10000,        # Minimum subset size for splitting
    tuning_metric='accuracy',     # Tuning metric
    split_method='top_vector_agop_on_subset'  # Splitting strategy
)
```

## File Structure

### Core Files

| File | Description |
|------|-------------|
| `xrfm/xrfm.py` | Main xRFM class implementing tree-based recursive splitting |
| `xrfm/rfm_src/recursive_feature_machine.py` | Base RFM class with core algorithm |
| `xrfm/rfm_src/kernels.py` | Kernel implementations (Laplace, Product Laplace, etc.) |
| `xrfm/rfm_src/eigenpro.py` | EigenPro optimization for large-scale training |
| `xrfm/rfm_src/utils.py` | Utility functions for matrix operations and metrics |
| `xrfm/rfm_src/svd.py` | SVD utilities for kernel computations |
| `xrfm/rfm_src/gpu_utils.py` | GPU memory management utilities |
| `xrfm/tree_utils.py` | Tree manipulation and parameter extraction utilities |

### Example Files

| File | Description |
|------|-------------|
| `examples/test.py` | Simple regression example with synthetic data |
| `examples/covertype.py` | Forest cover type classification example |

## API Reference

### Main Classes

#### `xRFM`
Tree-based Recursive Feature Machine for scalable learning.

**Constructor Parameters:**
- `rfm_params` (dict): Parameters for base RFM models
- `min_subset_size` (int, default=60000): Minimum subset size for splitting
- `max_depth` (int, default=None): Maximum tree depth
- `device` (str, default=None): Computing device ('cpu' or 'cuda')
- `tuning_metric` (str, default='mse'): Metric for model tuning
- `split_method` (str): Data splitting strategy

**Key Methods:**
- `fit(X, y, X_val, y_val)`: Train the model
- `predict(X)`: Make predictions
- `predict_proba(X)`: Predict class probabilities
- `score(X, y)`: Evaluate model performance

#### `RFM`
Base Recursive Feature Machine implementation.

**Constructor Parameters:**
- `kernel` (str or Kernel): Kernel type or kernel object
- `iters` (int, default=5): Number of training iterations
- `bandwidth` (float, default=10.0): Kernel bandwidth
- `device` (str, default=None): Computing device
- `tuning_metric` (str, default='mse'): Evaluation metric

### Available Kernels

| Kernel | String ID | Description |
|--------|-----------|-------------|
| `LaplaceKernel` | `'laplace'`, `'l2'` | Standard Laplace kernel |
| `LightLaplaceKernel` | `'l2_high_dim'`, `'l2_light'` | Memory-efficient Laplace kernel |
| `ProductLaplaceKernel` | `'product_laplace'`, `'l1'` | Product of Laplace kernels |
| `SumPowerLaplaceKernel` | `'sum_power_laplace'`, `'l1_power'` | Sum of powered Laplace kernels |
| `KermacProductLaplaceKernel` | `'l1_kermac'` | High-performance Product of Laplace kernels (requires install with `[cu11]` or `[cu12]`) |
| `KermacLpqLaplaceKernel` | `'lpq_kermac'` | High-performance p-norm, q-exponent Laplace kernels (requires install with `[cu11]` or `[cu12]`) |


### Splitting Methods

| Method | Description |
|--------|-------------|
| `'top_vector_agop_on_subset'` | Use top eigenvector of AGOP matrix |
| `'random_agop_on_subset'` | Use random eigenvector of AGOP matrix |
| `'top_pc_agop_on_subset'` | Use top principal component of AGOP |
| `'random_pca'` | Use vector sampled from Gaussian distribution with covariance $X^\top X$|
| `'linear'` | Use linear regression coefficients |
| `'fixed_vector'` | Use fixed projection vector |

### Tuning Metrics

| Metric | Description | Task Type |
|--------|-------------|-----------|
| `'mse'` | Mean Squared Error | Regression |
| `'accuracy'` | Classification Accuracy | Classification |
| `'auc'` | Area Under ROC Curve | Classification |
| `'f1'` | F1 Score | Classification |
