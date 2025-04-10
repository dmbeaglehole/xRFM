# xRFM - Recursive Feature Machines optimized for tabular data

# Installation

Can be installed using the command
```
 pip install tabrfm
```

# Standard Usage
```python
import torch
from tabrfm import TabRFM, LaplaceKernel



def fstar(X):
    return torch.cat([(X[:,0]>0)[:,None], 
	(X[:,1]<0.5)[:,None]], axis=1).float()


DEVICE = 'cpu'
model = TabRFM(kernel=LaplaceKernel(bandwidth=10., exponent=1.), device=DEVICE)

n = 1000 # samples
d = 100  # dimension
c = 2    # classes

X_train = torch.randn(n, d, device=DEVICE)
X_test = torch.randn(n, d, device=DEVICE)
y_train = fstar(X_train)
y_test = fstar(X_test)

model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=5,
    reg=1e-3,
    classification=False
)
```
