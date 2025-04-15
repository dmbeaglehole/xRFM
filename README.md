# xRFM - Recursive Feature Machines optimized for tabular data

# Installation

Can be installed using the command
```
 pip install .
```

# Standard Usage
```python
import torch
from xrfm import xRFM



def fstar(X):
    return torch.cat([(X[:,0]>0)[:,None], 
	(X[:,1]<0.5)[:,None]], axis=1).float()


DEVICE = 'cpu'
model = xRFM(device=DEVICE, tuning_metric='mse')


n = 1000 # samples
d = 100  # dimension
c = 2    # classes

X_train = torch.randn(n, d, device=DEVICE)
X_test = torch.randn(n, d, device=DEVICE)
y_train = fstar(X_train)
y_test = fstar(X_test)

model.fit(X_train, y_train, X_test, y_test)
y_pred = model.predict(X_test)
```
