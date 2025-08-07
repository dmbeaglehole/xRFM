from sklearn.metrics import root_mean_squared_error
import torch
from sklearn.model_selection import train_test_split

from xrfm import xRFM


def test_xrfm():
    def target_function(X):
        return torch.cat([
            (X[:, 0] > 0)[:, None],
            (X[:, 1] < 0.5)[:, None]
        ], dim=1).float()

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xRFM(device=device, tuning_metric='mse')

    n_samples = 2000
    n_features = 10
    X = torch.randn(n_samples, n_features, device=device)
    y = target_function(X)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=0)

    model.fit(X_train, y_train, X_val, y_val)
    rmse = root_mean_squared_error(y_test, model.predict(X_test))
    if rmse > 0.3:
        raise AssertionError(f'RMSE was too large: {rmse:g} is larger than 0.3')