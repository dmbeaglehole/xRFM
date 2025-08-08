import pytest
from sklearn.metrics import root_mean_squared_error, accuracy_score
import torch
from sklearn.model_selection import train_test_split

from xrfm import xRFM

@pytest.mark.parametrize(
    'time_limit_s', [0, None]
)
def test_regression(time_limit_s):
    def target_function(X):
        return torch.cat([
            (X[:, 0] > 0)[:, None],
            (X[:, 1] < 0.5)[:, None]
        ], dim=1).float()

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xRFM(device=device, tuning_metric='mse', time_limit_s=time_limit_s)

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


@pytest.mark.parametrize(
    "tuning_metric", [None, 'accuracy', 'brier', 'logloss', 'f1', 'auc']
)
@pytest.mark.parametrize(
    'classification_mode', ['zero_one', 'prevalence']
)
@pytest.mark.parametrize(
    'n_classes', [2, 3]
)
def test_classification(tuning_metric, classification_mode, n_classes):
    def target_function(X):
        return torch.clamp(X[:, 0].long(), min=0, max=n_classes-1)

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xRFM(device=device, tuning_metric=tuning_metric, classification_mode=classification_mode)

    n_samples = 2000
    n_features = 10
    X = torch.randn(n_samples, n_features, device=device)
    y = target_function(X)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=0)

    model.fit(X_train, y_train, X_val, y_val)
    acc = accuracy_score(y_test, model.predict(X_test))
    if acc < 0.8:
        raise AssertionError(f'Accuracy was too small: {acc:g} is smaller than 0.8')