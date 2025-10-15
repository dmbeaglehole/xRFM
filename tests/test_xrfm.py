import pytest
from sklearn.metrics import root_mean_squared_error, accuracy_score
import torch
from sklearn.model_selection import train_test_split
import numpy as np

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
    model = xRFM(device=device, tuning_metric='mse', time_limit_s=time_limit_s, n_threads=1)

    n_samples = 2000
    n_features = 10
    X = torch.randn(n_samples, n_features, device=device)
    y = target_function(X)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=0)

    model.fit(X_train, y_train, X_val, y_val)
    rmse = root_mean_squared_error(y_test.cpu().numpy(), model.predict(X_test))
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
    acc = accuracy_score(y_test.cpu().numpy(), model.predict(X_test))
    if acc < 0.8:
        raise AssertionError(f'Accuracy was too small: {acc:g} is smaller than 0.8')


def _make_regression_data(seed=0, n_train=256, n_val=128, d=8, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    X = torch.randn(n_train + n_val, d, generator=rng, device=device)
    y = (X[:, 0] ** 2 + 0.5 * X[:, 1])[:, None].float()
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _train_and_predict(kernel, exponent, norm_p=None, bandwidth=5.0, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = _make_regression_data(seed=seed, device=device)

    rfm_params = {
        'model': {
            'kernel': kernel,
            'bandwidth': bandwidth,
            'exponent': exponent,
            'norm_p': norm_p,
            'diag': False,
            'bandwidth_mode': 'constant'
        },
        'fit': {
            'reg': 1e-4,
            'iters': 3,
            'verbose': False,
            'early_stop_rfm': False,
            'return_best_params': False,
            'verbose': True
        }
    }

    model = xRFM(rfm_params=rfm_params, device=device, min_subset_size=10_000, tuning_metric='mse', n_threads=1)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    return preds


@pytest.mark.parametrize('exponent', [0.7, 1.0, 1.2, 1.4])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_lpq_kermac_matches_l1_kermac_when_p_equals_exponent(exponent):
    # lpq_kermac with p=q should match l1_kermac with exponent=q
    preds_lpq = _train_and_predict(kernel='lpq_kermac', exponent=exponent, norm_p=exponent)
    preds_l1 = _train_and_predict(kernel='l1_kermac', exponent=exponent)

    print(f"preds_lpq: {preds_lpq.shape}, preds_l1: {preds_l1.shape}")
    print(f"preds_lpq[:5]: {preds_lpq[:5]}, preds_l1[:5]: {preds_l1[:5]}")

    np.testing.assert_allclose(preds_lpq, preds_l1, atol=1e-2)


@pytest.mark.parametrize('exponent', [0.7, 1.0, 1.2, 1.4])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_lpq_kermac_matches_l2_when_p_equals_2(exponent):
    # lpq_kermac with p=2 should match l2 with same exponent
    preds_lpq = _train_and_predict(kernel='lpq_kermac', exponent=exponent, norm_p=2.0)
    preds_l2 = _train_and_predict(kernel='l2', exponent=exponent)

    print(f"preds_lpq[:5]: {preds_lpq[:5]}, preds_l2[:5]: {preds_l2[:5]}")

    np.testing.assert_allclose(preds_lpq, preds_l2, atol=1e-2)


@pytest.mark.parametrize('exponent', [0.8, 1.0, 1.2])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_l1_kermac_matches_l1(exponent):

    preds_l1 = _train_and_predict(kernel='l1', exponent=exponent)
    preds_l1_kermac = _train_and_predict(kernel='l1_kermac', exponent=exponent)

    np.testing.assert_allclose(preds_l1, preds_l1_kermac, atol=1e-1)
