import copy

import pytest
from sklearn.metrics import root_mean_squared_error, accuracy_score
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F

from xrfm import xRFM
from xrfm.rfm_src import kernels as kernel_module


class _DummyLeaf:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return torch.full((X.shape[0], 1), self.value, dtype=X.dtype, device=X.device)

    def predict_proba(self, X):
        return self.predict(X)


def _make_manual_tree(model, left_value=1.0, right_value=0.0):
    split_direction = torch.tensor([1.0, 0.0], dtype=torch.float32, device=model.device)
    split_point = torch.tensor(0.0, dtype=torch.float32, device=model.device)
    empty_indices = torch.zeros(0, dtype=torch.long, device=model.device)
    tree = {
        'type': 'split',
        'split_direction': split_direction,
        'split_point': split_point,
        'left': {
            'type': 'leaf',
            'model': _DummyLeaf(left_value),
            'train_indices': empty_indices,
            'is_root': False
        },
        'right': {
            'type': 'leaf',
            'model': _DummyLeaf(right_value),
            'train_indices': empty_indices,
            'is_root': False
        },
        'is_root': True,
        'adaptive_temp_scaling': 1.0,
    }
    return tree


def _make_manual_model(split_temperature=None):
    device = torch.device('cpu')
    model = xRFM(min_subset_size=1, n_trees=1, device=device, split_temperature=split_temperature)
    model.n_classes_ = 0
    tree = _make_manual_tree(model)
    model.trees = [tree]
    model._register_tree_cache(tree)
    return model, tree


def test_hard_routing_dispatch_matches_hard_path():
    model, tree = _make_manual_model(split_temperature=None)
    X = torch.tensor([[-1.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device=model.device)
    pred_hard = model._predict_tree_hard(X, tree)
    pred_dispatch = model._predict_tree(X, tree)
    torch.testing.assert_close(pred_dispatch, pred_hard)


def test_soft_routing_manual_tree_matches_expected_weights():
    model, tree = _make_manual_model(split_temperature=None)
    model.split_temperature = 2.0
    X = torch.tensor([[-1.0, 0.0], [2.0, 0.0], [0.5, 0.0]], dtype=torch.float32, device=model.device)
    preds = model._predict_tree(X, tree)

    temperature = model.split_temperature
    inv_temperature = 1.0 / temperature
    logits = inv_temperature * (X[:, 0] - 0.0)
    log_prob_left = F.logsigmoid(-logits)
    log_prob_right = F.logsigmoid(logits)
    leaf_log_probs = torch.stack([log_prob_left, log_prob_right], dim=1)
    leaf_log_probs = torch.clamp(leaf_log_probs, min=-50.0)
    max_log_prob = torch.max(leaf_log_probs, dim=1, keepdim=True).values
    stable_log_probs = leaf_log_probs - max_log_prob
    leaf_probs = torch.exp(stable_log_probs)
    normalizer = torch.clamp(
        leaf_probs.sum(dim=1, keepdim=True),
        min=torch.finfo(leaf_probs.dtype).tiny
    )
    weights = leaf_probs / normalizer
    expected = weights[:, :1]

    torch.testing.assert_close(preds, expected)

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


def _train_and_predict(kernel, exponent, norm_p=None, bandwidth=10.0, bandwidth_mode='constant', seed=0, iters=3, d=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = _make_regression_data(seed=seed, device=device, d=d)

    rfm_params = {
        'model': {
            'kernel': kernel,
            'bandwidth': bandwidth,
            'exponent': exponent,
            'norm_p': norm_p,
            'diag': False,
            'bandwidth_mode': bandwidth_mode
        },
        'fit': {
            'reg': 1e-3,
            'iters': iters,
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


@pytest.mark.parametrize('bandwidth_mode', ['constant', 'adaptive'])
@pytest.mark.parametrize('exponent', [0.7, 1.0, 1.2, 1.4])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_lpq_kermac_matches_l1_kermac_when_p_equals_exponent(exponent, bandwidth_mode):
    # lpq_kermac with p=q should match l1_kermac with exponent=q
    preds_lpq = _train_and_predict(kernel='lpq_kermac', exponent=exponent, norm_p=exponent, bandwidth_mode=bandwidth_mode)
    preds_l1 = _train_and_predict(kernel='l1_kermac', exponent=exponent, bandwidth_mode=bandwidth_mode)

    print(f"preds_lpq: {preds_lpq.shape}, preds_l1: {preds_l1.shape}")
    print(f"preds_lpq[:5]: {preds_lpq[:5]}, preds_l1[:5]: {preds_l1[:5]}")

    np.testing.assert_allclose(preds_lpq, preds_l1, atol=1e-2)


@pytest.mark.parametrize('bandwidth_mode', ['constant', 'adaptive'])
@pytest.mark.parametrize('exponent', [0.7, 1.0, 1.2, 1.4])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_lpq_kermac_matches_l2_when_p_equals_2(exponent, bandwidth_mode):
    # lpq_kermac with p=2 should match l2 with same exponent
    preds_lpq = _train_and_predict(kernel='lpq_kermac', exponent=exponent, norm_p=2.0, bandwidth_mode=bandwidth_mode)
    preds_l2 = _train_and_predict(kernel='l2', exponent=exponent, bandwidth_mode=bandwidth_mode)

    print(f"preds_lpq[:5]: {preds_lpq[:5]}, preds_l2[:5]: {preds_l2[:5]}")

    np.testing.assert_allclose(preds_lpq, preds_l2, atol=1e-2)


@pytest.mark.parametrize(
    "norm_p, exponent",
    [
        (1.0, 0.8),
        (1.5, 1.0),
        (2.0, 1.5),
    ],
)
@pytest.mark.parametrize('iters', [3, 0])
@pytest.mark.parametrize('bandwidth_mode', ['constant', 'adaptive'])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_lpq_kermac_matches_lpq(norm_p, exponent, iters, bandwidth_mode):
    # lpq_kermac should match the legacy lpq kernel for the same p, q configuration
    preds_lpq_kermac = _train_and_predict(kernel="lpq_kermac", exponent=exponent, norm_p=norm_p, bandwidth_mode=bandwidth_mode, iters=iters)
    preds_lpq_legacy = _train_and_predict(kernel="lpq_legacy", exponent=exponent, norm_p=norm_p, bandwidth_mode=bandwidth_mode, iters=iters)

    np.testing.assert_allclose(preds_lpq_kermac, preds_lpq_legacy, atol=1e-2)


@pytest.mark.parametrize('bandwidth_mode', ['constant', 'adaptive'])
@pytest.mark.parametrize('exponent', [0.8, 1.0, 1.2])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_l1_kermac_matches_l1(exponent, bandwidth_mode):

    preds_l1_legacy = _train_and_predict(kernel='l1_legacy', exponent=exponent, bandwidth_mode=bandwidth_mode)
    preds_l1_kermac = _train_and_predict(kernel='l1_kermac', exponent=exponent, bandwidth_mode=bandwidth_mode)

    np.testing.assert_allclose(preds_l1_legacy, preds_l1_kermac, atol=1e-1)


@pytest.mark.parametrize('bandwidth_mode', ['constant', 'adaptive'])
@pytest.mark.parametrize('exponent', [0.8, 1.0, 1.2])
@pytest.mark.parametrize('d', [8, 64])
def test_l2_high_dim_matches_l2(exponent, bandwidth_mode, d):
    preds_high_dim = _train_and_predict(kernel='l2_high_dim', exponent=exponent, bandwidth_mode=bandwidth_mode, d=d)
    preds_l2 = _train_and_predict(kernel='l2', exponent=exponent, bandwidth_mode=bandwidth_mode, d=d)

    # Small numerical differences arise for high dimensions with adaptive bandwidth selection.
    np.testing.assert_allclose(preds_high_dim, preds_l2, atol=2e-2)


@pytest.mark.parametrize(
    "kernel, norm_p, exponent",
    [
        ("l1", None, 1.0),
        ("lpq", 1.5, 1.0),
    ]
)
@pytest.mark.parametrize('bandwidth_mode', ['constant', 'adaptive'])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.skipif(kernel_module.kermac is None, reason="requires kermac")
def test_cpu_gpu_l1_lpq_routing(kernel, norm_p, exponent, bandwidth_mode):
    # Generate a shared dataset on CPU so both devices see identical tensors
    torch.manual_seed(7)
    np.random.seed(7)
    n_train, n_val, d = 96, 48, 6
    X = torch.randn(n_train + n_val, d)
    y = (0.5 * X[:, 0] + torch.sin(X[:, 1])).unsqueeze(1)

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    base_rfm_params = {
        "model": {
            "kernel": kernel,
            "bandwidth": 2.5,
            "exponent": exponent,
            "norm_p": norm_p,
            "diag": False,
            "bandwidth_mode": bandwidth_mode,
        },
        "fit": {
            "reg": 1e-4,
            "iters": 2,
            "verbose": False,
            "early_stop_rfm": False,
            "return_best_params": False,
        },
    }

    def _run_on(device):
        torch.manual_seed(11)
        np.random.seed(11)
        if device.type == "cuda":
            torch.cuda.manual_seed(11)
        model = xRFM(
            rfm_params=base_rfm_params,
            device=device,
            min_subset_size=10_000,
            tuning_metric="mse",
            n_threads=1,
        )
        model.fit(
            X_train.to(device),
            y_train.to(device),
            X_val.to(device),
            y_val.to(device),
        )
        preds = model.predict(X_val.to(device))

        def _collect_leaf_models(node):
            if node.get("type") == "leaf":
                return [node["model"]]
            leaves = []
            if node.get("left_child") is not None:
                leaves.extend(_collect_leaf_models(node["left_child"]))
            if node.get("right_child") is not None:
                leaves.extend(_collect_leaf_models(node["right_child"]))
            return leaves

        leaf_models = _collect_leaf_models(model.trees[0])
        leaf_kernel_types = {type(leaf.kernel_obj) for leaf in leaf_models}
        if device.type == "cuda":
            torch.cuda.synchronize()
        return preds, leaf_kernel_types

    cpu_preds, cpu_kernel_types = _run_on(torch.device("cpu"))
    gpu_preds, gpu_kernel_types = _run_on(torch.device("cuda"))

    expected_cpu_cls = (
        kernel_module.ProductLaplaceKernel if kernel == "l1" else kernel_module.LpqLaplaceKernel
    )
    expected_gpu_cls = (
        kernel_module.KermacProductLaplaceKernel if kernel == "l1" else kernel_module.KermacLpqLaplaceKernel
    )

    assert cpu_kernel_types == {expected_cpu_cls}
    assert gpu_kernel_types == {expected_gpu_cls}

    np.testing.assert_allclose(cpu_preds, gpu_preds, atol=1e-2, rtol=1e-3)
