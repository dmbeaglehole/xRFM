import pytest
import torch

from xrfm import RFM, xRFM


def _build_manual_model(device, adaptive_temp_scaling, left_params, right_params):
    model = xRFM(min_subset_size=1, n_trees=1, device=device)
    model.n_classes_ = 0
    model.categorical_info = None
    model.extra_rfm_params_ = {}

    def _make_leaf(params):
        leaf_model = RFM(kernel='laplace', device=device)
        leaf_model.kernel_obj.bandwidth = params['bandwidth']
        leaf_model.weights = params['weights']
        leaf_model.M = params['M']
        leaf_model.sqrtM = params['sqrtM']
        return {
            'type': 'leaf',
            'model': leaf_model,
            'train_indices': params['train_indices'],
            'is_root': False,
        }

    tree = {
        'type': 'node',
        'split_direction': torch.tensor([1.0, 0.5], dtype=torch.float32, device=device),
        'split_point': torch.tensor(-0.25, dtype=torch.float32, device=device),
        'left': _make_leaf(left_params),
        'right': _make_leaf(right_params),
        'is_root': True,
        'adaptive_temp_scaling': adaptive_temp_scaling,
    }

    model._reset_tree_tables()
    model.trees = [tree]
    model._register_tree_cache(tree)
    return model


@pytest.fixture
def manual_model_setup():
    device = torch.device('cpu')
    adaptive_temp_scaling = 3.7
    left_params = {
        'bandwidth': 2.5,
        'weights': torch.tensor([0.1, 0.3, -0.2], dtype=torch.float32, device=device),
        'M': torch.tensor([[1.0, 0.2], [0.2, 0.5]], dtype=torch.float32, device=device),
        'sqrtM': torch.tensor([[1.0, 0.1], [0.1, 0.7]], dtype=torch.float32, device=device),
        'train_indices': torch.tensor([0, 2], dtype=torch.long, device=device),
    }
    right_params = {
        'bandwidth': 1.25,
        'weights': torch.tensor([0.4, -0.6], dtype=torch.float32, device=device),
        'M': torch.tensor([0.5, 1.5], dtype=torch.float32, device=device),
        'sqrtM': torch.tensor([0.8, 1.2], dtype=torch.float32, device=device),
        'train_indices': torch.tensor([1], dtype=torch.long, device=device),
    }
    model = _build_manual_model(device, adaptive_temp_scaling, left_params, right_params)
    return model, adaptive_temp_scaling, left_params, right_params


def test_get_state_dict_serializes_adaptive_scaling(manual_model_setup):
    model, adaptive_temp_scaling, left_params, right_params = manual_model_setup
    state_dict = model.get_state_dict()
    (param_tree,) = state_dict['param_trees']

    assert param_tree['adaptive_temp_scaling'] == pytest.approx(adaptive_temp_scaling)
    torch.testing.assert_close(param_tree['left']['weights'], left_params['weights'])
    torch.testing.assert_close(param_tree['left']['M'], left_params['M'])
    torch.testing.assert_close(param_tree['left']['sqrtM'], left_params['sqrtM'])
    assert param_tree['left']['bandwidth'] == pytest.approx(left_params['bandwidth'])

    torch.testing.assert_close(param_tree['right']['weights'], right_params['weights'])
    torch.testing.assert_close(param_tree['right']['M'], right_params['M'])
    torch.testing.assert_close(param_tree['right']['sqrtM'], right_params['sqrtM'])
    assert param_tree['right']['bandwidth'] == pytest.approx(right_params['bandwidth'])


def test_load_state_dict_restores_tree_parameters(manual_model_setup):
    model, adaptive_temp_scaling, left_params, right_params = manual_model_setup
    state_dict = model.get_state_dict()
    device = torch.device('cpu')
    new_model = xRFM(min_subset_size=1, n_trees=1, device=device)

    X_train = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    new_model.load_state_dict(state_dict, X_train)

    loaded_tree = new_model.trees[0]
    assert loaded_tree['adaptive_temp_scaling'] == pytest.approx(adaptive_temp_scaling)
    temp_scalings = new_model._split_temp_scaling_tables[0]
    assert temp_scalings[0] == pytest.approx(adaptive_temp_scaling)

    left_model = loaded_tree['left']['model']
    torch.testing.assert_close(left_model.weights, left_params['weights'])
    torch.testing.assert_close(left_model.M, left_params['M'])
    torch.testing.assert_close(left_model.sqrtM, left_params['sqrtM'])
    assert left_model.kernel_obj.bandwidth == pytest.approx(left_params['bandwidth'])
    torch.testing.assert_close(left_model.centers, X_train[left_params['train_indices']])

    right_model = loaded_tree['right']['model']
    torch.testing.assert_close(right_model.weights, right_params['weights'])
    torch.testing.assert_close(right_model.M, right_params['M'])
    torch.testing.assert_close(right_model.sqrtM, right_params['sqrtM'])
    assert right_model.kernel_obj.bandwidth == pytest.approx(right_params['bandwidth'])
    torch.testing.assert_close(right_model.centers, X_train[right_params['train_indices']])
