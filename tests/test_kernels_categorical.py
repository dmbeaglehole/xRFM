import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from xrfm.rfm_src import kernels as kernel_module


pytestmark = pytest.mark.skipif(
    kernel_module.kermac is None or not torch.cuda.is_available(),
    reason="Kermac kernels require CUDA for categorical comparisons",
)


def _make_categorical_inputs(device):
    torch.manual_seed(42)
    n_x, n_z = 6, 4
    num_numeric = 2
    cat_sizes = [3, 2]

    x_num = torch.randn(n_x, num_numeric, device=device, dtype=torch.float32)
    z_num = torch.randn(n_z, num_numeric, device=device, dtype=torch.float32)

    x_parts = [x_num]
    z_parts = [z_num]
    for size in cat_sizes:
        x_idx = torch.randint(size, (n_x,), device=device)
        z_idx = torch.randint(size, (n_z,), device=device)
        x_parts.append(F.one_hot(x_idx, num_classes=size).to(device=device, dtype=torch.float32))
        z_parts.append(F.one_hot(z_idx, num_classes=size).to(device=device, dtype=torch.float32))

    x = torch.cat(x_parts, dim=1)
    z = torch.cat(z_parts, dim=1)

    numerical_indices = torch.arange(num_numeric, dtype=torch.long, device=device)
    categorical_indices = []
    offset = num_numeric
    for size in cat_sizes:
        categorical_indices.append(torch.arange(offset, offset + size, dtype=torch.long, device=device))
        offset += size

    categorical_vectors = [
        torch.eye(size, dtype=torch.float32, device=device) for size in cat_sizes
    ]

    return x, z, numerical_indices, categorical_indices, categorical_vectors


@pytest.mark.parametrize(
    "kernel_pair",
    [
        (
            kernel_module.ProductLaplaceKernel,
            kernel_module.KermacProductLaplaceKernel,
            {"bandwidth": 1.7, "exponent": 1.1},
        ),
        (
            kernel_module.LpqLaplaceKernel,
            kernel_module.KermacLpqLaplaceKernel,
            {"bandwidth": 1.3, "p": 1.5, "q": 1.2},
        ),
    ],
)
def test_fast_categorical_kernels_match(kernel_pair):
    torch_kernel_cls, kermac_kernel_cls, kwargs = kernel_pair
    device = torch.device("cuda")

    x, z, num_idx, cat_indices, cat_vectors = _make_categorical_inputs(device)

    torch_kernel = torch_kernel_cls(**kwargs)
    kermac_kernel = kermac_kernel_cls(**kwargs)

    torch_kernel.set_categorical_indices(
        num_idx.clone(),
        [idx.clone() for idx in cat_indices],
        [vec.clone() for vec in cat_vectors],
        device=device,
    )
    kermac_kernel.set_categorical_indices(
        num_idx.clone(),
        [idx.clone() for idx in cat_indices],
        [vec.clone() for vec in cat_vectors],
        device=device,
    )

    torch_mat = torch_kernel.get_kernel_matrix(x, z)
    kermac_mat = kermac_kernel.get_kernel_matrix(x, z)

    torch.testing.assert_close(torch_mat, kermac_mat, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("exponent", [0.8, 1.2])
@pytest.mark.parametrize("use_mat", [False, True])
def test_kermac_sum_power_laplace_fast_categorical_matches_full(exponent, use_mat):
    """
    `KermacSumPowerLaplaceKernel` categorical path should match the "full" computation
    done directly on the expanded one-hot input matrix by `SumPowerLaplaceKernel`.

    This relies on categorical vectors being the identity (so categories correspond to
    one-hot basis vectors), and on `mat` being either None or diagonal (no mixing across
    feature blocks).
    """
    device = torch.device("cuda")
    x, z, num_idx, cat_indices, cat_vectors = _make_categorical_inputs(device)

    bandwidth = 1.3
    const_mix = 0.25
    power = 3

    ref = kernel_module.SumPowerLaplaceKernel(
        bandwidth=bandwidth,
        exponent=exponent,
        const_mix=const_mix,
        power=power,
        bandwidth_mode="constant",
    )
    fast = kernel_module.KermacSumPowerLaplaceKernel(
        bandwidth=bandwidth,
        exponent=exponent,
        const_mix=const_mix,
        power=power,
        bandwidth_mode="constant",
    )
    fast.set_categorical_indices(
        num_idx.clone(),
        [idx.clone() for idx in cat_indices],
        [vec.clone() for vec in cat_vectors],
        device=device,
    )

    mat = None
    if use_mat:
        torch.manual_seed(0)
        # Diagonal transform only (keeps categorical blocks independent).
        mat = (0.5 + torch.rand(x.shape[1], device=device, dtype=torch.float32)).contiguous()

    ref_mat = ref.get_kernel_matrix(x, z, mat)
    fast_mat = fast.get_kernel_matrix(x, z, mat)

    torch.testing.assert_close(fast_mat, ref_mat, atol=1e-4, rtol=1e-4)
