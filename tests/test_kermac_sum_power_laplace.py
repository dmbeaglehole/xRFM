import pytest

torch = pytest.importorskip("torch")

from xrfm.rfm_src import kernels as kernel_module


pytestmark = pytest.mark.skipif(
    kernel_module.kermac is None or not torch.cuda.is_available(),
    reason="Kermac kernels require CUDA",
)


@pytest.mark.parametrize("K", [7, 12, 29])  # include non-multiple-of-8 for padding behavior
@pytest.mark.parametrize("exponent", [0.8, 1.2])
def test_kermac_sum_power_laplace_matches_reference(K, exponent):
    device = torch.device("cuda")
    torch.manual_seed(0)

    M, N = 16, 11
    bandwidth = 1.3
    const_mix = 0.25
    power = 3

    x = torch.randn(M, K, device=device, dtype=torch.float32)
    z = torch.randn(N, K, device=device, dtype=torch.float32)

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

    ref_mat = ref.get_kernel_matrix(x, z)
    fast_mat = fast.get_kernel_matrix(x, z)

    torch.testing.assert_close(fast_mat, ref_mat, atol=1e-4, rtol=1e-4)


