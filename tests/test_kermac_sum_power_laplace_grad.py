import pytest

torch = pytest.importorskip("torch")

from xrfm.rfm_src import kernels as kernel_module


pytestmark = pytest.mark.skipif(
    kernel_module.kermac is None or not torch.cuda.is_available(),
    reason="Kermac kernels require CUDA",
)


@pytest.mark.parametrize("d", [7, 12, 29])  # include non-multiple-of-8 for padding behavior
@pytest.mark.parametrize("exponent", [0.8, 1.2])
def test_kermac_sum_power_laplace_grad_matches_reference(d, exponent):
    device = torch.device("cuda")
    torch.manual_seed(0)

    n_x, n_z = 9, 6
    n_f = 3

    bandwidth = 1.3
    const_mix = 0.25
    power = 3

    x = torch.randn(n_x, d, device=device, dtype=torch.float32)
    z = torch.randn(n_z, d, device=device, dtype=torch.float32)
    coefs = torch.randn(n_f, n_x, device=device, dtype=torch.float32)

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

    # Both return shape (f, n_z, d) (potentially with leading batch dim=1 for kermac paths).
    ref_grads = ref.get_function_grads(x, z, coefs)
    fast_grads = fast.get_function_grads(x, z, coefs)

    print("ref_grads", ref_grads)
    print("fast_grads", fast_grads)


    # Normalize potential leading batch dim=1 coming from kermac kernels.
    if fast_grads.dim() == 4 and fast_grads.size(0) == 1:
        fast_grads = fast_grads.squeeze(0)
    if ref_grads.dim() == 4 and ref_grads.size(0) == 1:
        ref_grads = ref_grads.squeeze(0)

    torch.testing.assert_close(fast_grads, ref_grads, atol=5e-3, rtol=5e-3)


