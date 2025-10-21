import pytest
import torch

from xrfm.rfm_src import kernels as kernel_module


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    if kernel_module.kermac is None:
        pytest.skip("requires kermac")
    return torch.device("cuda")


@pytest.mark.parametrize(
    "bandwidth,power,exponent,const_mix",
    [
        (0.9, 1.0, 0.8, 0.0),
        (1.5, 1.5, 1.0, 0.25),
        (2.1, 2.0, 1.6, 0.4),
    ],
)
def test_sum_power_kernel_matrix_matches_kermac(device, bandwidth, power, exponent, const_mix):
    torch.manual_seed(17)
    x = torch.randn(7, 5, device=device)
    z = torch.randn(6, 5, device=device)

    torch_kernel = kernel_module.SumPowerLaplaceKernel(
        bandwidth=bandwidth,
        exponent=exponent,
        const_mix=const_mix,
        power=power,
    )
    kermac_kernel = kernel_module.KermacSumPowerLaplaceKernel(
        bandwidth=bandwidth,
        p=power,
        q=exponent,
        const_mix=const_mix,
    )

    torch_kernel_matrix = torch_kernel.get_kernel_matrix(x, z)
    kermac_kernel_matrix = kermac_kernel.get_kernel_matrix(x, z)

    print("torch_kernel_matrix", torch_kernel_matrix[:5, :5])
    print("kermac_kernel_matrix", kermac_kernel_matrix[:5, :5])

    torch.cuda.synchronize()

    assert torch.allclose(
        torch_kernel_matrix,
        kermac_kernel_matrix,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "bandwidth,power,exponent,const_mix",
    [
        (0.7, 1.0, 0.8, 0.0),
        (1.3, 1.5, 1.2, 0.3),
    ],
)
def test_sum_power_grad_matches_kermac(device, bandwidth, power, exponent, const_mix):
    torch.manual_seed(23)
    x = torch.randn(5, 4, device=device)
    z = torch.randn(3, 4, device=device)
    coefs = torch.randn(2, x.shape[0], device=device)

    torch_kernel = kernel_module.SumPowerLaplaceKernel(
        bandwidth=bandwidth,
        exponent=exponent,
        const_mix=const_mix,
        power=power,
    )
    kermac_kernel = kernel_module.KermacSumPowerLaplaceKernel(
        bandwidth=bandwidth,
        p=power,
        q=exponent,
        const_mix=const_mix,
    )

    torch_grads = torch_kernel.get_function_grads(x, z, coefs)
    kermac_grads = kermac_kernel.get_function_grads(x, z, coefs)

    print("torch_grads", torch_grads[:5, :5])
    print("kermac_grads", kermac_grads[:5, :5])

    torch.cuda.synchronize()

    assert torch.allclose(
        torch_grads,
        kermac_grads,
        atol=3e-3,
        rtol=3e-3,
    )
