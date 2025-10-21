import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import torch

from xrfm.rfm_src import kernels as kernel_module


@dataclass
class KernelSpec:
    name: str
    factory: Callable[[], object]


@dataclass
class BenchmarkConfig:
    dims: List[int]
    sample_sizes: List[int]
    kernels: List[KernelSpec]
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


def benchmark_kernel_matrices(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA available")
    if kernel_module.kermac is None:
        raise RuntimeError("kermac module is not available")

    device = torch.device(config.device)
    torch.manual_seed(0)

    timings: Dict[str, torch.Tensor] = {
        spec.name: torch.zeros(len(config.dims), len(config.sample_sizes), dtype=torch.float64)
        for spec in config.kernels
    }

    for spec in config.kernels:
        print(f"Benchmarking {spec.name}...")
        kernel_timings = timings[spec.name]
        for dim_idx, dim in enumerate(config.dims):
            for size_idx, n in enumerate(config.sample_sizes):
                x = torch.randn(n, dim, device=device, dtype=config.dtype)
                z = torch.randn(n, dim, device=device, dtype=config.dtype)

                kernel_obj = spec.factory()

                # Warm-up
                kernel_obj.get_kernel_matrix(x[:64], z[:64])
                torch.cuda.synchronize()

                start = time.perf_counter()
                kernel_obj.get_kernel_matrix(x, z)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                kernel_timings[dim_idx, size_idx] = elapsed

    return timings


def plot_results(config: BenchmarkConfig, timings: Dict[str, torch.Tensor], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Per-kernel plots showing all dimensions
    for spec in config.kernels:
        kernel_timings = timings[spec.name]
        plt.figure(figsize=(8, 5))
        for dim_idx, dim in enumerate(config.dims):
            plt.plot(
                config.sample_sizes,
                kernel_timings[dim_idx].tolist(),
                marker="o",
                label=f"d = {dim}",
            )
        plt.title(f"{spec.name} Kernel Scaling")
        plt.xlabel("Sample Size (n)")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(output_dir, f"{spec.name.lower().replace(' ', '_')}_scaling.png")
        plt.savefig(filename, dpi=200)
        plt.close()

    # Cross-kernel comparison for each dimension
    for dim_idx, dim in enumerate(config.dims):
        plt.figure(figsize=(8, 5))
        for spec in config.kernels:
            plt.plot(
                config.sample_sizes,
                timings[spec.name][dim_idx].tolist(),
                marker="o",
                label=spec.name,
            )
        plt.title(f"Kernel Comparison at d = {dim}")
        plt.xlabel("Sample Size (n)")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(output_dir, f"comparison_dim_{dim}.png")
        plt.savefig(filename, dpi=200)
        plt.close()

    # Overall summary (largest dimension curve for each kernel)
    plt.figure(figsize=(10, 6))
    for spec in config.kernels:
        plt.plot(
            config.sample_sizes,
            timings[spec.name][-1].tolist(),
            marker="o",
            label=f"{spec.name} (d={config.dims[-1]})",
        )
    plt.title("Kernel Scaling Comparison (Largest Dimension)")
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(output_dir, "comparison_max_dimension.png")
    plt.savefig(filename, dpi=200)
    plt.close()


def main() -> None:
    sum_power_params = dict(bandwidth=1.2, p=1.5, q=1.0, const_mix=0.2)
    lpq_params = dict(bandwidth=1.2, p=1.5, q=1.0)
    product_params = dict(bandwidth=1.2, exponent=1.0)

    config = BenchmarkConfig(
        dims=[10, 100, 500],
        sample_sizes=[1000, 5000, 10000, 50000],
        kernels=[
            KernelSpec(
                "SumPower",
                lambda params=sum_power_params: kernel_module.KermacSumPowerLaplaceKernel(**params),
            ),
            KernelSpec(
                "Lpq",
                lambda params=lpq_params: kernel_module.KermacLpqLaplaceKernel(**params),
            ),
            KernelSpec(
                "ProductLaplace",
                lambda params=product_params: kernel_module.KermacProductLaplaceKernel(**params),
            ),
        ],
    )

    timings = benchmark_kernel_matrices(config)
    plot_results(config, timings, output_dir=os.path.join("examples", "plots"))

    for spec in config.kernels:
        print(f"{spec.name} timings (seconds):")
        for dim, row in zip(config.dims, timings[spec.name]):
            formatted = ", ".join(f"{t:.4f}" for t in row.tolist())
            print(f"\td={dim}: {formatted}")


if __name__ == "__main__":
    main()
