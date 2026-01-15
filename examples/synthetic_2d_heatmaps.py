"""
Generate 2D heatmaps for xRFM predictions on synthetic 2D functions.

This script fits xRFM regression models with multiple kernels and saves
side-by-side heatmaps for:
  - ground truth
  - prediction
  - absolute error

Kernels:
  - l2
  - lpq
  - product_laplace   (product kernel)
  - sum_power_laplace (sum-power kernel)

Example:
  python examples/synthetic_2d_heatmaps.py --device cuda --out_dir /tmp/xrfm_heatmaps
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt

# Allow running as a standalone script without requiring `pip install -e .`
try:
    from xrfm import xRFM
except ModuleNotFoundError:  # pragma: no cover
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    from xrfm import xRFM


TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Synthetic2DFunction:
    name: str
    domain: Tuple[float, float]  # min/max for both x and y (square domain)
    fn: TensorFn


def _peaks(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    z = (
        3.0 * (1 - x).pow(2) * torch.exp(-(x.pow(2)) - (y + 1).pow(2))
        - 10.0 * (x / 5.0 - x.pow(3) - y.pow(5)) * torch.exp(-(x.pow(2)) - y.pow(2))
        - (1.0 / 3.0) * torch.exp(-(x + 1).pow(2) - y.pow(2))
    )
    return z.unsqueeze(-1)


def _sine_mix(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    z = torch.sin(2 * np.pi * x) + torch.cos(2 * np.pi * y) + 0.5 * torch.sin(2 * np.pi * (x + y))
    return z.unsqueeze(-1)


def _smooth_checker(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    z = torch.tanh(5.0 * torch.sin(np.pi * x) * torch.sin(np.pi * y))
    return z.unsqueeze(-1)


def _himmelblau_scaled(xy: torch.Tensor) -> torch.Tensor:
    # Classic Himmelblau function; we return a scaled version for easier regression.
    x = xy[:, 0]
    y = xy[:, 1]
    f = (x.pow(2) + y - 11).pow(2) + (x + y.pow(2) - 7).pow(2)
    z = -torch.log1p(f)  # in (-inf, 0], smoother dynamic range
    return z.unsqueeze(-1)

def _quadrant_step(xy: torch.Tensor) -> torch.Tensor:
    """Discontinuous: +1 in first quadrant, -1 elsewhere."""
    x = xy[:, 0]
    y = xy[:, 1]
    z = torch.where((x >= 0) & (y >= 0), torch.ones_like(x), -torch.ones_like(x))
    return z.unsqueeze(-1)


def _circle_indicator(xy: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Discontinuous indicator of a disk."""
    x = xy[:, 0]
    y = xy[:, 1]
    inside = (x * x + y * y) <= (radius * radius)
    z = torch.where(inside, torch.ones_like(x), torch.zeros_like(x))
    return z.unsqueeze(-1)


def _abs_ridge(xy: torch.Tensor) -> torch.Tensor:
    """Non-smooth along axes due to absolute values."""
    x = xy[:, 0]
    y = xy[:, 1]
    z = torch.abs(x) + 0.5 * torch.abs(y)
    return z.unsqueeze(-1)


def _max_hinge(xy: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear with kinks along x=y and a hinge plane."""
    x = xy[:, 0]
    y = xy[:, 1]
    z = torch.maximum(x, y) + 0.75 * torch.relu(x + y - 0.5)
    return z.unsqueeze(-1)


def _sign_checkerboard(xy: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    """Discontinuous sign of a sinusoidal checkerboard."""
    x = xy[:, 0]
    y = xy[:, 1]
    z = torch.sign(torch.sin(k * x) * torch.sin(k * y))
    return z.unsqueeze(-1)


def get_synthetic_functions() -> List[Synthetic2DFunction]:
    return [
        # Smooth-ish
        Synthetic2DFunction("peaks", (-3.0, 3.0), _peaks),
        Synthetic2DFunction("sine_mix", (-1.5, 1.5), _sine_mix),
        Synthetic2DFunction("smooth_checker", (-2.0, 2.0), _smooth_checker),
        Synthetic2DFunction("himmelblau_scaled", (-5.0, 5.0), _himmelblau_scaled),
        # Non-smooth / discontinuous
        Synthetic2DFunction("quadrant_step", (-2.0, 2.0), _quadrant_step),
        Synthetic2DFunction("circle_indicator", (-2.0, 2.0), _circle_indicator),
        Synthetic2DFunction("abs_ridge", (-2.0, 2.0), _abs_ridge),
        Synthetic2DFunction("max_hinge", (-2.0, 2.0), _max_hinge),
        Synthetic2DFunction("sign_checkerboard", (-2.0, 2.0), _sign_checkerboard),
    ]


def _canon_kernel_name(name: str) -> str:
    k = name.strip().lower()
    aliases = {
        "l2": "l2",
        "laplace": "l2",
        "lpq": "lpq",
        "product": "product_laplace",
        "product_laplace": "product_laplace",
        "l1": "product_laplace",
        "sum": "sum_power_laplace",
        "sum_power": "sum_power_laplace",
        "sum_power_laplace": "sum_power_laplace",
        "l1_power": "sum_power_laplace",
    }
    if k not in aliases:
        raise ValueError(f"Unknown kernel '{name}'. Expected one of: {sorted(set(aliases.keys()))}")
    return aliases[k]


def _make_grid(domain: Tuple[float, float], grid_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    lo, hi = domain
    xs = np.linspace(lo, hi, grid_size, dtype=np.float32)
    ys = np.linspace(lo, hi, grid_size, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    grid_t = torch.as_tensor(grid, device=device, dtype=dtype)
    return grid_t, X, Y


def _sample_uniform(domain: Tuple[float, float], n: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    lo, hi = domain
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    u = torch.rand((n, 2), generator=gen, device=device, dtype=dtype)
    return lo + (hi - lo) * u


def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def _fit_predict_heatmaps(
    fn: Synthetic2DFunction,
    kernel: str,
    device: torch.device,
    n_train: int,
    n_val: int,
    grid_size: int,
    noise_std: float,
    bandwidth: float,
    exponent: float,
    lpq_p: float,
    lpq_q: float,
    iters: int,
    reg: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    dtype = torch.float32

    X_train = _sample_uniform(fn.domain, n_train, device=device, dtype=dtype, seed=seed + 1)
    X_val = _sample_uniform(fn.domain, n_val, device=device, dtype=dtype, seed=seed + 2)

    y_train = fn.fn(X_train)
    y_val = fn.fn(X_val)

    if noise_std > 0:
        gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
        gen.manual_seed(int(seed + 3))
        y_train = y_train + noise_std * torch.randn_like(y_train, generator=gen)

    kernel = _canon_kernel_name(kernel)
    model_params = {
        "kernel": kernel,
        "bandwidth": float(bandwidth),
        "exponent": float(lpq_q if kernel == "lpq" else exponent),
        "norm_p": float(lpq_p),
        "diag": False,
        "bandwidth_mode": "constant",
    }
    fit_params = {
        "solver": "solve",
        "reg": float(reg),
        "iters": int(iters),
        "early_stop_rfm": False,
        "verbose": False,
    }

    rfm_params = {"model": model_params, "fit": fit_params}

    # Force a single leaf for small synthetic 2D demos (keeps visuals interpretable).
    model = xRFM(
        rfm_params=rfm_params,
        device=device,
        max_leaf_size=1_000_000,
        n_trees=1,
        n_tree_iters=0,
        tuning_metric="mse",
        split_method="top_vector_agop_on_subset",
        split_temperature=None,
        n_threads=1,
        random_state=seed,
    )

    model.fit(X_train, y_train, X_val, y_val)

    grid_t, Xg, Yg = _make_grid(fn.domain, grid_size, device=device, dtype=dtype)
    y_true = _to_numpy_1d(fn.fn(grid_t))
    y_pred = _to_numpy_1d(model.predict(grid_t))

    y_true_grid = y_true.reshape(grid_size, grid_size)
    y_pred_grid = y_pred.reshape(grid_size, grid_size)
    abs_err = np.abs(y_pred_grid - y_true_grid)

    return {
        "X": Xg,
        "Y": Yg,
        "y_true": y_true_grid,
        "y_pred": y_pred_grid,
        "abs_err": abs_err,
    }


def _plot_triptych(
    out_file: Path,
    title: str,
    domain: Tuple[float, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abs_err: np.ndarray,
):
    lo, hi = domain
    extent = (lo, hi, lo, hi)

    vmin = float(np.nanmin(y_true))
    vmax = float(np.nanmax(y_true))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(y_true, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Truth")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Prediction")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(abs_err, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[2].set_title("|Error|")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate 2D prediction heatmaps for xRFM on synthetic functions.")
    parser.add_argument("--out_dir", type=str, default="outputs/synthetic_2d_heatmaps", help="Directory to write PNGs.")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--grid_size", type=int, default=200)
    parser.add_argument("--noise_std", type=float, default=0.0)

    parser.add_argument(
        "--kernels",
        type=str,
        nargs="+",
        default=["l2", "lpq", "product_laplace", "sum_power_laplace"],
        help="Kernel list. Aliases: product, sum, laplace, l1, l1_power.",
    )
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=None,
        help="Subset of functions to run (by name). Defaults to all.",
    )

    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--exponent", type=float, default=1.0, help="Exponent for l2/product/sum_power kernels.")
    parser.add_argument("--lpq_p", type=float, default=2.0, help="p for lpq kernel.")
    parser.add_argument("--lpq_q", type=float, default=1.0, help="q for lpq kernel (passed via 'exponent').")

    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--reg", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    all_fns = get_synthetic_functions()
    fn_by_name = {f.name: f for f in all_fns}
    if args.functions is None:
        selected_fns = all_fns
    else:
        missing = [n for n in args.functions if n not in fn_by_name]
        if missing:
            raise ValueError(f"Unknown function names: {missing}. Available: {sorted(fn_by_name.keys())}")
        selected_fns = [fn_by_name[n] for n in args.functions]

    kernels = [_canon_kernel_name(k) for k in args.kernels]

    print(f"Device: {device}")
    print(f"Writing outputs to: {out_dir.resolve()}")
    print(f"Functions: {[f.name for f in selected_fns]}")
    print(f"Kernels: {kernels}")

    for f in selected_fns:
        for k in kernels:
            results = _fit_predict_heatmaps(
                fn=f,
                kernel=k,
                device=device,
                n_train=args.n_train,
                n_val=args.n_val,
                grid_size=args.grid_size,
                noise_std=args.noise_std,
                bandwidth=args.bandwidth,
                exponent=args.exponent,
                lpq_p=args.lpq_p,
                lpq_q=args.lpq_q,
                iters=args.iters,
                reg=args.reg,
                seed=args.seed,
            )

            out_file = out_dir / f"{f.name}__{k}.png"
            title = f"xRFM heatmaps — fn={f.name} kernel={k} (train={args.n_train}, iters={args.iters}, bw={args.bandwidth})"
            _plot_triptych(
                out_file=out_file,
                title=title,
                domain=f.domain,
                y_true=results["y_true"],
                y_pred=results["y_pred"],
                abs_err=results["abs_err"],
            )
            print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()


