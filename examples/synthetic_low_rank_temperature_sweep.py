"""Temperature sweep demo on a synthetic low-rank polynomial dataset."""

import time
from pathlib import Path

import numpy as np
import torch

from xrfm import xRFM


# Dataset configuration
TOTAL_TRAIN = 20_000
TOTAL_VAL = 5_000
TOTAL_TEST = 5_000
NUM_FEATURES = 128
LOW_RANK = 5
NOISE_STD = 0.25
RANDOM_SEED = 0

MIN_SUBSET_SIZE = 1000
TEMPERATURES = [None] + list(np.logspace(np.log10(0.01), np.log10(3), num=20))


def _make_generator(device: torch.device) -> torch.Generator:
    """Create a torch.Generator tied to the target device."""
    if device.type == "cuda":
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()
    gen.manual_seed(RANDOM_SEED)
    return gen


def prepare_data(device: torch.device):
    """Generate train/val/test splits for the low-rank polynomial dataset."""
    gen = _make_generator(device)

    U = torch.randn(LOW_RANK, NUM_FEATURES, generator=gen, device=device) / np.sqrt(NUM_FEATURES)
    linear_weights = torch.randn(LOW_RANK, generator=gen, device=device)
    quadratic_weights = torch.randn(LOW_RANK, generator=gen, device=device)

    def sample(n_samples: int):
        X = torch.randn(n_samples, NUM_FEATURES, generator=gen, device=device)
        Z = X @ U.t()
        polynomial = Z @ linear_weights + (Z.pow(2) @ quadratic_weights)
        noise = NOISE_STD * torch.randn(n_samples, generator=gen, device=device)
        logits = polynomial + noise
        y = (logits > 0).to(dtype=torch.long)
        return X, y

    X_train, y_train = sample(TOTAL_TRAIN)
    X_val, y_val = sample(TOTAL_VAL)
    X_test, y_test = sample(TOTAL_TEST)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(device: torch.device):
    rfm_params = {
        "model": {
            "kernel": "l2",
            "exponent": 1.0,
            "bandwidth": 5.0,
            "diag": False,
            "bandwidth_mode": "constant",
        },
        "fit": {
            "reg": 1e-3,
            "iters": 3,
            "early_stop_rfm": False,
        },
    }

    model = xRFM(
        rfm_params=rfm_params,
        device=device,
        min_subset_size=MIN_SUBSET_SIZE,
        tuning_metric="accuracy",
        split_method="top_vector_agop_on_subset",
        split_temperature=None,
        overlap_fraction=0.1,
        n_threads=1,
    )
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(device)
    y_test_numpy = y_test.cpu().numpy()

    model = build_model(device)

    print("Fitting xRFM on synthetic low-rank data...")
    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    fit_time = time.time() - start_time
    print(f"Training completed in {fit_time:.2f} seconds.")

    accuracies = []
    temperature_labels = []
    x_positions = np.arange(len(TEMPERATURES))

    for temperature in TEMPERATURES:
        if temperature is None:
            model.split_temperature = None
            temp_label = "hard routing"
        else:
            model.split_temperature = float(temperature)
            temp_label = f"{temperature:.3f}"

        start_time = time.time()
        preds = model.predict(X_test)
        predict_time = time.time() - start_time

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            
        correct = (preds == y_test_numpy).mean()
        accuracies.append(correct)
        temperature_labels.append(temp_label)
        print(f"Temperature {temp_label}: test accuracy = {correct:.4f}, predict time = {predict_time:.2f} seconds")

if __name__ == "__main__":
    main()
