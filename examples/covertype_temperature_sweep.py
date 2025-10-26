import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_covtype

from xrfm import xRFM


TOTAL_TRAIN = 10_000
TOTAL_VAL = 20_000
TOTAL_TEST = 20_000
MIN_SUBSET_SIZE = 1000
TEMPERATURES = [None] + list(np.logspace(np.log10(0.01), np.log10(2.5), num=10))


def prepare_data(device: torch.device):
    X, y = fetch_covtype(return_X_y=True)
    y = y.astype(np.int64) - 1  # convert to 0-based class labels

    total_required = TOTAL_TRAIN + TOTAL_VAL + TOTAL_TEST
    if total_required > len(X):
        raise ValueError(f"Requested {total_required} samples but dataset only has {len(X)}")

    rng = np.random.default_rng(seed=0)
    selected_indices = rng.choice(len(X), size=total_required, replace=False)
    X_subset = X[selected_indices]
    y_subset = y[selected_indices]

    X_train = torch.as_tensor(X_subset[:TOTAL_TRAIN], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_subset[:TOTAL_TRAIN], dtype=torch.long, device=device)

    start_val = TOTAL_TRAIN
    end_val = TOTAL_TRAIN + TOTAL_VAL
    X_val = torch.as_tensor(X_subset[start_val:end_val], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_subset[start_val:end_val], dtype=torch.long, device=device)

    X_test = torch.as_tensor(X_subset[end_val:], dtype=torch.float32, device=device)
    y_test = torch.as_tensor(y_subset[end_val:], dtype=torch.long, device=device)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(device: torch.device):
    rfm_params = {
        'model': {
            "kernel": 'l2',
            "exponent": 1.0,
            "bandwidth": 10.0,
            "diag": False,
            "bandwidth_mode": "constant"
        },
        'fit': {
            "reg": 1e-3,
            "iters": 3,
        }
    }

    model = xRFM(
        rfm_params=rfm_params,
        device=device,
        min_subset_size=MIN_SUBSET_SIZE,
        tuning_metric='accuracy',
        split_method='top_vector_agop_on_subset',
        split_temperature=None,
        n_threads=1,
    )
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(device)
    y_test_numpy = y_test.cpu().numpy()

    model = build_model(device)

    print("Fitting xRFM...")
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
