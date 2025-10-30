"""
Temperature sweep demo that pulls a classification dataset from OpenML.

Run with e.g.
    python examples/openml_temperature_sweep.py --data-id 1461
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from xrfm import xRFM


DEFAULT_DATA_ID = 45072  # Bank Marketing dataset; override via --data-id
DEFAULT_TOTAL_TRAIN = 100_000
DEFAULT_TOTAL_VAL = 20_000
DEFAULT_TOTAL_TEST = 20_000
DEFAULT_MIN_SUBSET_SIZE = 50_000
TEMPERATURES = [None] + list(np.logspace(np.log10(0.05), np.log10(3), num=20))


def parse_args():
    parser = argparse.ArgumentParser(description="Temperature sweep on an OpenML classification dataset.")
    parser.add_argument("--data-id", type=int, default=DEFAULT_DATA_ID, help="OpenML dataset id (classification).")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TOTAL_TRAIN,
                        help="Number of training samples to use.")
    parser.add_argument("--val-size", type=int, default=DEFAULT_TOTAL_VAL, help="Number of validation samples to use.")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TOTAL_TEST, help="Number of test samples to use.")
    parser.add_argument("--min-subset-size", type=int, default=DEFAULT_MIN_SUBSET_SIZE,
                        help="Minimum subset size passed to xRFM.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for NumPy shuffling.")
    return parser.parse_args()


def _compute_split_sizes(n_available: int, train: int, val: int, test: int):
    desired_total = train + val + test
    if desired_total <= n_available:
        return train, val, test

    print(
        f"Requested {desired_total} total samples but only {n_available} are available after preprocessing. "
        "Using proportional split."
    )

    train_ratio = train / desired_total
    val_ratio = val / desired_total
    test_ratio = test / desired_total

    adjusted_train = max(1, int(round(train_ratio * n_available)))
    adjusted_val = max(1, int(round(val_ratio * n_available)))
    adjusted_test = n_available - adjusted_train - adjusted_val
    if adjusted_test <= 0:
        adjusted_test = 1
        if adjusted_val > 1:
            adjusted_val -= 1
        if adjusted_train + adjusted_val + adjusted_test > n_available and adjusted_train > 1:
            adjusted_train -= 1

    # Ensure the counts sum exactly to the available samples.
    diff = (adjusted_train + adjusted_val + adjusted_test) - n_available
    if diff > 0 and adjusted_train > diff:
        adjusted_train -= diff
    elif diff > 0 and adjusted_val > diff:
        adjusted_val -= diff

    return adjusted_train, adjusted_val, adjusted_test


def prepare_data(args, device: torch.device):
    dataset = fetch_openml(data_id=args.data_id, as_frame=True)
    X = dataset.data
    y = dataset.target

    if X is None or y is None:
        raise ValueError(f"OpenML dataset {args.data_id} does not contain both data and target.")

    # Normalize missing markers; features get imputed later.
    X = X.replace("?", np.nan)
    y_series = pd.Series(y, name="_target").replace("?", np.nan)
    if y_series.isna().any():
        raise ValueError(
            f"OpenML dataset {args.data_id} contains missing target labels; cannot run classification."
        )
    y = y_series

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    transformers = []
    numeric_transformer = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found after preprocessing.")

    column_transformer = ColumnTransformer(transformers, sparse_threshold=0.0)
    X_processed = column_transformer.fit_transform(X)
    X_array = np.ascontiguousarray(X_processed, dtype=np.float32)

    label_encoder = LabelEncoder()
    y_array = np.ascontiguousarray(label_encoder.fit_transform(y), dtype=np.int64)

    total_available = len(X_array)
    train_count, val_count, test_count = _compute_split_sizes(
        total_available, args.train_size, args.val_size, args.test_size
    )
    total_required = train_count + val_count + test_count

    if total_required > total_available:
        raise ValueError("Unable to allocate requested splits after adjustment.")

    rng = np.random.default_rng(seed=args.seed)
    indices = rng.choice(total_available, size=total_required, replace=False)
    X_subset = X_array[indices]
    y_subset = y_array[indices]

    X_train = torch.from_numpy(X_subset[:train_count]).to(device)
    y_train = torch.from_numpy(y_subset[:train_count]).to(device)

    val_start = train_count
    val_end = train_count + val_count
    X_val = torch.from_numpy(X_subset[val_start:val_end]).to(device)
    y_val = torch.from_numpy(y_subset[val_start:val_end]).to(device)

    X_test = torch.from_numpy(X_subset[val_end:]).to(device)
    y_test = torch.from_numpy(y_subset[val_end:]).to(device)

    return X_train, y_train, X_val, y_val, X_test, y_test, train_count


def build_model(device: torch.device, min_subset_size: int):
    rfm_params = {
        "model": {
            "kernel": "l2",
            "exponent": 1.0,
            "bandwidth": 10.0,
            "diag": False,
            "bandwidth_mode": "adaptive",
        },
        "fit": {
            "reg": 1e-3,
            "iters": 3,
            "early_stop_rfm": True,
        },
    }

    model = xRFM(
        rfm_params=rfm_params,
        device=device,
        min_subset_size=min_subset_size,
        tuning_metric="accuracy",
        split_method="top_vector_agop_on_subset",
        split_temperature=None,
        overlap_fraction=0.1,
    )
    return model


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_count,
    ) = prepare_data(args, device)
    y_test_numpy = y_test.cpu().numpy()

    min_subset_size = min(args.min_subset_size, train_count)
    if min_subset_size <= 0:
        raise ValueError("min_subset_size must be positive after adjustment.")

    model = build_model(device, min_subset_size=min_subset_size)

    print(f"Fitting xRFM on OpenML dataset {args.data_id}...")
    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    fit_time = time.time() - start_time
    print(f"Training completed in {fit_time:.2f} seconds.")

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

        accuracy = (preds == y_test_numpy).mean()
        print(
            f"Temperature {temp_label}: test accuracy = {accuracy:.4f}, "
            f"predict time = {predict_time:.2f} seconds"
        )


if __name__ == "__main__":
    main()
