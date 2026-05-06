import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SPARSE_TAIL_BOUND = -1.0

# fixed from the data audit
ZERO_DISTURBANCE_RAW_INDICES = [8095, 8311]
EXPECTED_ZERO_DISTURBANCE_ROWS = 2
EXPECTED_SPARSE_TAIL_ROWS = 90
EXPECTED_CLEAN_ROWS = 10302

TEST_SIZE = 0.30
RANDOM_STATE = 42
EXPECTED_TRAIN_ROWS = 7211
EXPECTED_TEST_ROWS = 3091

ROCOF_COLS = [f"RoCoF Bus {i}" for i in range(1, 10)]
NADIR_COLS = [f"Nadir Bus {i}" for i in range(1, 10)]

INPUT_FEATURES: List[str] = [
    "SG 1 MVA",
    "SG 2 MVA",
    "SG 3 MVA",
    "System Loading",
    "CIG MW",
    "Outage SG",
    "SG 1 MW",
    "SG 2 MW",
    "SG 3 MW",
]

# keep this order fixed for the training and SHAP scripts
ALL_TARGETS: List[str] = (
    [f"RoCoF Bus {i}" for i in range(1, 10)]
    + ["RoCoF Worst"]
    + [f"Nadir Bus {i}" for i in range(1, 10)]
    + ["Nadir Worst"]
)


# loads the raw CSV, applies the dissertation cleaning rules, splits, scales, and saves artefacts
def load_and_preprocess(
    csv_path: str,
    output_dir: str = "data/processed",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    StandardScaler,
]:
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    n_raw = len(df_raw)
    print(f"Loaded {n_raw} rows from {csv_path}")

    # check raw indices before reset_index removes the original row labels
    zero_rocof_mask = (df_raw[ROCOF_COLS] == 0).all(axis=1)
    zero_nadir_mask = (df_raw[NADIR_COLS] == 60.0).all(axis=1)
    zero_mask = zero_rocof_mask & zero_nadir_mask
    zero_indices = df_raw.index[zero_mask].tolist()
    n_zero = len(zero_indices)

    assert zero_indices == ZERO_DISTURBANCE_RAW_INDICES, (
        "Unexpected zero-disturbance raw indices. "
        f"Expected {ZERO_DISTURBANCE_RAW_INDICES}, got {zero_indices}."
    )
    assert n_zero == EXPECTED_ZERO_DISTURBANCE_ROWS, (
        f"Expected {EXPECTED_ZERO_DISTURBANCE_ROWS} zero-disturbance rows; "
        f"got {n_zero}."
    )

    df = df_raw.loc[~zero_mask].copy()
    print(f"Removed {n_zero} zero-disturbance rows. Remaining: {len(df)}")

    # not a physics cut, just the modelling domain used for this surrogate
    sparse_tail_mask = (df[ROCOF_COLS] < SPARSE_TAIL_BOUND).any(axis=1)
    n_sparse = int(sparse_tail_mask.sum())

    assert n_sparse == EXPECTED_SPARSE_TAIL_ROWS, (
        f"Expected {EXPECTED_SPARSE_TAIL_ROWS} sparse-tail rows; got {n_sparse}."
    )

    df = df.loc[~sparse_tail_mask].reset_index(drop=True)
    print(
        f"Removed {n_sparse} sparse-tail rows "
        f"(any RoCoF < {SPARSE_TAIL_BOUND} Hz/s). Remaining: {len(df)}"
    )

    n_clean = len(df)
    assert n_clean == EXPECTED_CLEAN_ROWS, (
        f"Expected {EXPECTED_CLEAN_ROWS} rows after cleaning; got {n_clean}."
    )

    # build worst-case targets after filtering so the domain is consistent
    df["RoCoF Worst"] = df[ROCOF_COLS].min(axis=1)
    df["Nadir Worst"] = df[NADIR_COLS].min(axis=1)

    X = df[INPUT_FEATURES].copy()

    X_train, X_test, idx_train, idx_test = train_test_split(
        X,
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Outage SG"],
    )

    scaler_X = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler_X.fit_transform(X_train),
        columns=INPUT_FEATURES,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler_X.transform(X_test),
        columns=INPUT_FEATURES,
        index=X_test.index,
    )

    n_train = len(X_train_scaled)
    n_test = len(X_test_scaled)
    assert n_train == EXPECTED_TRAIN_ROWS, (
        f"Expected {EXPECTED_TRAIN_ROWS} training rows; got {n_train}."
    )
    assert n_test == EXPECTED_TEST_ROWS, (
        f"Expected {EXPECTED_TEST_ROWS} test rows; got {n_test}."
    )

    y_train_dict: Dict[str, np.ndarray] = {}
    y_test_dict: Dict[str, np.ndarray] = {}
    for target_name in ALL_TARGETS:
        y_train_dict[target_name] = df.loc[idx_train, target_name].to_numpy()
        y_test_dict[target_name] = df.loc[idx_test, target_name].to_numpy()

    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(X_train_scaled, os.path.join(output_dir, "X_train_scaled.pkl"))
    joblib.dump(X_test_scaled, os.path.join(output_dir, "X_test_scaled.pkl"))
    joblib.dump(y_train_dict, os.path.join(output_dir, "y_train_dict.pkl"))
    joblib.dump(y_test_dict, os.path.join(output_dir, "y_test_dict.pkl"))
    joblib.dump(INPUT_FEATURES, os.path.join(output_dir, "input_features.pkl"))
    joblib.dump(ALL_TARGETS, os.path.join(output_dir, "all_targets.pkl"))

    train_outage = df.loc[idx_train, "Outage SG"]
    test_outage = df.loc[idx_test, "Outage SG"]

    print("\nPreprocessing summary")
    print(f"Raw rows:                   {n_raw}")
    print(f"Zero-disturbance removed:   {n_zero}")
    print(f"Sparse-tail removed:        {n_sparse}")
    print(f"Final rows:                 {n_clean}")
    print(f"Train size:                 {n_train}")
    print(f"Test size:                  {n_test}")
    print(f"Total targets:              {len(ALL_TARGETS)}")
    print(f"Canonical target order:     {ALL_TARGETS}")
    print("Outage SG - Train:", train_outage.value_counts().sort_index().to_dict())
    print("Outage SG - Test: ", test_outage.value_counts().sort_index().to_dict())

    return X_train_scaled, X_test_scaled, y_train_dict, y_test_dict, scaler_X


if __name__ == "__main__":
    default_csv = os.path.join(
        "data",
        "raw",
        "Nine_bus_system_frequency_response_N_minus_one_disturbance.csv",
    )
    default_out = os.path.join("data", "processed")
    load_and_preprocess(default_csv, default_out)