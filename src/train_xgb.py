import csv
import json
import os
import time
import traceback
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


# Fixed settings used for reproducibility and consistent evaluation.
RANDOM_STATE = 42
CV_FOLDS = 5
REPORT_R2_THRESHOLD = 0.96
UNDERESTIMATE_BREACH = 1.0
OVERFIT_THRESHOLD = 0.03


# Default target order used if the preprocessing artefact is not available.
DEFAULT_TARGETS: List[str] = (
    [f"RoCoF Bus {i}" for i in range(1, 10)]
    + ["RoCoF Worst"]
    + [f"Nadir Bus {i}" for i in range(1, 10)]
    + ["Nadir Worst"]
)


# XGBoost uses one fixed grid search phase for every target.
# Internal n_jobs is kept at 1 so GridSearchCV controls the outer parallelism.
XGB_PARAM_GRID = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "reg_alpha": [0],
    "reg_lambda": [1],
    "random_state": [RANDOM_STATE],
    "tree_method": ["hist"],
    "n_jobs": [1],
}
# 3 x 4 x 3 x 2 x 2 x 2 = 144 parameter combinations.


def compute_metrics(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    cv_r2: float,
) -> Dict[str, float]:
    # Accuracy metrics reported in the dissertation tables.
    r2_test = float(r2_score(y_test, y_pred_test))
    r2_train = float(r2_score(y_train, y_pred_train))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))

    # Positive residuals represent optimistic predictions for both target types.
    # This gives the Maximum Underestimate metric used in the safety discussion.
    residuals = y_pred_test - y_test
    max_ae = float(np.max(np.abs(residuals)))
    mu = float(np.max(residuals)) if np.any(residuals > 0) else 0.0
    mo = float(np.max(-residuals)) if np.any(residuals < 0) else 0.0

    return {
        "r2_test": r2_test,
        "r2_train": r2_train,
        "r2_cv": cv_r2,
        "rmse": rmse,
        "mae": mae,
        "max_absolute_error": max_ae,
        "max_underestimate": mu,
        "max_overestimate": mo,
    }


def _json_convert(value: Any) -> Any:
    # Convert numpy objects before writing results to JSON.
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_target_list(data_dir: str) -> List[str]:
    # Use the preprocessing target order to keep all output tables aligned.
    pkl_path = os.path.join(data_dir, "all_targets.pkl")
    if os.path.isfile(pkl_path):
        loaded = joblib.load(pkl_path)
        if isinstance(loaded, list) and loaded:
            return list(loaded)
    return list(DEFAULT_TARGETS)


def _save_xgb_model(model: XGBRegressor, path_without_ext: str) -> str:
    # Native XGBoost format is used for later SHAP loading.
    ubj_path = f"{path_without_ext}.ubj"
    model.get_booster().save_model(ubj_path)
    return ubj_path


def train_single_target(
    target_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    results_so_far: Dict[str, Dict[str, Any]],
    tables_dir: str,
    index: int,
    targets: List[str],
) -> Dict[str, Any]:
    n_total = len(targets)

    # Count the total number of fitted models for the training log.
    n_combos = 1
    for values in XGB_PARAM_GRID.values():
        n_combos *= len(values)
    n_fits = n_combos * CV_FOLDS

    print(f"\n{'=' * 60}")
    print(f"[{index}/{n_total}] Training XGBoost for {target_name}")
    print(f"{'=' * 60}")
    print(f"  Grid: {n_combos} combinations x {CV_FOLDS}-fold CV = {n_fits} fits")

    try:
        # Ensure consistent numeric dtype after loading arrays from joblib.
        y_train = np.asarray(y_train, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)

        start_time = time.time()
        grid = GridSearchCV(
            XGBRegressor(),
            XGB_PARAM_GRID,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        elapsed = time.time() - start_time

        best_model = grid.best_estimator_
        cv_r2 = float(grid.best_score_)
        y_pred_test = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)
        metrics = compute_metrics(y_test, y_pred_test, y_train, y_pred_train, cv_r2)

        print(f"  Complete in {elapsed:.1f}s")
        print(
            f"  CV R2 = {metrics['r2_cv']:.4f} | "
            f"Test R2 = {metrics['r2_test']:.4f} | "
            f"Train R2 = {metrics['r2_train']:.4f}"
        )
        print(
            f"  RMSE = {metrics['rmse']:.4f} | "
            f"MAE = {metrics['mae']:.4f} | "
            f"MU = {metrics['max_underestimate']:.4f}"
        )

        # Train-test R2 gap is recorded as a simple overfitting diagnostic.
        gap = metrics["r2_train"] - metrics["r2_test"]
        overfit_flag = gap > OVERFIT_THRESHOLD
        if overfit_flag:
            print(
                f"  WARNING: train R2 ({metrics['r2_train']:.4f}) exceeds "
                f"test R2 ({metrics['r2_test']:.4f}) by {gap:.4f}, "
                f"above threshold {OVERFIT_THRESHOLD}"
            )

        result = {
            "algorithm": "XGBoost",
            "target": target_name,
            "best_params": grid.best_params_,
            "r2_cv": cv_r2,
            "r2_test": metrics["r2_test"],
            "r2_train": metrics["r2_train"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "max_absolute_error": metrics["max_absolute_error"],
            "max_underestimate": metrics["max_underestimate"],
            "max_overestimate": metrics["max_overestimate"],
            "train_time_seconds": elapsed,
            "overfit_flag": bool(overfit_flag),
            "status": "OK",
        }

        safe_name = target_name.replace(" ", "_")
        model_path_base = os.path.join(output_dir, safe_name)
        saved_path = _save_xgb_model(best_model, model_path_base)
        print(f"  Model saved: {saved_path}")

        prediction_path = os.path.join(output_dir, f"{safe_name}_predictions.npz")
        np.savez(
            prediction_path,
            y_test=y_test,
            y_pred_test=y_pred_test,
            y_train=y_train,
            y_pred_train=y_pred_train,
        )

        # Save progress after each target so partial runs are recoverable.
        results_so_far[target_name] = result
        json_path = os.path.join(output_dir, "xgb_results.json")
        with open(json_path, "w", encoding="utf-8") as file_obj:
            json.dump(results_so_far, file_obj, indent=2, default=_json_convert)

        _append_result_to_csv(result, tables_dir)

        print("-" * 60)
        return result

    except Exception as exc:  # noqa: BLE001 - continue training remaining targets
        err_msg = f"{exc}\n{traceback.format_exc()}"
        print(f"  ERROR: {exc}")
        print(traceback.format_exc())

        # Record the failed target instead of terminating the full training run.
        result = {
            "algorithm": "XGBoost",
            "target": target_name,
            "error": err_msg,
            "status": "ERROR",
        }
        results_so_far[target_name] = result
        json_path = os.path.join(output_dir, "xgb_results.json")
        with open(json_path, "w", encoding="utf-8") as file_obj:
            json.dump(results_so_far, file_obj, indent=2, default=_json_convert)
        print("-" * 60)
        return result


def _append_result_to_csv(result: Dict[str, Any], tables_dir: str) -> None:
    # Append one row at a time during training.
    csv_path = os.path.join(tables_dir, "xgb_results.csv")
    file_exists = os.path.isfile(csv_path)
    fieldnames = [
        "target",
        "r2_cv",
        "r2_test",
        "r2_train",
        "rmse",
        "mae",
        "max_absolute_error",
        "max_underestimate",
        "max_overestimate",
        "train_time_seconds",
        "overfit_flag",
        "status",
    ]
    row = {key: result.get(key) for key in fieldnames}
    row["target"] = result["target"]
    row["status"] = result.get("status", "OK")

    with open(csv_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_full_csv(
    results: Dict[str, Dict[str, Any]],
    tables_dir: str,
    targets: List[str],
) -> None:
    # Rewrite the final table in canonical target order.
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "xgb_results.csv")
    fieldnames = [
        "target",
        "r2_cv",
        "r2_test",
        "r2_train",
        "rmse",
        "mae",
        "max_absolute_error",
        "max_underestimate",
        "max_overestimate",
        "train_time_seconds",
        "overfit_flag",
        "status",
        "error",
    ]

    rows = []
    for target_name in targets:
        result = results.get(target_name, {})
        row = {key: result.get(key) for key in fieldnames}
        row["target"] = target_name
        row["status"] = result.get("status", "ERROR" if "error" in result else "OK")
        row["error"] = result.get("error", "")
        rows.append(row)

    if not rows:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_all_xgb(
    data_dir: str = "data/processed",
    output_dir: str = "models/XGBoost",
    tables_dir: str = "tables",
) -> Dict[str, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Clear only the XGBoost summary from a previous run.
    csv_path = os.path.join(tables_dir, "xgb_results.csv")
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    X_train = joblib.load(os.path.join(data_dir, "X_train_scaled.pkl"))
    X_test = joblib.load(os.path.join(data_dir, "X_test_scaled.pkl"))
    y_train_dict = joblib.load(os.path.join(data_dir, "y_train_dict.pkl"))
    y_test_dict = joblib.load(os.path.join(data_dir, "y_test_dict.pkl"))
    targets = _load_target_list(data_dir)

    # Fail early if the target order and saved target dictionaries are inconsistent.
    missing = [name for name in targets if name not in y_train_dict or name not in y_test_dict]
    if missing:
        raise KeyError(
            f"Target list includes names missing from y_train/y_test dicts: {missing}. "
            "Re-run preprocessing.py to regenerate artefacts."
        )

    results: Dict[str, Dict[str, Any]] = {}
    overall_start = time.time()

    for index, target_name in enumerate(targets, start=1):
        train_single_target(
            target_name,
            X_train,
            X_test,
            y_train_dict[target_name],
            y_test_dict[target_name],
            output_dir,
            results,
            tables_dir,
            index,
            targets,
        )

    total_time = time.time() - overall_start
    _write_full_csv(results, tables_dir, targets)

    n_ok = sum(1 for result in results.values() if "error" not in result)
    n_errors = sum(1 for result in results.values() if "error" in result)
    n_above = sum(
        1
        for result in results.values()
        if "r2_test" in result and result["r2_test"] >= REPORT_R2_THRESHOLD
    )
    n_overfit = sum(1 for result in results.values() if result.get("overfit_flag"))

    breaches = [
        (target_name, result.get("max_underestimate"))
        for target_name, result in results.items()
        if "Nadir" in target_name
        and "max_underestimate" in result
        and result["max_underestimate"] > UNDERESTIMATE_BREACH
    ]

    print("\n")
    print("=" * 70)
    print("                   XGBOOST TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Target':<28} {'CV R2':>8} {'Test R2':>8} {'RMSE':>8}  Status")
    print("-" * 70)
    for target_name in targets:
        result = results.get(target_name, {})
        if "error" in result:
            print(f"{target_name:<28} {'---':>8} {'---':>8} {'---':>8}  ERROR")
        else:
            status = result.get("status", "OK")
            if result.get("overfit_flag"):
                status = f"{status} (overfit)"
            print(
                f"{target_name:<28} "
                f"{result.get('r2_cv', 0):>8.4f} "
                f"{result.get('r2_test', 0):>8.4f} "
                f"{result.get('rmse', 0):>8.4f}  "
                f"{status}"
            )
    print("=" * 70)
    print(f"Total training time: {total_time / 3600:.2f} h")

    if n_ok:
        print(f"Models achieving test R2 >= {REPORT_R2_THRESHOLD}: {n_above}/{n_ok}")
    else:
        print("No models completed.")

    print(f"Overfitting (train - test > {OVERFIT_THRESHOLD}): {n_overfit}/{len(targets)}")

    if n_errors:
        print(f"Errors: {n_errors}/{len(targets)}")

    if breaches:
        print()
        print(f"Nadir targets with max_underestimate > {UNDERESTIMATE_BREACH} Hz:")
        for target_name, value in breaches:
            print(f"  {target_name}: {value:.4f} Hz")

    print()
    return results


if __name__ == "__main__":
    train_all_xgb(
        data_dir="data/processed",
        output_dir="models/XGBoost",
        tables_dir="tables",
    )