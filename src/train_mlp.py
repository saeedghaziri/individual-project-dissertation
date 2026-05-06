import csv
import json
import os
import time
import traceback
import warnings
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


# Convergence warnings are expected during the grid search and do not affect
# model selection, which is based on the recorded cross-validation scores.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Fixed settings used for reproducibility across all MLP training runs.
RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_R2 = 0.96
PHASE2_SEEDS = [42, 123, 456]
UNDERESTIMATE_BREACH = 1.0


# Default target order used if the preprocessing artefact is not available.
DEFAULT_TARGETS: List[str] = (
    [f"RoCoF Bus {i}" for i in range(1, 10)]
    + ["RoCoF Worst"]
    + [f"Nadir Bus {i}" for i in range(1, 10)]
    + ["Nadir Worst"]
)


# RoCoF targets use the smaller Phase 1 grid because they were less difficult
# to fit during the preliminary model-selection runs.
ROCOF_PARAM_GRID = {
    "hidden_layer_sizes": [
        (100, 100, 100),
        (128, 128, 128),
        (200, 100, 50),
    ],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.005],
    "batch_size": [32, 64],
    "activation": ["relu"],
    "solver": ["adam"],
    "max_iter": [2000],
    "early_stopping": [True],
    "validation_fraction": [0.1],
    "n_iter_no_change": [20],
    "random_state": [RANDOM_STATE],
}
# 3 x 2 x 2 x 2 = 24 parameter combinations.


# Nadir targets use a wider Phase 1 grid because their response surface is more
# nonlinear and was harder for the MLP to approximate.
NADIR_PARAM_GRID = {
    "hidden_layer_sizes": [
        (100, 100, 100),
        (200, 200, 200),
        (256, 128, 64),
        (300, 200, 100),
        (200, 200, 200, 200),
    ],
    "alpha": [0.0001, 0.0005, 0.001],
    "learning_rate_init": [0.001, 0.005, 0.01],
    "batch_size": [32, 64],
    "activation": ["relu"],
    "solver": ["adam"],
    "max_iter": [2000],
    "early_stopping": [True],
    "validation_fraction": [0.1],
    "n_iter_no_change": [20],
    "random_state": [RANDOM_STATE],
}
# 5 x 3 x 3 x 2 = 90 parameter combinations.


def get_param_grid(target_name: str) -> Dict[str, List[Any]]:
    # Select the Phase 1 search space according to the target family.
    if "RoCoF" in target_name:
        return ROCOF_PARAM_GRID.copy()
    return NADIR_PARAM_GRID.copy()


def build_refined_grid(best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    # Phase 2 refines the search around the best Phase 1 configuration.
    return {
        "hidden_layer_sizes": [
            best_params["hidden_layer_sizes"],
            (300, 300, 300),
            (400, 300, 200),
            (256, 256, 256, 128),
        ],
        "alpha": [
            best_params["alpha"] / 5,
            best_params["alpha"],
            best_params["alpha"] * 3,
        ],
        "learning_rate_init": [
            best_params["learning_rate_init"] / 2,
            best_params["learning_rate_init"],
            best_params["learning_rate_init"] * 2,
        ],
        "batch_size": [32, 64],
        "activation": ["relu"],
        "solver": ["adam"],
        "max_iter": [3000],
        "early_stopping": [True],
        "validation_fraction": [0.1],
        "n_iter_no_change": [30],
    }


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
    param_grid = get_param_grid(target_name)

    # Count the total number of Phase 1 fits for the training log.
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)
    n_fits_phase1 = n_combos * CV_FOLDS

    print(f"\n{'=' * 60}")
    print(f"[{index}/{n_total}] Training MLP for {target_name} (Phase 1)")
    print(f"{'=' * 60}")
    print(
        f"  Grid: {n_combos} combinations x {CV_FOLDS}-fold CV = "
        f"{n_fits_phase1} fits"
    )

    try:
        # Ensure consistent numeric dtype after loading arrays from joblib.
        y_train = np.asarray(y_train, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)

        phase1_start = time.time()
        grid = GridSearchCV(
            MLPRegressor(),
            param_grid,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        phase1_time = time.time() - phase1_start

        best_model = grid.best_estimator_
        cv_r2 = float(grid.best_score_)
        best_params = grid.best_params_
        refined = False
        total_time = phase1_time

        print(f"  Phase 1 complete in {phase1_time:.1f}s | CV R2 = {cv_r2:.4f}")

        # Phase 2 is only triggered by cross-validation performance, not by
        # test-set performance.
        trigger_phase2 = "Nadir" in target_name and cv_r2 < TARGET_R2
        if trigger_phase2:
            print(
                f"  CV R2 = {cv_r2:.4f} < {TARGET_R2} - "
                "running Phase 2 refined search..."
            )
            refined_grid_template = build_refined_grid(best_params)

            best_cv = cv_r2
            best_candidate_model = best_model
            best_candidate_params = best_params

            phase2_start = time.time()
            for seed in PHASE2_SEEDS:
                # Repeat Phase 2 with different seeds to reduce dependence on
                # one neural-network initialisation.
                refined_grid = dict(refined_grid_template)
                refined_grid["random_state"] = [seed]

                grid2 = GridSearchCV(
                    MLPRegressor(),
                    refined_grid,
                    cv=CV_FOLDS,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=1,
                )
                grid2.fit(X_train, y_train)
                seed_cv = float(grid2.best_score_)

                marker = ""
                if seed_cv > best_cv:
                    best_cv = seed_cv
                    best_candidate_model = grid2.best_estimator_
                    best_candidate_params = grid2.best_params_
                    marker = " <- new best"
                print(f"  [Phase 2] Seed {seed}: CV R2 = {seed_cv:.4f}{marker}")

            phase2_time = time.time() - phase2_start
            total_time = phase1_time + phase2_time
            print(f"  Phase 2 complete in {phase2_time:.1f}s")

            # Retain the Phase 1 model unless Phase 2 improves the CV score.
            if best_cv > cv_r2:
                print(f"  Refined CV R2: {best_cv:.4f} (improved from {cv_r2:.4f})")
                best_model = best_candidate_model
                best_params = best_candidate_params
                cv_r2 = best_cv
                refined = True
            else:
                print("  No Phase 2 candidate improved on Phase 1. Keeping Phase 1 model.")

        # The held-out test set is evaluated only after final model selection.
        y_pred_test = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)
        metrics = compute_metrics(y_test, y_pred_test, y_train, y_pred_train, cv_r2)

        print(
            f"  Final: CV R2 = {metrics['r2_cv']:.4f} | "
            f"Test R2 = {metrics['r2_test']:.4f} | "
            f"Train R2 = {metrics['r2_train']:.4f}"
        )
        print(
            f"         RMSE = {metrics['rmse']:.4f} | "
            f"MAE = {metrics['mae']:.4f} | "
            f"MU = {metrics['max_underestimate']:.4f}"
        )

        result = {
            "algorithm": "MLP",
            "target": target_name,
            "best_params": best_params,
            "r2_cv": cv_r2,
            "r2_test": metrics["r2_test"],
            "r2_train": metrics["r2_train"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "max_absolute_error": metrics["max_absolute_error"],
            "max_underestimate": metrics["max_underestimate"],
            "max_overestimate": metrics["max_overestimate"],
            "train_time_seconds": total_time,
            "refined": refined,
            "status": "Refined" if refined else "Phase 1",
        }

        safe_name = target_name.replace(" ", "_")
        model_path = os.path.join(output_dir, f"{safe_name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"  Model saved: {model_path}")

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
        json_path = os.path.join(output_dir, "mlp_results.json")
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
            "algorithm": "MLP",
            "target": target_name,
            "error": err_msg,
            "status": "ERROR",
        }
        results_so_far[target_name] = result
        json_path = os.path.join(output_dir, "mlp_results.json")
        with open(json_path, "w", encoding="utf-8") as file_obj:
            json.dump(results_so_far, file_obj, indent=2, default=_json_convert)
        print("-" * 60)
        return result


def _append_result_to_csv(result: Dict[str, Any], tables_dir: str) -> None:
    # Append one row at a time during training.
    csv_path = os.path.join(tables_dir, "mlp_results.csv")
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
        "refined",
        "status",
    ]
    row = {key: result.get(key) for key in fieldnames}
    row["target"] = result["target"]
    row["status"] = result.get("status", "Phase 1")

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
    csv_path = os.path.join(tables_dir, "mlp_results.csv")
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
        "refined",
        "status",
        "error",
    ]

    rows = []
    for target_name in targets:
        result = results.get(target_name, {})
        row = {key: result.get(key) for key in fieldnames}
        row["target"] = target_name
        row["status"] = result.get("status", "ERROR" if "error" in result else "Phase 1")
        row["error"] = result.get("error", "")
        rows.append(row)

    if not rows:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_all_mlp(
    data_dir: str = "data/processed",
    output_dir: str = "models/MLP",
    tables_dir: str = "tables",
) -> Dict[str, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Clear only the MLP summary from a previous run.
    csv_path = os.path.join(tables_dir, "mlp_results.csv")
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    X_train = joblib.load(os.path.join(data_dir, "X_train_scaled.pkl"))
    X_test = joblib.load(os.path.join(data_dir, "X_test_scaled.pkl"))
    y_train_dict = joblib.load(os.path.join(data_dir, "y_train_dict.pkl"))
    y_test_dict = joblib.load(os.path.join(data_dir, "y_test_dict.pkl"))
    targets = _load_target_list(data_dir)

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
    n_refined = sum(1 for result in results.values() if result.get("refined"))
    n_above = sum(
        1
        for result in results.values()
        if "r2_test" in result and result["r2_test"] >= TARGET_R2
    )
    breaches = [
        (target_name, result.get("max_underestimate"))
        for target_name, result in results.items()
        if "Nadir" in target_name
        and "max_underestimate" in result
        and result["max_underestimate"] > UNDERESTIMATE_BREACH
    ]

    print("\n")
    print("=" * 70)
    print("                    MLP TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Target':<28} {'CV R2':>8} {'Test R2':>8} {'RMSE':>8}  Status")
    print("-" * 70)
    for target_name in targets:
        result = results.get(target_name, {})
        if "error" in result:
            print(f"{target_name:<28} {'---':>8} {'---':>8} {'---':>8}  ERROR")
        else:
            print(
                f"{target_name:<28} "
                f"{result.get('r2_cv', 0):>8.4f} "
                f"{result.get('r2_test', 0):>8.4f} "
                f"{result.get('rmse', 0):>8.4f}  "
                f"{result.get('status', 'Phase 1')}"
            )
    print("=" * 70)
    print(f"Total training time: {total_time / 3600:.2f} h")

    if n_ok:
        print(f"Models achieving R2 >= {TARGET_R2}: {n_above}/{n_ok}")
    else:
        print("No models completed.")

    print(f"Models requiring refinement: {n_refined}/{len(targets)}")

    if breaches:
        print()
        print(f"Nadir targets with max_underestimate > {UNDERESTIMATE_BREACH} Hz:")
        for target_name, value in breaches:
            print(f"  {target_name}: {value:.4f} Hz")

    print()
    return results


if __name__ == "__main__":
    train_all_mlp(
        data_dir="data/processed",
        output_dir="models/MLP",
        tables_dir="tables",
    )