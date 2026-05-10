"""
Phase 3 - Hierarchical Training & Reconciliation
=================================================
1. Load featured data, split train/val (last 28 days).
2. Train global LightGBM at SKU level, log as baseline_lgbm_sku.
3. Apply Bottom-Up, Top-Down, MinTrace reconciliation.
4. Evaluate WRMSSE + coherence error for each method.
5. Log all runs to MLflow, print comparison table.

Run:  python scripts/run_phase3.py
"""

import gc
import io
import os
import sys
import time
import logging
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import reduce_mem_usage, DATA_DIR
from src.evaluation.wrmsse import compute_full_wrmsse, LEVEL_NAMES
from src.models.reconciliation import (
    bottom_up, top_down, mintrace_ols, compute_coherence_error,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase3")

PROCESSED_DIR = DATA_DIR / "processed"
INPUT_PATH = PROCESSED_DIR / "featured_data.parquet"
SEP = "=" * 72

# Train/Val split boundary
VAL_START_DAY = 1886
VAL_END_DAY = 1913

# Features for LightGBM
CATEGORICAL_FEATURES = ["store_id", "item_id", "cat_id", "dept_id", "state_id"]
DROP_COLS = [
    "id", "d", "date", "sales", "weekday",
    "event_name_1", "event_name_2", "event_type_1", "event_type_2",
]

# LightGBM hyperparameters
LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_estimators": 500,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix and target for LightGBM.

    Args:
        df: Featured DataFrame.

    Returns:
        Tuple of (X, y, feature_names).
    """
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()

    # Ensure categoricals are properly typed for LightGBM
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    y = df["sales"].values.astype(np.float32)
    return X, y, feature_cols


def main() -> None:
    t0 = time.time()

    # ==================================================================
    # 1. LOAD DATA & SPLIT
    # ==================================================================
    print(f"\n{SEP}")
    print("  STEP 1: Load data & train/val split")
    print(SEP)

    log.info("Loading %s ...", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    log.info("Shape: %s  (%.1f MB)", df.shape, mem_mb(df))

    # Fill NaN sell_price with 0 (items not yet available)
    df["sell_price"] = df["sell_price"].fillna(0).astype(np.float32)

    # Split
    df_train = df[df["day_num"] < VAL_START_DAY].copy()
    df_val = df[df["day_num"] >= VAL_START_DAY].copy()
    del df; gc.collect()

    log.info("Train: %s (days < %d)", df_train.shape, VAL_START_DAY)
    log.info("Val:   %s (days %d-%d)", df_val.shape, VAL_START_DAY, VAL_END_DAY)

    print(f"  Train: {len(df_train):,} rows  |  Val: {len(df_val):,} rows")

    # ==================================================================
    # 2. TRAIN LightGBM
    # ==================================================================
    print(f"\n{SEP}")
    print("  STEP 2: Train LightGBM (baseline_lgbm_sku)")
    print(SEP)

    X_train, y_train, feature_names = prepare_features(df_train)
    X_val, y_val, _ = prepare_features(df_val)

    log.info("Feature matrix: %d features", len(feature_names))

    MODEL_PATH = PROCESSED_DIR / "lgbm_model.txt"
    PREDS_PATH = PROCESSED_DIR / "lgbm_preds.npy"

    if MODEL_PATH.exists() and PREDS_PATH.exists():
        log.info("Loading cached model + predictions ...")
        model = lgb.Booster(model_file=str(MODEL_PATH))
        preds = np.load(str(PREDS_PATH))
        train_time = 0.0
    else:
        log.info("Training LightGBM ...")
        train_t0 = time.time()
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=100)],
        )
        train_time = time.time() - train_t0

        # Save model and predictions
        model.booster_.save_model(str(MODEL_PATH))
        preds = model.predict(X_val)
        np.save(str(PREDS_PATH), preds)
        log.info("Saved model + preds to cache.")

    # Predict
    if not hasattr(model, 'predict') or isinstance(model, lgb.Booster):
        preds = model.predict(X_val)
    preds = np.clip(preds, 0, None).astype(np.float32)
    df_val["pred"] = preds

    rmse_val = np.sqrt(np.mean((y_val - preds) ** 2))
    log.info("Validation RMSE: %.4f", rmse_val)
    log.info("Training time: %.1fs", train_time)

    # Feature importance (top 15)
    if isinstance(model, lgb.Booster):
        fi = model.feature_importance(importance_type="split")
        fi_names = model.feature_name()
    else:
        fi = model.feature_importances_
        fi_names = feature_names

    importance = pd.DataFrame({
        "feature": fi_names,
        "importance": fi,
    }).sort_values("importance", ascending=False).head(15)

    print(f"\n  Top 15 features:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:25s}  {row['importance']:>8.0f}")

    # Free training data
    del X_train, y_train, X_val
    gc.collect()

    # ==================================================================
    # 3. COMPUTE BASELINE WRMSSE
    # ==================================================================
    print(f"\n{SEP}")
    print("  STEP 3: Baseline WRMSSE (before reconciliation)")
    print(SEP)

    baseline_scores = compute_full_wrmsse(df_train, df_val, pred_col="pred")
    baseline_coherence = compute_coherence_error(df_val, "pred")
    log.info("Baseline coherence error: %.4f", baseline_coherence)

    # ==================================================================
    # 4. RECONCILIATION STUDY
    # ==================================================================
    # --- 4a: Bottom-Up ---
    print(f"\n{SEP}")
    print("  STEP 4a: Bottom-Up Reconciliation")
    print(SEP)

    df_val = bottom_up(df_val, pred_col="pred")
    bu_scores = compute_full_wrmsse(df_train, df_val, pred_col="pred_bu")
    bu_coherence = compute_coherence_error(df_val, "pred_bu")
    log.info("BU coherence error: %.4f", bu_coherence)

    # --- 4b: Top-Down ---
    print(f"\n{SEP}")
    print("  STEP 4b: Top-Down Reconciliation")
    print(SEP)

    df_val = top_down(df_train, df_val, pred_col="pred")
    td_scores = compute_full_wrmsse(df_train, df_val, pred_col="pred_td")
    td_coherence = compute_coherence_error(df_val, "pred_td")
    log.info("TD coherence error: %.4f", td_coherence)

    # --- 4c: MinTrace ---
    print(f"\n{SEP}")
    print("  STEP 4c: MinTrace (OLS) Reconciliation")
    print(SEP)

    df_val = mintrace_ols(df_train, df_val, pred_col="pred")
    mint_scores = compute_full_wrmsse(df_train, df_val, pred_col="pred_mint")
    mint_coherence = compute_coherence_error(df_val, "pred_mint")
    log.info("MinTrace coherence error: %.4f", mint_coherence)

    # ==================================================================
    # 5. LOG TO MLFLOW
    # ==================================================================
    print(f"\n{SEP}")
    print("  STEP 5: Log to MLflow")
    print(SEP)

    mlflow_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("M5_Hierarchical_Forecasting")

    # --- Baseline run ---
    with mlflow.start_run(run_name="baseline_lgbm_sku"):
        mlflow.log_params(LGB_PARAMS)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_rows", len(df_train))
        mlflow.log_param("val_rows", len(df_val))
        mlflow.log_metric("rmse", rmse_val)
        mlflow.log_metric("training_time_sec", train_time)
        mlflow.log_metric("coherence_error", baseline_coherence)
        for k, v in baseline_scores.items():
            mlflow.log_metric(k, v)
        log.info("  Logged: baseline_lgbm_sku")

    # --- Bottom-Up run ---
    with mlflow.start_run(run_name="reconciliation_bottom_up"):
        mlflow.log_param("method", "BU")
        mlflow.log_param("description", "Aggregate SKU forecasts upward")
        mlflow.log_metric("coherence_error", bu_coherence)
        for k, v in bu_scores.items():
            mlflow.log_metric(k, v)
        log.info("  Logged: reconciliation_bottom_up")

    # --- Top-Down run ---
    with mlflow.start_run(run_name="reconciliation_top_down"):
        mlflow.log_param("method", "TD")
        mlflow.log_param("description", "National forecast disaggregated by proportions")
        mlflow.log_metric("coherence_error", td_coherence)
        for k, v in td_scores.items():
            mlflow.log_metric(k, v)
        log.info("  Logged: reconciliation_top_down")

    # --- MinTrace run ---
    with mlflow.start_run(run_name="reconciliation_mintrace"):
        mlflow.log_param("method", "MinT_OLS")
        mlflow.log_param("description", "MinTrace OLS at store_x_dept, distributed to item")
        mlflow.log_metric("coherence_error", mint_coherence)
        for k, v in mint_scores.items():
            mlflow.log_metric(k, v)
        log.info("  Logged: reconciliation_mintrace")

    # ==================================================================
    # 6. COMPARISON TABLE
    # ==================================================================
    print(f"\n{SEP}")
    print("  WRMSSE COMPARISON ACROSS ALL METHODS")
    print(SEP)

    methods = {
        "Baseline": baseline_scores,
        "Bottom-Up": bu_scores,
        "Top-Down": td_scores,
        "MinTrace": mint_scores,
    }
    coherence = {
        "Baseline": baseline_coherence,
        "Bottom-Up": bu_coherence,
        "Top-Down": td_coherence,
        "MinTrace": mint_coherence,
    }

    # Header
    header = f"  {'Level':20s}"
    for m in methods:
        header += f"  {m:>12s}"
    print(header)
    print(f"  {'-'*20}" + f"  {'-'*12}" * len(methods))

    # Per-level rows
    for lvl in range(1, 13):
        key = f"wrmsse_level_{lvl}"
        name = LEVEL_NAMES[lvl]
        row = f"  {lvl:2d}. {name:16s}"
        for m_name, scores in methods.items():
            row += f"  {scores[key]:>12.6f}"
        print(row)

    # Overall
    print(f"  {'-'*20}" + f"  {'-'*12}" * len(methods))
    row = f"  {'OVERALL':20s}"
    for m_name, scores in methods.items():
        row += f"  {scores['wrmsse_overall']:>12.6f}"
    print(row)

    # Coherence error
    row = f"  {'COHERENCE ERROR':20s}"
    for m_name in methods:
        row += f"  {coherence[m_name]:>12.4f}"
    print(row)

    # Best method
    best = min(methods, key=lambda m: methods[m]["wrmsse_overall"])
    print(f"\n  BEST METHOD: {best} (WRMSSE = {methods[best]['wrmsse_overall']:.6f})")

    elapsed = time.time() - t0
    print(f"  Total Phase 3 time: {elapsed:.1f}s")

    # Cleanup
    del df_train, df_val
    gc.collect()

    print(f"\n{SEP}")
    print("  PHASE 3 COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
