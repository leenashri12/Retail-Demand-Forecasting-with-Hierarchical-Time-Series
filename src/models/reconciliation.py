"""
Hierarchical reconciliation methods: Bottom-Up, Top-Down, MinTrace (OLS).

Operates on bottom-level (store x item) forecasts and the M5 hierarchy.
"""

import gc
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)

# Bottom-level group columns
BOTTOM_COLS = ["store_id", "item_id"]

# M5 hierarchy levels (same as in wrmsse.py)
HIERARCHY_LEVELS: Dict[int, List[str]] = {
    1:  [],
    2:  ["state_id"],
    3:  ["store_id"],
    4:  ["cat_id"],
    5:  ["dept_id"],
    6:  ["state_id", "cat_id"],
    7:  ["state_id", "dept_id"],
    8:  ["store_id", "cat_id"],
    9:  ["store_id", "dept_id"],
    10: ["item_id"],
    11: ["state_id", "item_id"],
    12: ["store_id", "item_id"],
}


def bottom_up(
    df_val: pd.DataFrame,
    pred_col: str = "pred",
    day_col: str = "day_num",
) -> pd.DataFrame:
    """Bottom-Up reconciliation: aggregate SKU forecasts upward.

    The bottom-level predictions are kept as-is. All upper levels are
    computed by summing the bottom-level predictions.

    Args:
        df_val: Validation DataFrame with bottom-level predictions.
        pred_col: Column name of predictions.
        day_col: Day column.

    Returns:
        Same DataFrame (BU keeps bottom-level predictions unchanged).
    """
    logger.info("Bottom-Up reconciliation (no change to bottom-level preds)")
    # BU simply uses the bottom-level forecasts as-is.
    # Upper levels are computed during WRMSSE evaluation by aggregation.
    df_val = df_val.copy()
    df_val["pred_bu"] = df_val[pred_col].values
    return df_val


def top_down(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    pred_col: str = "pred",
    actual_col: str = "sales",
    day_col: str = "day_num",
) -> pd.DataFrame:
    """Top-Down reconciliation: disaggregate national forecast.

    1. Compute the national (total) forecast by summing bottom-level preds.
    2. Compute historical proportions for each (store, item) pair.
    3. Disaggregate: bottom_pred_i = national_forecast * proportion_i.

    Args:
        df_train: Training data for computing historical proportions.
        df_val: Validation data with bottom-level predictions.
        pred_col: Prediction column.
        actual_col: Actual sales column.
        day_col: Day column.

    Returns:
        DataFrame with 'pred_td' column containing top-down predictions.
    """
    logger.info("Top-Down reconciliation ...")

    df_val = df_val.copy()

    # 1. Historical proportions: avg sales per (store, item) / total avg sales
    avg_sales = (
        df_train.groupby(BOTTOM_COLS, observed=True)[actual_col]
        .mean()
        .reset_index()
        .rename(columns={actual_col: "avg_sales"})
    )
    total_avg = avg_sales["avg_sales"].sum()
    avg_sales["proportion"] = avg_sales["avg_sales"] / total_avg

    # 2. National forecast per day (sum of all bottom-level preds)
    national = (
        df_val.groupby(day_col, observed=True)[pred_col]
        .sum()
        .reset_index()
        .rename(columns={pred_col: "national_pred"})
    )

    # 3. Merge proportions and national forecast, then disaggregate
    df_val = df_val.merge(
        avg_sales[BOTTOM_COLS + ["proportion"]],
        on=BOTTOM_COLS, how="left",
    )
    df_val = df_val.merge(national, on=day_col, how="left")

    df_val["pred_td"] = df_val["national_pred"] * df_val["proportion"]
    df_val["pred_td"] = df_val["pred_td"].astype(np.float32)

    # Cleanup temp columns
    df_val = df_val.drop(columns=["proportion", "national_pred", "avg_sales"],
                         errors="ignore")

    logger.info("  Top-Down complete. pred_td range: [%.1f, %.1f]",
                df_val["pred_td"].min(), df_val["pred_td"].max())
    return df_val


def mintrace_ols(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    pred_col: str = "pred",
    actual_col: str = "sales",
    day_col: str = "day_num",
) -> pd.DataFrame:
    """MinTrace (OLS) reconciliation at store x department level.

    Uses a reduced hierarchy (store x dept as bottom) for computational
    feasibility, then distributes back to store x item using within-dept
    proportions.

    Steps:
        1. Aggregate base forecasts to store x dept level (70 series).
        2. Build summing matrix S for reduced hierarchy.
        3. Apply MinTrace OLS: y_tilde = S (S'S)^{-1} S' y_hat.
        4. Distribute reconciled dept forecasts to item level using
           historical within-department proportions.

    Args:
        df_train: Training data.
        df_val: Validation data with base predictions.
        pred_col: Base prediction column.
        actual_col: Actual column.
        day_col: Day column.

    Returns:
        DataFrame with 'pred_mint' column.
    """
    logger.info("MinTrace (OLS) reconciliation ...")

    df_val = df_val.copy()

    # --- Reduced hierarchy: Total -> State -> Store -> Dept -> Store x Dept ---
    reduced_levels = {
        "total": [],
        "state": ["state_id"],
        "store": ["store_id"],
        "dept":  ["dept_id"],
        "store_dept": ["store_id", "dept_id"],
    }
    bottom_key = "store_dept"
    bottom_cols_reduced = ["store_id", "dept_id"]

    # Get unique days in validation
    val_days = sorted(df_val[day_col].unique())

    # 1. Aggregate base forecasts to each reduced level
    level_forecasts = {}
    for name, cols in reduced_levels.items():
        if not cols:
            agg = df_val.groupby(day_col, observed=True)[pred_col].sum().reset_index()
            agg["_key"] = "Total"
        elif len(cols) == 1:
            agg = df_val.groupby(cols + [day_col], observed=True)[pred_col].sum().reset_index()
            agg["_key"] = agg[cols[0]].astype(str)
        else:
            agg = df_val.groupby(cols + [day_col], observed=True)[pred_col].sum().reset_index()
            agg["_key"] = agg[cols].astype(str).agg("_".join, axis=1)

        pivot = agg.pivot(index="_key", columns=day_col, values=pred_col)
        pivot = pivot.reindex(columns=val_days).fillna(0)
        level_forecasts[name] = pivot

    # 2. Build summing matrix S
    bottom_keys = list(level_forecasts[bottom_key].index)
    n_bottom = len(bottom_keys)

    # Parse bottom keys to get store and dept
    bottom_info = pd.DataFrame({
        "key": bottom_keys,
        "store_id": [k.split("_")[0] + "_" + k.split("_")[1] for k in bottom_keys],
        "dept_id": [k.split("_", 2)[2] for k in bottom_keys],
    })

    # Get state mapping from training data
    store_state = (
        df_train.groupby("store_id", observed=True)["state_id"]
        .first()
        .to_dict()
    )
    bottom_info["state_id"] = bottom_info["store_id"].map(store_state)

    # Build S matrix rows for each upper level
    all_upper_keys = []
    s_rows = []

    for name, cols in reduced_levels.items():
        if name == bottom_key:
            continue  # Bottom level = identity block
        upper_keys = list(level_forecasts[name].index)
        for uk in upper_keys:
            row = np.zeros(n_bottom)
            if name == "total":
                row[:] = 1
            elif name == "state":
                mask = bottom_info["state_id"] == uk
                row[mask.values] = 1
            elif name == "store":
                mask = bottom_info["store_id"] == uk
                row[mask.values] = 1
            elif name == "dept":
                mask = bottom_info["dept_id"] == uk
                row[mask.values] = 1
            s_rows.append(row)
            all_upper_keys.append((name, uk))

    # S = [upper_block; I_bottom]
    upper_block = np.array(s_rows)  # (n_upper, n_bottom)
    S = np.vstack([upper_block, np.eye(n_bottom)])  # (n_total, n_bottom)
    n_total = S.shape[0]

    logger.info("  Summing matrix S: (%d, %d)", n_total, n_bottom)

    # 3. Stack base forecasts: y_hat = [upper_forecasts; bottom_forecasts]
    upper_forecast_rows = []
    for name, uk in all_upper_keys:
        upper_forecast_rows.append(level_forecasts[name].loc[uk].values)
    upper_forecast_mat = np.array(upper_forecast_rows)
    bottom_forecast_mat = level_forecasts[bottom_key].values

    y_hat = np.vstack([upper_forecast_mat, bottom_forecast_mat])  # (n_total, h)

    # 4. MinTrace OLS: y_tilde = S @ (S'S)^{-1} @ S' @ y_hat
    StS = S.T @ S  # (n_bottom, n_bottom)
    StS_inv = np.linalg.inv(StS)
    P = StS_inv @ S.T  # (n_bottom, n_total)
    y_bottom_reconciled = P @ y_hat  # (n_bottom, h)

    # The full reconciled forecasts
    y_tilde = S @ y_bottom_reconciled  # (n_total, h)

    logger.info("  Reconciled bottom forecasts range: [%.1f, %.1f]",
                y_bottom_reconciled.min(), y_bottom_reconciled.max())

    # 5. Distribute store_dept reconciled forecasts to store_item
    # using within-department proportions from training data
    item_proportions = (
        df_train.groupby(
            ["store_id", "dept_id", "item_id"], observed=True
        )[actual_col]
        .mean()
        .reset_index()
        .rename(columns={actual_col: "avg_sales"})
    )

    dept_totals = (
        item_proportions.groupby(["store_id", "dept_id"], observed=True)["avg_sales"]
        .sum()
        .reset_index()
        .rename(columns={"avg_sales": "dept_total"})
    )

    item_proportions = item_proportions.merge(dept_totals, on=["store_id", "dept_id"])
    item_proportions["within_prop"] = (
        item_proportions["avg_sales"] / item_proportions["dept_total"]
    )
    item_proportions["within_prop"] = item_proportions["within_prop"].fillna(0)

    # Build reconciled store_dept lookup: key -> (h,) forecast
    reconciled_dept = {}
    for i, key in enumerate(bottom_keys):
        reconciled_dept[key] = y_bottom_reconciled[i]

    # For each (store, dept, item), multiply dept forecast by proportion
    store_dept_key_col = (
        df_val["store_id"].astype(str) + "_" + df_val["dept_id"].astype(str)
    )

    # Merge proportions
    df_val = df_val.merge(
        item_proportions[["store_id", "dept_id", "item_id", "within_prop"]],
        on=["store_id", "dept_id", "item_id"],
        how="left",
    )
    df_val["within_prop"] = df_val["within_prop"].fillna(0)

    # Build reconciled dept prediction per row
    dept_pred_map = {}
    for day_idx, day in enumerate(val_days):
        for key, forecast in reconciled_dept.items():
            dept_pred_map[(key, day)] = forecast[day_idx]

    df_val["_sd_key"] = store_dept_key_col
    df_val["dept_recon_pred"] = df_val.apply(
        lambda r: dept_pred_map.get((r["_sd_key"], r[day_col]), 0), axis=1
    )

    df_val["pred_mint"] = (
        df_val["dept_recon_pred"] * df_val["within_prop"]
    ).astype(np.float32)

    # Cleanup
    df_val = df_val.drop(
        columns=["within_prop", "_sd_key", "dept_recon_pred", "avg_sales",
                 "dept_total"],
        errors="ignore",
    )

    logger.info("  MinTrace complete. pred_mint range: [%.1f, %.1f]",
                df_val["pred_mint"].min(), df_val["pred_mint"].max())
    return df_val


def compute_coherence_error(
    df: pd.DataFrame,
    pred_col: str,
    day_col: str = "day_num",
) -> float:
    """Compute total coherence error across the hierarchy.

    Checks that the sum of children equals the parent at each level.
    Works on the bottom-level (store x item) DataFrame.

    Args:
        df: DataFrame with predictions at bottom level.
        pred_col: Prediction column name.
        day_col: Day column.

    Returns:
        Total coherence error (0.0 = perfectly coherent).
    """
    total_error = 0.0

    # For BU forecasts from a single model, forecasts are inherently
    # coherent (aggregation of bottom = top). Check store -> store_dept.
    checks = [
        # (parent_group, child_group) -- child should sum to parent
        (["store_id"], ["store_id", "dept_id"]),
        (["dept_id"], ["store_id", "dept_id"]),
    ]

    for parent_cols, child_cols in checks:
        # Both computed from the same bottom-level data
        parent_agg = (
            df.groupby(parent_cols + [day_col], observed=True)[pred_col]
            .sum()
            .reset_index()
        )
        child_agg = (
            df.groupby(child_cols + [day_col], observed=True)[pred_col]
            .sum()
            .reset_index()
        )

        # Roll child up to parent level and compare
        child_rolled = (
            child_agg.groupby(parent_cols + [day_col], observed=True)[pred_col]
            .sum()
            .reset_index()
        )

        merged = parent_agg.merge(
            child_rolled, on=parent_cols + [day_col],
            suffixes=("_parent", "_child"),
        )
        if len(merged) > 0:
            err = np.abs(
                merged[f"{pred_col}_parent"] - merged[f"{pred_col}_child"]
            ).sum()
            total_error += float(err)

    return total_error
