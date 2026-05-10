"""
WRMSSE - Weighted Root Mean Scaled Squared Error.

Official M5 competition metric implementation.
Weights errors by dollar sales and scales by historical variability.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# M5 hierarchy: 12 aggregation levels
HIERARCHY_LEVELS: Dict[int, List[str]] = {
    1:  [],                              # Total
    2:  ["state_id"],                    # State
    3:  ["store_id"],                    # Store
    4:  ["cat_id"],                      # Category
    5:  ["dept_id"],                     # Department
    6:  ["state_id", "cat_id"],          # State x Category
    7:  ["state_id", "dept_id"],         # State x Department
    8:  ["store_id", "cat_id"],          # Store x Category
    9:  ["store_id", "dept_id"],         # Store x Department
    10: ["item_id"],                     # Item
    11: ["state_id", "item_id"],         # State x Item
    12: ["store_id", "item_id"],         # Store x Item (bottom)
}

LEVEL_NAMES: Dict[int, str] = {
    1: "Total", 2: "State", 3: "Store", 4: "Category",
    5: "Department", 6: "State_x_Cat", 7: "State_x_Dept",
    8: "Store_x_Cat", 9: "Store_x_Dept", 10: "Item",
    11: "State_x_Item", 12: "Store_x_Item",
}


def _compute_scale(
    train_sales: np.ndarray,
) -> np.ndarray:
    """Compute naive-method scale factor for each series.

    scale_i = (1/(n-1)) * sum_{t=2}^{n} (y_t - y_{t-1})^2

    Args:
        train_sales: (n_series, n_train_days) array of training actuals.

    Returns:
        (n_series,) array of scale factors.
    """
    diffs = np.diff(train_sales, axis=1)
    scale = np.mean(diffs ** 2, axis=1)
    # Avoid division by zero for constant series
    scale = np.where(scale == 0, 1.0, scale)
    return scale


def _compute_weights(
    train_sales: np.ndarray,
    train_prices: np.ndarray,
) -> np.ndarray:
    """Compute dollar-sales weights for each series.

    w_i = sum(price_i * sales_i) / sum_j(sum(price_j * sales_j))

    Args:
        train_sales: (n_series, n_train_days).
        train_prices: (n_series, n_train_days).

    Returns:
        (n_series,) array of weights summing to 1.
    """
    dollar_sales = np.nansum(train_sales * train_prices, axis=1)
    total = dollar_sales.sum()
    if total == 0:
        return np.ones(len(dollar_sales)) / len(dollar_sales)
    return dollar_sales / total


def _rmsse_per_series(
    actuals: np.ndarray,
    preds: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Compute RMSSE for each series.

    RMSSE_i = sqrt( (1/h) * sum(y - y_hat)^2 / scale_i )

    Args:
        actuals: (n_series, h) actual values.
        preds: (n_series, h) predicted values.
        scale: (n_series,) scale factors.

    Returns:
        (n_series,) RMSSE values.
    """
    h = actuals.shape[1]
    sse = np.sum((actuals - preds) ** 2, axis=1)
    rmsse = np.sqrt(sse / (h * scale))
    return rmsse


def compute_wrmsse_level(
    actuals: np.ndarray,
    preds: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute WRMSSE for a single hierarchy level.

    WRMSSE_l = sqrt( sum_i(w_i * RMSSE_i^2) )

    Args:
        actuals: (n_series, h).
        preds: (n_series, h).
        scale: (n_series,) from training data.
        weights: (n_series,) dollar-sales weights.

    Returns:
        WRMSSE value for this level.
    """
    rmsse = _rmsse_per_series(actuals, preds, scale)
    wrmsse = np.sqrt(np.sum(weights * rmsse ** 2))
    return float(wrmsse)


def aggregate_to_level(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    day_col: str = "day_num",
) -> pd.DataFrame:
    """Aggregate a long-format DataFrame to a hierarchy level.

    Args:
        df: Long-format DataFrame with value_col and day_col.
        group_cols: Columns defining the aggregation level.
        value_col: Column to sum (e.g. 'sales' or 'pred').
        day_col: Day identifier column.

    Returns:
        Aggregated DataFrame with group_cols + day_col + value_col.
    """
    if not group_cols:
        # Total level: sum across everything per day
        agg = df.groupby(day_col, observed=True)[value_col].sum().reset_index()
        agg["_total"] = "Total"
        return agg
    else:
        return (
            df.groupby(group_cols + [day_col], observed=True)[value_col]
            .sum()
            .reset_index()
        )


def pivot_to_matrix(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    day_col: str = "day_num",
) -> Tuple[np.ndarray, list]:
    """Pivot aggregated data into (n_series, n_days) matrix.

    Args:
        df: Aggregated DataFrame.
        group_cols: Group columns (or ["_total"] for level 1).
        value_col: Value column name.
        day_col: Day column name.

    Returns:
        Tuple of (matrix, series_keys).
    """
    if not group_cols:
        group_cols = ["_total"]

    if len(group_cols) == 1:
        pivot = df.pivot(index=group_cols[0], columns=day_col, values=value_col)
    else:
        df = df.copy()
        df["_key"] = df[group_cols].astype(str).agg("_".join, axis=1)
        pivot = df.pivot(index="_key", columns=day_col, values=value_col)

    pivot = pivot.sort_index(axis=1)
    series_keys = list(pivot.index)
    matrix = pivot.values.astype(np.float64)
    return matrix, series_keys


def compute_full_wrmsse(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    pred_col: str = "pred",
    actual_col: str = "sales",
    price_col: str = "sell_price",
    day_col: str = "day_num",
) -> Dict[str, float]:
    """Compute WRMSSE across all 12 M5 hierarchy levels.

    Args:
        df_train: Training data (long format) with actual_col, price_col.
        df_val: Validation data (long format) with actual_col, pred_col.
        pred_col: Column with predictions.
        actual_col: Target column.
        price_col: Price column for dollar-weights.
        day_col: Day identifier column.

    Returns:
        Dict with keys: 'wrmsse_overall', 'wrmsse_level_1' .. 'wrmsse_level_12',
        and 'level_name_1' .. 'level_name_12'.
    """
    logger.info("Computing WRMSSE across 12 hierarchy levels ...")

    # Fill NaN prices with 0 for weight computation
    df_train = df_train.copy()
    df_val = df_val.copy()
    if price_col in df_train.columns:
        df_train[price_col] = df_train[price_col].fillna(0).astype(np.float32)
    else:
        df_train[price_col] = np.float32(1.0)

    level_scores = {}

    for level_num, group_cols in HIERARCHY_LEVELS.items():
        level_name = LEVEL_NAMES[level_num]

        # Aggregate training actuals and prices
        train_agg = aggregate_to_level(df_train, group_cols, actual_col, day_col)
        train_mat, keys = pivot_to_matrix(train_agg, group_cols, actual_col, day_col)

        # Aggregate training prices for weights
        train_price_agg = aggregate_to_level(df_train, group_cols, price_col, day_col)
        train_price_mat, _ = pivot_to_matrix(
            train_price_agg, group_cols, price_col, day_col
        )

        # Aggregate validation actuals and predictions
        val_actual_agg = aggregate_to_level(df_val, group_cols, actual_col, day_col)
        val_actual_mat, _ = pivot_to_matrix(
            val_actual_agg, group_cols, actual_col, day_col
        )

        val_pred_agg = aggregate_to_level(df_val, group_cols, pred_col, day_col)
        val_pred_mat, _ = pivot_to_matrix(
            val_pred_agg, group_cols, pred_col, day_col
        )

        # Compute scale from training data
        scale = _compute_scale(train_mat)

        # Compute weights from training data
        weights = _compute_weights(train_mat, train_price_mat)

        # Compute WRMSSE for this level
        wrmsse_l = compute_wrmsse_level(
            val_actual_mat, val_pred_mat, scale, weights
        )

        level_scores[f"wrmsse_level_{level_num}"] = wrmsse_l
        logger.info(
            "  Level %2d (%s): WRMSSE = %.6f  (%d series)",
            level_num, level_name, wrmsse_l, len(keys),
        )

    # Overall WRMSSE = average across 12 levels
    overall = np.mean(list(level_scores.values()))
    level_scores["wrmsse_overall"] = float(overall)
    logger.info("  OVERALL WRMSSE = %.6f", overall)

    return level_scores
