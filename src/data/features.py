"""
Feature engineering pipeline for M5 Forecasting.

Builds features on the long-format 500-day DataFrame:
  - Sales lags (7, 14, 28 days)
  - Rolling statistics (7, 28, 60-day mean & std)
  - Calendar encodings (day-of-week, month, holidays, SNAP)
  - Price features (price momentum = pct change from prior week)

All operations are grouped by (store_id, item_id) to prevent
data leakage across series. Memory-First approach per AGENTS.md S5.
"""

import gc
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loader import reduce_mem_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAG_DAYS: List[int] = [7, 14, 28]
ROLLING_WINDOWS: List[int] = [7, 28, 60]
GROUP_COLS: List[str] = ["store_id", "item_id"]

# Day-of-week encoding (sine/cosine for cyclical nature)
# wday in M5: 1=Saturday … 7=Friday
DOW_PERIOD: int = 7
MONTH_PERIOD: int = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_day_num(d_col: pd.Series) -> pd.Series:
    """Convert 'd_XXXX' strings to integer day numbers.

    Args:
        d_col: Series of strings like 'd_1414', 'd_1415', ...

    Returns:
        Integer Series of day numbers.
    """
    return d_col.astype(str).str.split("_").str[1].astype(np.int16)


def _cyclical_encode(series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    """Encode a periodic integer feature as sin/cos pair.

    Args:
        series: Integer series (e.g. day-of-week 1-7, month 1-12).
        period: The period length.

    Returns:
        Tuple of (sin_encoded, cos_encoded) as float32 Series.
    """
    angle = 2 * np.pi * series / period
    return (
        np.sin(angle).astype(np.float32),
        np.cos(angle).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Feature Builders
# ---------------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add sales lag features grouped by (store_id, item_id).

    Creates columns: sales_lag_7, sales_lag_14, sales_lag_28.
    Grouped shift prevents leakage across different items/stores.

    Args:
        df: Long-format DataFrame with 'sales' column, sorted by
            (store_id, item_id, day_num).
        lags: List of lag periods. Defaults to [7, 14, 28].

    Returns:
        DataFrame with new lag columns appended.
    """
    lags = lags or LAG_DAYS
    logger.info("Adding %d lag features: %s ...", len(lags), lags)

    grouped = df.groupby(GROUP_COLS, observed=True)["sales"]
    for lag in lags:
        col_name = f"sales_lag_{lag}"
        df[col_name] = grouped.shift(lag).astype(np.float32)
        logger.debug("  Created %s", col_name)

    del grouped
    gc.collect()
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add rolling mean & std features grouped by (store_id, item_id).

    For each window W, creates:
      - sales_rmean_{W}  (rolling mean over past W days)
      - sales_rstd_{W}   (rolling std over past W days)

    Uses shift(1) inside rolling to avoid including the current day.

    Args:
        df: Long-format DataFrame sorted by (store_id, item_id, day_num).
        windows: List of window sizes. Defaults to [7, 28, 60].

    Returns:
        DataFrame with new rolling columns appended.
    """
    windows = windows or ROLLING_WINDOWS
    logger.info("Adding rolling features for windows: %s ...", windows)

    grouped = df.groupby(GROUP_COLS, observed=True)["sales"]

    for w in windows:
        # shift(1) so the rolling window only uses past data
        rolling = grouped.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"sales_rmean_{w}"] = rolling.astype(np.float32)

        rolling_std = grouped.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std()
        )
        df[f"sales_rstd_{w}"] = rolling_std.astype(np.float32)

        logger.debug("  Created sales_rmean_%d, sales_rstd_%d", w, w)

        del rolling, rolling_std
        gc.collect()

    del grouped
    gc.collect()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode calendar features: cyclical day-of-week/month, event flags, SNAP.

    Creates:
      - day_num: integer day number extracted from 'd' column
      - wday_sin, wday_cos: cyclical encoding of wday
      - month_sin, month_cos: cyclical encoding of month
      - is_event: binary flag (1 if any event on that day)
      - snap: unified SNAP flag per row's own state

    Args:
        df: Long-format DataFrame with calendar columns already merged.

    Returns:
        DataFrame with new calendar feature columns.
    """
    logger.info("Adding calendar features ...")

    # Day number
    if "day_num" not in df.columns:
        df["day_num"] = _extract_day_num(df["d"])

    # Cyclical day-of-week
    df["wday_sin"], df["wday_cos"] = _cyclical_encode(df["wday"], DOW_PERIOD)

    # Cyclical month
    df["month_sin"], df["month_cos"] = _cyclical_encode(df["month"], MONTH_PERIOD)

    # Binary event flag: 1 if event_name_1 is not "NoEvent"
    if "event_name_1" in df.columns:
        event_col = df["event_name_1"].astype(str)
        df["is_event"] = (event_col != "NoEvent").astype(np.int8)
        del event_col
    else:
        df["is_event"] = np.int8(0)

    # Unified SNAP flag: pick the SNAP column matching the row's state
    if all(c in df.columns for c in ["snap_CA", "snap_TX", "snap_WI"]):
        state = df["state_id"].astype(str)
        df["snap"] = np.int8(0)
        df.loc[state == "CA", "snap"] = df.loc[state == "CA", "snap_CA"]
        df.loc[state == "TX", "snap"] = df.loc[state == "TX", "snap_TX"]
        df.loc[state == "WI", "snap"] = df.loc[state == "WI", "snap_WI"]
        df["snap"] = df["snap"].astype(np.int8)
        del state
    else:
        df["snap"] = np.int8(0)

    gc.collect()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-derived features.

    Creates:
      - price_momentum: pct change in sell_price from the prior week
        for each (store_id, item_id), grouped to avoid leakage.

    Args:
        df: Long-format DataFrame with 'sell_price' and 'wm_yr_wk'.

    Returns:
        DataFrame with price feature columns appended.
    """
    logger.info("Adding price features ...")

    if "sell_price" not in df.columns:
        logger.warning("sell_price not found; skipping price features.")
        return df

    # Cast to float32 for safe arithmetic (float16 breaks pct_change internals)
    price = df["sell_price"].astype(np.float32)

    # Price momentum: manual pct_change per (store, item) using shift
    grouped = df.groupby(GROUP_COLS, observed=True)
    prev_price = grouped["sell_price"].shift(1).astype(np.float32)

    # pct_change = (current - previous) / previous
    df["price_momentum"] = np.where(
        prev_price != 0,
        (price - prev_price) / prev_price,
        0.0,
    ).astype(np.float32)

    # Fill NaN (first row per group) with 0
    df["price_momentum"] = df["price_momentum"].fillna(0.0).astype(np.float32)

    del price, prev_price, grouped
    gc.collect()
    return df


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
      1. Sort by (store_id, item_id, day_num) for correct lag/rolling.
      2. Add lag features.
      3. Add rolling statistics.
      4. Add calendar features.
      5. Add price features.
      6. Drop rows where lags are NaN (burn-in period).
      7. Final downcast via reduce_mem_usage.

    Args:
        df: Long-format DataFrame from Phase 1 merge.
        lags: Override lag periods.
        windows: Override rolling window sizes.

    Returns:
        Feature-enriched DataFrame, downcasted, with burn-in rows dropped.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE START")
    logger.info("=" * 60)

    mem_before = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info("Input shape: %s  (%.1f MB)", df.shape, mem_before)

    # 0. Extract day_num and sort for correct temporal ordering
    logger.info("Step 0: Sorting by (store_id, item_id, day_num) ...")
    df["day_num"] = _extract_day_num(df["d"])
    df = df.sort_values(GROUP_COLS + ["day_num"]).reset_index(drop=True)
    gc.collect()

    # 1. Lag features
    df = add_lag_features(df, lags)
    gc.collect()

    # 2. Rolling features
    df = add_rolling_features(df, windows)
    gc.collect()

    # 3. Calendar features
    df = add_calendar_features(df)
    gc.collect()

    # 4. Price features
    df = add_price_features(df)
    gc.collect()

    # 5. Drop burn-in rows (where largest lag = 28 creates NaNs)
    max_lag = max(lags or LAG_DAYS)
    burn_in_days = max_lag
    logger.info(
        "Dropping burn-in rows (first %d days per series) ...", burn_in_days
    )
    n_before = len(df)

    # Keep rows where the largest lag is not null
    largest_lag_col = f"sales_lag_{max_lag}"
    df = df.dropna(subset=[largest_lag_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    logger.info("  Dropped %s rows (%.1f%%)", f"{n_dropped:,}", 100 * n_dropped / n_before)

    # 6. Final downcast
    logger.info("Final reduce_mem_usage ...")
    df = reduce_mem_usage(df, verbose=True)

    mem_after = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info("=" * 60)
    logger.info(
        "PIPELINE COMPLETE: %s  (%.1f MB -> %.1f MB)",
        df.shape, mem_before, mem_after,
    )
    logger.info("=" * 60)

    return df
