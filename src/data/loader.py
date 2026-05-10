"""
Memory-optimized data loading utilities for M5 Forecasting.

Implements aggressive dtype downcasting per AGENTS.md Section 5
(Memory-First approach). Every load function logs memory usage
before and after optimization.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# M5 hierarchy columns (top → bottom)
HIERARCHY_COLS = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]

# Join keys across datasets
JOIN_KEYS = {
    "sales_to_calendar": "d",          # day column (d_1 … d_1941)
    "sales_to_prices": ["store_id", "item_id", "wm_yr_wk"],
}

# Sliding-window boundaries (last 500 days of sales_train_validation)
WINDOW_START = 1414   # d_1414
WINDOW_END = 1913     # d_1913
WINDOW_DAYS = list(range(WINDOW_START, WINDOW_END + 1))
WINDOW_D_COLS = [f"d_{i}" for i in WINDOW_DAYS]


# ---------------------------------------------------------------------------
# Core: reduce_mem_usage
# ---------------------------------------------------------------------------
def reduce_mem_usage(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Downcast every column in *df* to the smallest viable dtype.

    Rules (per AGENTS.md §5):
      - Integer columns → int8 / int16 / int32 (fit to actual min/max).
      - Float columns   → float16 / float32    (fit to actual min/max).
      - Object columns  → category dtype.

    Args:
        df: Input DataFrame (modified **in-place** and also returned).
        verbose: If True, log per-column dtype changes and total savings.

    Returns:
        The same DataFrame with optimized dtypes.
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min, c_max = df[col].min(), df[col].max()

            if np.issubdtype(col_type, np.integer):
                # Try progressively larger int types
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                # else keep int64

            elif np.issubdtype(col_type, np.floating):
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                # else keep float64

        elif col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            # Convert to category if cardinality < 50% of rows
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction = 100 * (start_mem - end_mem) / start_mem

    if verbose:
        logger.info(
            "Memory: %.2f MB → %.2f MB (↓ %.1f%%)",
            start_mem,
            end_mem,
            reduction,
        )

    return df


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------
def load_calendar(path: Optional[Path] = None) -> pd.DataFrame:
    """Load calendar.csv with optimized dtypes.

    Args:
        path: Override path; defaults to DATA_DIR / 'calendar.csv'.

    Returns:
        Optimized calendar DataFrame.
    """
    path = path or DATA_DIR / "calendar.csv"
    logger.info("Loading %s …", path.name)
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)
    return df


def load_sales(
    variant: str = "validation",
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load sales_train_validation.csv or sales_train_evaluation.csv.

    Args:
        variant: 'validation' or 'evaluation'.
        path: Override path.

    Returns:
        Optimized sales DataFrame.
    """
    if path is None:
        fname = f"sales_train_{variant}.csv"
        path = DATA_DIR / fname
    logger.info("Loading %s …", path.name)
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)
    return df


def load_sell_prices(path: Optional[Path] = None) -> pd.DataFrame:
    """Load sell_prices.csv with optimized dtypes.

    Args:
        path: Override path.

    Returns:
        Optimized sell_prices DataFrame.
    """
    path = path or DATA_DIR / "sell_prices.csv"
    logger.info("Loading %s …", path.name)
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)
    return df


def load_sample_submission(path: Optional[Path] = None) -> pd.DataFrame:
    """Load sample_submission.csv with optimized dtypes.

    Args:
        path: Override path.

    Returns:
        Optimized sample_submission DataFrame.
    """
    path = path or DATA_DIR / "sample_submission.csv"
    logger.info("Loading %s …", path.name)
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)
    return df


# ---------------------------------------------------------------------------
# Convenience: load all
# ---------------------------------------------------------------------------
def load_all_data(
    data_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load and optimise every M5 CSV into a dict of DataFrames.

    Args:
        data_dir: Root data directory; defaults to DATA_DIR.

    Returns:
        Dict with keys: 'calendar', 'sales_validation', 'sales_evaluation',
        'sell_prices', 'sample_submission'.
    """
    d = data_dir or DATA_DIR
    datasets = {
        "calendar": load_calendar(d / "calendar.csv"),
        "sales_validation": load_sales("validation", d / "sales_train_validation.csv"),
        "sales_evaluation": load_sales("evaluation", d / "sales_train_evaluation.csv"),
        "sell_prices": load_sell_prices(d / "sell_prices.csv"),
        "sample_submission": load_sample_submission(d / "sample_submission.csv"),
    }
    gc.collect()
    return datasets
