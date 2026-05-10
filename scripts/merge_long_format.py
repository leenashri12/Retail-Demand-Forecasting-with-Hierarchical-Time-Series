"""
Phase 1 - Merge to Long Format (500-day sliding window)
========================================================
Implements the 4-step merge strategy from the Phase 1 audit report:
  Step 1: Melt sales (wide -> long) for d_1414..d_1913 only
  Step 2: Merge calendar metadata
  Step 3: Merge sell_prices (chunked by store_id)
  Step 4: Final downcast + save to Parquet

Target: Peak memory <= 3-4 GB.

Run:  python scripts/merge_long_format.py
"""

import gc
import io
import os
import sys
import time
import logging
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    reduce_mem_usage,
    DATA_DIR,
    HIERARCHY_COLS,
    WINDOW_START,
    WINDOW_END,
    WINDOW_D_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge")

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED_DIR / "sales_long_500d.parquet"

SEPARATOR = "=" * 72


def mem_mb(df: pd.DataFrame) -> float:
    """Return DataFrame memory in MB (deep introspection)."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def current_process_mem_mb() -> float:
    """Return current process RSS in MB (Windows-compatible)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        return -1.0


def main() -> None:
    t0 = time.time()

    # ==================================================================
    # STEP 1: Melt sales (wide -> long) -- ONLY 500-day window
    # ==================================================================
    print(f"\n{SEPARATOR}")
    print("  STEP 1: Melt sales (wide -> long) -- 500-day window")
    print(SEPARATOR)

    log.info("Loading sales_train_validation.csv ...")
    sales_wide = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
    log.info("Raw sales shape: %s  (%.1f MB)", sales_wide.shape, mem_mb(sales_wide))

    # Keep only hierarchy columns + the 500-day window columns
    id_cols = ["id"] + HIERARCHY_COLS
    keep_cols = id_cols + WINDOW_D_COLS
    missing_d = [c for c in WINDOW_D_COLS if c not in sales_wide.columns]
    if missing_d:
        log.error("Missing day columns: %s", missing_d[:5])
        sys.exit(1)

    sales_wide = sales_wide[keep_cols]
    log.info("After slicing to 500 days: %s  (%.1f MB)", sales_wide.shape, mem_mb(sales_wide))

    # Downcast before melt to minimize memory during melt
    sales_wide = reduce_mem_usage(sales_wide, verbose=False)
    log.info("After downcasting wide: %.1f MB", mem_mb(sales_wide))

    # Melt: wide -> long
    log.info("Melting wide -> long ...")
    sales_long = pd.melt(
        sales_wide,
        id_vars=id_cols,
        value_vars=WINDOW_D_COLS,
        var_name="d",
        value_name="sales",
    )

    # Free wide DataFrame immediately
    del sales_wide
    gc.collect()

    log.info("Long shape: %s  (%.1f MB)", sales_long.shape, mem_mb(sales_long))

    # Downcast the melted result
    sales_long["sales"] = sales_long["sales"].astype(np.int16)
    log.info("After downcast sales col: %.1f MB", mem_mb(sales_long))

    print(f"  Rows: {len(sales_long):,}")
    print(f"  Cols: {list(sales_long.columns)}")
    print(f"  Mem:  {mem_mb(sales_long):.1f} MB")
    print(f"  Process RSS: {current_process_mem_mb():.0f} MB")

    # ==================================================================
    # STEP 2: Merge calendar metadata
    # ==================================================================
    print(f"\n{SEPARATOR}")
    print("  STEP 2: Merge calendar metadata")
    print(SEPARATOR)

    log.info("Loading calendar.csv ...")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    calendar = reduce_mem_usage(calendar, verbose=False)
    log.info("Calendar shape: %s  (%.1f MB)", calendar.shape, mem_mb(calendar))

    # Fill event nulls with "NoEvent" before merge
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in calendar.columns:
            # Must add "NoEvent" as a valid category before fillna
            if hasattr(calendar[col], "cat"):
                calendar[col] = calendar[col].cat.add_categories("NoEvent")
            calendar[col] = calendar[col].fillna("NoEvent")
            calendar[col] = calendar[col].astype("category")

    # Merge on 'd' column
    log.info("Merging sales_long with calendar on 'd' ...")
    sales_long = sales_long.merge(calendar, on="d", how="left")

    # Free calendar
    del calendar
    gc.collect()

    # Downcast after merge
    sales_long = reduce_mem_usage(sales_long, verbose=False)

    log.info("After calendar merge: %s  (%.1f MB)", sales_long.shape, mem_mb(sales_long))
    print(f"  Shape: {sales_long.shape}")
    print(f"  Mem:   {mem_mb(sales_long):.1f} MB")
    print(f"  New cols added: wm_yr_wk, weekday, wday, month, year, event_*, snap_*")
    print(f"  Process RSS: {current_process_mem_mb():.0f} MB")

    # ==================================================================
    # STEP 3: Merge sell_prices -- CHUNKED by store_id
    # ==================================================================
    print(f"\n{SEPARATOR}")
    print("  STEP 3: Merge sell_prices (chunked by store_id)")
    print(SEPARATOR)

    log.info("Loading sell_prices.csv ...")
    sell_prices = pd.read_csv(DATA_DIR / "sell_prices.csv")
    sell_prices = reduce_mem_usage(sell_prices, verbose=False)
    log.info("sell_prices shape: %s  (%.1f MB)", sell_prices.shape, mem_mb(sell_prices))

    # Get unique stores
    stores = sorted(sales_long["store_id"].unique())
    log.info("Merging across %d stores ...", len(stores))

    chunks = []
    for i, store in enumerate(stores):
        # Filter both sides to this store
        mask_long = sales_long["store_id"] == store
        mask_price = sell_prices["store_id"] == store

        chunk_long = sales_long.loc[mask_long].copy()
        chunk_price = sell_prices.loc[mask_price, ["store_id", "item_id", "wm_yr_wk", "sell_price"]]

        # Merge on (store_id, item_id, wm_yr_wk)
        merged = chunk_long.merge(
            chunk_price,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )
        chunks.append(merged)

        del chunk_long, chunk_price, merged
        gc.collect()

        log.info(
            "  [%2d/%d] %s -- %s rows merged",
            i + 1, len(stores), store, f"{len(chunks[-1]):,}",
        )

    # Free sell_prices and original sales_long
    del sell_prices, sales_long
    gc.collect()

    # Concatenate all chunks
    log.info("Concatenating %d store chunks ...", len(chunks))
    sales_long = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    log.info("After price merge: %s  (%.1f MB)", sales_long.shape, mem_mb(sales_long))
    print(f"  Shape: {sales_long.shape}")
    print(f"  Mem:   {mem_mb(sales_long):.1f} MB")
    print(f"  sell_price nulls: {sales_long['sell_price'].isnull().sum():,}")
    print(f"  Process RSS: {current_process_mem_mb():.0f} MB")

    # ==================================================================
    # STEP 4: Final downcast + save to Parquet
    # ==================================================================
    print(f"\n{SEPARATOR}")
    print("  STEP 4: Final downcast + save to Parquet")
    print(SEPARATOR)

    # Final downcast
    sales_long = reduce_mem_usage(sales_long, verbose=False)
    final_mem = mem_mb(sales_long)
    log.info("Final DataFrame: %s  (%.1f MB)", sales_long.shape, final_mem)

    # Save to parquet
    log.info("Saving to %s ...", OUTPUT_PATH)
    sales_long.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)
    log.info("Parquet file size: %.1f MB", file_size_mb)

    # Summary
    elapsed = time.time() - t0
    print(f"\n  Final shape:    {sales_long.shape}")
    print(f"  Final memory:   {final_mem:.1f} MB")
    print(f"  Parquet size:   {file_size_mb:.1f} MB")
    print(f"  Peak RSS:       {current_process_mem_mb():.0f} MB")
    print(f"  Time elapsed:   {elapsed:.1f}s")
    print(f"  Output:         {OUTPUT_PATH}")

    # Quick data preview
    print(f"\n  Column dtypes:")
    for col in sales_long.columns:
        print(f"    {col:25s}  {str(sales_long[col].dtype):15s}")

    print(f"\n  Head (5 rows):")
    print(sales_long.head().to_string(index=False))

    # Cleanup
    del sales_long
    gc.collect()

    print(f"\n{SEPARATOR}")
    print("  MERGE COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
