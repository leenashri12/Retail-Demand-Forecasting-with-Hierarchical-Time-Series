"""
Phase 2 - Feature Engineering Runner
=====================================
Loads the Phase 1 long-format parquet, runs the full feature pipeline,
saves the result to data/processed/featured_data.parquet, and prints
the final memory footprint + full feature list.

Run:  python scripts/run_features.py
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

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import reduce_mem_usage, DATA_DIR
from src.data.features import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("features_runner")

PROCESSED_DIR = DATA_DIR / "processed"
INPUT_PATH = PROCESSED_DIR / "sales_long_500d.parquet"
OUTPUT_PATH = PROCESSED_DIR / "featured_data.parquet"

SEPARATOR = "=" * 72


def current_process_mem_mb() -> float:
    """Return current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        return -1.0


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load Phase 1 output
    # ------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("  LOADING Phase 1 Parquet")
    print(SEPARATOR)

    log.info("Loading %s ...", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    input_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    log.info("Loaded: %s  (%.1f MB)", df.shape, input_mem)
    print(f"  Shape:  {df.shape}")
    print(f"  Memory: {input_mem:.1f} MB")
    print(f"  RSS:    {current_process_mem_mb():.0f} MB")

    # ------------------------------------------------------------------
    # 2. Run feature pipeline
    # ------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("  RUNNING Feature Engineering Pipeline")
    print(SEPARATOR)

    df = build_features(df)
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Save to parquet
    # ------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("  SAVING Featured Data")
    print(SEPARATOR)

    log.info("Saving to %s ...", OUTPUT_PATH)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)
    log.info("Parquet file size: %.1f MB", file_size_mb)

    # ------------------------------------------------------------------
    # 4. Final summary
    # ------------------------------------------------------------------
    final_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    elapsed = time.time() - t0

    print(f"\n{SEPARATOR}")
    print("  PHASE 2 RESULTS SUMMARY")
    print(SEPARATOR)

    print(f"\n  Final shape:       {df.shape}")
    print(f"  Final memory:      {final_mem:.1f} MB")
    print(f"  Parquet file size: {file_size_mb:.1f} MB")
    print(f"  Peak RSS:          {current_process_mem_mb():.0f} MB")
    print(f"  Time elapsed:      {elapsed:.1f}s")
    print(f"  Output:            {OUTPUT_PATH}")

    # Full feature list with dtypes
    print(f"\n  FULL FEATURE LIST ({len(df.columns)} columns):")
    print(f"  {'#':>3}  {'Column':30s}  {'Dtype':15s}  {'Nulls':>10}  {'Unique':>10}")
    print(f"  {'---':>3}  {'-'*30}  {'-'*15}  {'-'*10}  {'-'*10}")
    for i, col in enumerate(df.columns, 1):
        nulls = df[col].isnull().sum()
        nuniq = df[col].nunique()
        print(f"  {i:>3}  {col:30s}  {str(df[col].dtype):15s}  {nulls:>10,}  {nuniq:>10,}")

    # Sample rows
    print(f"\n  Sample (first 3 rows):")
    print(df.head(3).to_string(index=False))

    # Cleanup
    del df
    gc.collect()

    print(f"\n{SEPARATOR}")
    print("  PHASE 2 COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
