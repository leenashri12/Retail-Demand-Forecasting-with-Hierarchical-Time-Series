"""
Phase 1 - Data Audit & Optimization Script
===========================================
Loads every M5 CSV, applies reduce_mem_usage, and prints:
  1. Memory before / after downcasting per dataset.
  2. Shape, dtypes summary.
  3. Null-value report.
  4. Hierarchy-level verification.
  5. Join-key verification.

Run:  python scripts/audit_data.py
"""

import gc
import io
import os
import sys
import logging
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import reduce_mem_usage, DATA_DIR, HIERARCHY_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit")

SEPARATOR = "=" * 72


# ── Helpers ──────────────────────────────────────────────────────────────
def mem_mb(df: pd.DataFrame) -> float:
    """Return DataFrame memory in MB (deep introspection)."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def audit_nulls(name: str, df: pd.DataFrame) -> None:
    """Print null counts for columns that have any."""
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if null_cols.empty:
        print(f"  [{name}] No null values [OK]")
    else:
        print(f"  [{name}] Columns with nulls:")
        for col, cnt in null_cols.items():
            pct = 100 * cnt / len(df)
            print(f"      {col:30s}  {cnt:>10,}  ({pct:.2f}%)")


def audit_dtypes_summary(name: str, df: pd.DataFrame) -> None:
    """Print dtype distribution for a DataFrame."""
    dtype_counts = df.dtypes.value_counts()
    parts = [f"{dtype}: {cnt}" for dtype, cnt in dtype_counts.items()]
    print(f"  [{name}] dtypes → {', '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN AUDIT
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    results = {}  # name → (before_mb, after_mb, df)

    # ------------------------------------------------------------------
    # 1. Load each CSV raw, record memory, downcast, record again
    # ------------------------------------------------------------------
    files = {
        "calendar":            DATA_DIR / "calendar.csv",
        "sales_validation":    DATA_DIR / "sales_train_validation.csv",
        "sales_evaluation":    DATA_DIR / "sales_train_evaluation.csv",
        "sell_prices":         DATA_DIR / "sell_prices.csv",
        "sample_submission":   DATA_DIR / "sample_submission.csv",
    }

    print_section("1. LOAD & DOWNCAST — Memory Report")

    for name, fpath in files.items():
        log.info("Loading %s …", fpath.name)
        df = pd.read_csv(fpath)
        before = mem_mb(df)
        df = reduce_mem_usage(df, verbose=False)
        after = mem_mb(df)
        pct = 100 * (before - after) / before
        results[name] = (before, after, df)
        print(
            f"  {name:25s}  {df.shape[0]:>10,} rows x {df.shape[1]:>5} cols  "
            f"| {before:>9.2f} MB -> {after:>9.2f} MB  (down {pct:.1f}%)"
        )
        gc.collect()

    total_before = sum(v[0] for v in results.values())
    total_after = sum(v[1] for v in results.values())
    total_pct = 100 * (total_before - total_after) / total_before
    print(f"\n  {'TOTAL':25s}  {'':>10}       {'':>5}      "
          f"| {total_before:>9.2f} MB -> {total_after:>9.2f} MB  (down {total_pct:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Dtypes summary
    # ------------------------------------------------------------------
    print_section("2. DTYPE SUMMARY (after downcasting)")
    for name, (_, _, df) in results.items():
        audit_dtypes_summary(name, df)

    # ------------------------------------------------------------------
    # 3. Null-value report
    # ------------------------------------------------------------------
    print_section("3. NULL-VALUE REPORT")
    for name, (_, _, df) in results.items():
        audit_nulls(name, df)

    # ------------------------------------------------------------------
    # 4. Hierarchy-level verification
    # ------------------------------------------------------------------
    print_section("4. HIERARCHY VERIFICATION")

    sales_v = results["sales_validation"][2]

    # Check that all hierarchy columns exist
    missing = [c for c in HIERARCHY_COLS if c not in sales_v.columns]
    if missing:
        print(f"  [WARN] Missing hierarchy columns: {missing}")
    else:
        print(f"  All hierarchy columns present: {HIERARCHY_COLS} [OK]")

    # Unique counts per hierarchy level
    print("\n  Hierarchy cardinality:")
    print(f"    {'Level':20s}  {'Unique':>8}")
    print(f"    {'-'*20}  {'-'*8}")

    # National level (implicit — always 1)
    print(f"    {'National':20s}  {'1':>8}  (implicit)")

    for col in HIERARCHY_COLS:
        nuniq = sales_v[col].nunique()
        sample = ", ".join(
            str(v) for v in sorted(sales_v[col].unique())[:5]
        )
        extra = "…" if sales_v[col].nunique() > 5 else ""
        print(f"    {col:20s}  {nuniq:>8,}  (e.g. {sample}{extra})")

    # ------------------------------------------------------------------
    # 5. Join-key verification
    # ------------------------------------------------------------------
    print_section("5. JOIN-KEY VERIFICATION")

    cal = results["calendar"][2]
    prices = results["sell_prices"][2]

    # 5a. Calendar → sales via 'd' column
    day_cols_in_sales = [c for c in sales_v.columns if c.startswith("d_")]
    day_ints_in_sales = sorted(
        int(c.split("_")[1]) for c in day_cols_in_sales
    )
    cal_d_values = sorted(cal["d"].unique()) if "d" in cal.columns else []

    print(f"  [sales <-> calendar] via 'd' column:")
    print(f"    Sales day columns:   d_{day_ints_in_sales[0]} .. d_{day_ints_in_sales[-1]}  "
          f"({len(day_ints_in_sales)} days)")
    print(f"    Calendar 'd' values: {cal_d_values[0]} .. {cal_d_values[-1]}  "
          f"({len(cal_d_values)} rows)")

    # Check alignment
    cal_d_ints = sorted(int(str(v).split("_")[1]) for v in cal_d_values)
    overlap = set(day_ints_in_sales) & set(cal_d_ints)
    print(f"    Overlap:             {len(overlap)} days [OK]" if len(overlap) == len(day_ints_in_sales)
          else f"    [WARN] Mismatch: {len(day_ints_in_sales)} sale days vs {len(overlap)} in calendar")

    # 5b. Sales ↔ Prices via (store_id, item_id) + wm_yr_wk
    print(f"\n  [sales <-> sell_prices] via (store_id, item_id, wm_yr_wk):")
    sales_stores = set(sales_v["store_id"].unique())
    price_stores = set(prices["store_id"].unique())
    print(f"    store_id match:  {sales_stores == price_stores}  "
          f"(sales={len(sales_stores)}, prices={len(price_stores)})")

    sales_items = set(sales_v["item_id"].unique())
    price_items = set(prices["item_id"].unique())
    items_only_sales = sales_items - price_items
    items_only_prices = price_items - sales_items
    print(f"    item_id match:   {sales_items == price_items}  "
          f"(sales={len(sales_items):,}, prices={len(price_items):,})")
    if items_only_sales:
        print(f"      [WARN] {len(items_only_sales)} items in sales but NOT in prices")
    if items_only_prices:
        print(f"      [WARN] {len(items_only_prices)} items in prices but NOT in sales")

    # wm_yr_wk
    if "wm_yr_wk" in cal.columns:
        cal_weeks = set(cal["wm_yr_wk"].unique())
        price_weeks = set(prices["wm_yr_wk"].unique())
        overlap_w = cal_weeks & price_weeks
        print(f"    wm_yr_wk overlap: {len(overlap_w)} weeks "
              f"(cal={len(cal_weeks)}, prices={len(price_weeks)})")

    # ------------------------------------------------------------------
    # 6. 500-day sliding window preview
    # ------------------------------------------------------------------
    print_section("6. SLIDING WINDOW PREVIEW (d_1414 -> d_1913)")

    window_cols = [f"d_{i}" for i in range(1414, 1914)]
    present = [c for c in window_cols if c in sales_v.columns]
    print(f"  Window columns in sales_validation: {len(present)} / 500 expected")
    if len(present) == 500:
        print("  All 500 sliding-window day columns present [OK]")
    else:
        print(f"  [WARN] Only {len(present)} of 500 columns found")

    # Estimate long-format size for the window
    n_series = len(sales_v)
    n_days = len(present)
    # In long format: n_series × n_days rows, each with ~10 cols
    est_rows = n_series * n_days
    # Rough estimate: each row ≈ 50 bytes after downcasting
    est_mb = est_rows * 50 / (1024 ** 2)
    print(f"\n  Estimated long-format size (500 days):")
    print(f"    Rows:   {n_series:,} series × {n_days} days = {est_rows:,}")
    print(f"    Memory: ~{est_mb:,.0f} MB (rough, before calendar/price merge)")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    del results, sales_v, cal, prices
    gc.collect()
    print(f"\n{SEPARATOR}")
    print("  AUDIT COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
