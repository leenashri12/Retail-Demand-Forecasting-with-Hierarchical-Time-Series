# AGENTS.md — Project Charter & Engineering Guidelines

> **Project Title:** Retail Demand Forecasting with Hierarchical Time Series
> **Last Updated:** 2026-05-10
> **Lead ML Engineer:** AI Assistant (Antigravity)

---

## 1. Project Goal

Forecast sales across the **M5 Walmart hierarchy** ensuring coherence across all aggregation levels:

```
National → State (3) → Store (10) → Department (7) → Category (3) → SKU (~30,490)
```

This mirrors real-world retail planning workflows where forecasts at every level of the hierarchy must **add up** (i.e., the sum of store-level forecasts must equal the regional forecast, and so on).

---

## 2. Coherence Strategy

We will perform a **comparative study** between three reconciliation methods:

| Method | Description |
|---|---|
| **Bottom-Up (BU)** | Forecast at SKU level, aggregate upward. Simple but ignores top-level signal. |
| **Top-Down (TD)** | Forecast at national level, disaggregate downward via historical proportions. |
| **MinTrace (MinT)** | Optimal combination — minimizes trace of forecast error covariance matrix. |

**Hypothesis:** MinTrace will provide the best mathematical coherence and forecast accuracy as measured by WRMSSE.

---

## 3. Primary Metric

**WRMSSE — Weighted Root Mean Scaled Squared Error**

This is the official M5 competition metric. It:
- Scales errors by each series' historical variability (avoids penalizing intermittent demand series).
- Weights series by their dollar sales contribution (high-revenue SKUs matter more).
- Aggregates across all 12 hierarchical levels defined in the M5 competition.

---

## 4. Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core development |
| **ML Model** | LightGBM | Gradient boosting for tabular time series features |
| **Hierarchical Reconciliation** | `sktime` / `hierarchicalforecast` | BU, TD, MinTrace reconciliation |
| **Experiment Tracking** | MLflow | Log metrics, parameters, artifacts per run |
| **Data Versioning** | DVC | Track raw/processed data versions |
| **Serving** | FastAPI | REST API for inference |
| **Containerization** | Docker | Reproducible deployment |

---

## 5. Memory Constraint — "Memory-First" Approach

> **Target Environment:** Local laptop (limited RAM, no GPU required for LightGBM).

### Mandatory Optimizations

1. **Aggressive Type Downcasting**
   - All integer columns → `int8` or `int16` (sales counts, flags, etc.)
   - All float columns → `float16` or `float32` (prices, weights)
   - Categorical columns → `category` dtype

2. **Sliding Window Training Set**
   - Use only the **last 500 days** (`d_1414` → `d_1913`) for the final merged training set.
   - Earlier history is used **only** for feature engineering (lags, rolling stats) and then dropped.

3. **Chunked Processing**
   - Large merges (e.g., `sales × sell_prices`) must be done in chunks, never loading full cross-join into memory.
   - Intermediate DataFrames must be explicitly deleted with `del` + `gc.collect()`.

4. **Memory Profiling**
   - Every notebook/script must log peak memory usage.
   - Target: Full pipeline runnable in **≤ 8 GB RAM**.

---

## 6. Data Manifest

All data resides in `./data/` and comes from the [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) Kaggle competition.

| File | Size | Description |
|---|---|---|
| `calendar.csv` | ~101 KB | Date metadata: events, SNAP flags, weekday/month |
| `sales_train_validation.csv` | ~114 MB | Daily unit sales for 30,490 SKUs × 1,913 days (d_1 → d_1913) |
| `sales_train_evaluation.csv` | ~116 MB | Extended to d_1941 (includes 28-day evaluation period) |
| `sell_prices.csv` | ~194 MB | Weekly selling price per store-SKU pair |
| `sample_submission.csv` | ~5 MB | Submission template (28-day forecast horizon) |

---

## 7. Project Structure (Target)

```
DataScience_Project/
├── AGENTS.md                  # This file — project charter
├── README.md                  # Public-facing project documentation
├── pyproject.toml             # Dependencies & project metadata
├── requirements.txt           # Pinned dependencies
├── .dvc/                      # DVC configuration
├── .dvcignore
├── data/
│   ├── raw/                   # Original M5 CSVs (DVC tracked)
│   └── processed/             # Feature-engineered parquet files
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_eng.ipynb   # Feature engineering walkthrough
│   └── 03_modeling.ipynb      # Training & evaluation
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Memory-optimized data loading
│   │   ├── features.py        # Feature engineering pipeline
│   │   └── validation.py      # Data quality checks
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lgbm_forecaster.py # LightGBM training & prediction
│   │   └── reconciliation.py  # BU / TD / MinTrace wrappers
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── wrmsse.py          # WRMSSE metric implementation
│   └── api/
│       ├── __init__.py
│       └── main.py            # FastAPI inference endpoint
├── configs/
│   └── default.yaml           # Hyperparameters & pipeline config
├── tests/
│   ├── test_loader.py
│   ├── test_features.py
│   └── test_wrmsse.py
├── mlruns/                    # MLflow experiment tracking
├── Dockerfile
├── docker-compose.yml
└── dvc.yaml                   # DVC pipeline definition
```

---

## 8. Development Workflow

1. **Phase 1 — Data Audit & EDA:** Validate data integrity, profile memory, explore distributions.
2. **Phase 2 — Feature Engineering:** Calendar features, lag/rolling features, price features, hierarchy encodings.
3. **Phase 3 — Baseline Model:** Train LightGBM at SKU level, evaluate with WRMSSE.
4. **Phase 4 — Hierarchical Reconciliation:** Apply BU, TD, MinTrace; compare coherence & accuracy.
5. **Phase 5 — MLflow Tracking:** Log all experiments, register best model.
6. **Phase 6 — API & Docker:** Serve forecasts via FastAPI, containerize with Docker.
7. **Phase 7 — Documentation & Presentation:** README, walkthrough, results summary.

---

## 9. Coding Standards

- **Docstrings:** Google-style on all public functions.
- **Type Hints:** Required on all function signatures.
- **Logging:** Use `logging` module (not `print`). Level: `INFO` for pipelines, `DEBUG` for diagnostics.
- **Memory:** Every function that loads or transforms data must include memory-usage logging.
- **Reproducibility:** All random seeds set to `42`. All experiments tracked in MLflow.

---

## 10. Agent Instructions

When working on this project, the AI agent (Lead ML Engineer) must:

1. **Always check memory impact** before loading or merging large DataFrames.
2. **Downcast dtypes immediately** after any read or merge operation.
3. **Delete intermediate DataFrames** and call `gc.collect()` after each pipeline step.
4. **Log WRMSSE** as the primary metric for every model run.
5. **Track all experiments** in MLflow with descriptive run names.
6. **Validate hierarchy coherence** after every reconciliation step.
7. **Follow the Memory-First approach** — never trade memory for convenience.
8. **Use parquet format** for all intermediate/processed data (never pickle for DataFrames).
