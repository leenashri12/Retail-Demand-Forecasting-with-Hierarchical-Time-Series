<p align="center">
  <h1 align="center">📊 M5 Retail Demand Forecasting</h1>
  <p align="center">
    <strong>Hierarchical Time Series Forecasting with LightGBM &amp; Reconciliation</strong>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#results">Results</a> •
    <a href="#api-reference">API</a> •
    <a href="#docker">Docker</a>
  </p>
</p>

---

## Overview

An end-to-end **retail demand forecasting system** built on the [M5 Walmart dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy), producing coherent 28-day sales forecasts across **30,490 SKUs** and **12 aggregation levels** of the Walmart product hierarchy:

```
National (1) → State (3) → Store (10) → Department (7) → Category (3) → SKU (30,490)
```

The project implements a **comparative study** of three hierarchical reconciliation methods — **Bottom-Up**, **Top-Down**, and **MinTrace (OLS)** — evaluated using the official M5 competition metric, **WRMSSE** (Weighted Root Mean Scaled Squared Error).

### Key Highlights

| Metric | Value |
|---|---|
| **Best WRMSSE** | 0.6145 (Bottom-Up) |
| **Lowest Coherence Error** | 0.0188 (MinTrace) |
| **Total Features** | 39 (17 engineered) |
| **Peak RAM** | ~1 GB (Memory-First design) |
| **Model** | LightGBM (500 trees, Tweedie) |

---

## Quickstart

### Prerequisites

- Python 3.10+
- ~2 GB free RAM
- M5 dataset files in `data/` ([download from Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data))

### Installation

```bash
git clone <repo-url>
cd DataScience_Project

# Install dependencies
pip install -r requirements.txt

# Initialize DVC and pull data
dvc pull
```

### Run the Full Pipeline

```bash
# Phase 1: Data audit & long-format merge (500-day window)
python scripts/audit_data.py
python scripts/merge_long_format.py

# Phase 2: Feature engineering (lags, rolling stats, calendar, price)
python scripts/run_features.py

# Phase 3: Train LightGBM + evaluate BU/TD/MinTrace reconciliation
python scripts/run_phase3.py
```

### Launch the Application

```bash
# Terminal 1: FastAPI backend
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Streamlit dashboard
streamlit run src/api/dashboard.py --server.port 8501
```

Then open:
- **API Docs:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501

---

## Architecture

### Project Structure

```
DataScience_Project/
├── AGENTS.md                          # Project charter & engineering guidelines
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Multi-stage (API + Dashboard)
├── docker-compose.yml                 # Orchestrates both services
│
├── data/
│   ├── calendar.csv                   # M5 date metadata
│   ├── sales_train_validation.csv     # 30,490 SKUs × 1,913 days
│   ├── sales_train_evaluation.csv     # Extended to d_1941
│   ├── sell_prices.csv                # Weekly prices per store-SKU
│   ├── sample_submission.csv          # Submission template
│   └── processed/                     # Pipeline outputs (DVC-tracked)
│       ├── sales_long_500d.parquet    # Phase 1: Merged long format
│       ├── featured_data.parquet      # Phase 2: 39-feature dataset
│       ├── lgbm_model.txt             # Phase 3: Trained LightGBM
│       └── lgbm_preds.npy             # Phase 3: Validation predictions
│
├── scripts/
│   ├── audit_data.py                  # Data integrity & memory audit
│   ├── merge_long_format.py           # Wide → long format conversion
│   ├── run_features.py                # Feature engineering runner
│   └── run_phase3.py                  # Training & reconciliation runner
│
├── src/
│   ├── data/
│   │   ├── loader.py                  # Memory-optimized loaders & reduce_mem_usage
│   │   └── features.py                # Feature engineering pipeline
│   ├── models/
│   │   └── reconciliation.py          # BU, TD, MinTrace reconciliation
│   ├── evaluation/
│   │   └── wrmsse.py                  # Official M5 WRMSSE metric
│   └── api/
│       ├── __init__.py                # FastAPI application
│       ├── main.py                    # API entry point
│       └── dashboard.py               # Streamlit dashboard
│
└── mlruns/                            # MLflow experiment tracking
```

### Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| ML Model | **LightGBM** | Gradient boosting for tabular time series |
| Reconciliation | **Custom (NumPy)** | Bottom-Up, Top-Down, MinTrace OLS |
| Experiment Tracking | **MLflow** | Metrics, parameters, artifact logging |
| Data Versioning | **DVC** | Track processed data & model files |
| API | **FastAPI** | REST inference endpoint |
| Dashboard | **Streamlit + Plotly** | Interactive hierarchy visualization |
| Containerization | **Docker** | Reproducible deployment |

---

## Pipeline Phases

### Phase 1 — Data Audit & Optimization

- Loads the raw M5 CSV files with aggressive dtype downcasting (`reduce_mem_usage`)
- Converts wide-format sales (30,490 rows × 1,913 day columns) to long format
- Applies a **500-day sliding window** (d_1414 → d_1913) to limit memory
- Merges calendar metadata and sell prices
- **Output:** `sales_long_500d.parquet` (439.8 MB in RAM, 73.5 MB on disk)

### Phase 2 — Feature Engineering

17 new features built with **grouped operations** to prevent data leakage:

| Category | Features | Count |
|---|---|---|
| **Lags** | `sales_lag_7`, `sales_lag_14`, `sales_lag_28` | 3 |
| **Rolling Stats** | `sales_rmean_{7,28,60}`, `sales_rstd_{7,28,60}` | 6 |
| **Calendar** | `wday_sin/cos`, `month_sin/cos`, `is_event`, `snap` | 6 |
| **Price** | `price_momentum` (week-over-week % change) | 1 |
| **Temporal** | `day_num` | 1 |

- **Output:** `featured_data.parquet` (854.5 MB in RAM, 115.4 MB on disk)

### Phase 3 — Training & Reconciliation

- **Model:** Global LightGBM (Tweedie objective, 500 trees, 30 features)
- **Split:** Train on d_1442–d_1885, validate on d_1886–d_1913 (28 days)
- **Reconciliation Study:** BU, TD, MinTrace applied and evaluated at all 12 M5 hierarchy levels
- **Metric:** WRMSSE computed with dollar-sales weighting and naive-method scaling
- All runs logged to MLflow under `M5_Hierarchical_Forecasting`

### Phase 4 — Deployment

- FastAPI backend with `/predict`, `/metrics`, `/health`, `/hierarchy` endpoints
- Streamlit dashboard with **Drill-Down View** (hierarchy coherence) and **Research Insights** (WRMSSE leaderboard)
- Multi-stage Dockerfile + docker-compose for containerized deployment

---

## Results

### WRMSSE Comparison — All 12 Hierarchy Levels

| Level | Bottom-Up | Top-Down | MinTrace |
|---|---:|---:|---:|
| 1. Total | **0.4643** | **0.4643** | **0.4643** |
| 2. State | **0.4703** | 0.6124 | **0.4703** |
| 3. Store | **0.5138** | 0.7961 | **0.5138** |
| 4. Category | 0.5694 | **0.5299** | 0.5694 |
| 5. Department | **0.5913** | 0.6151 | **0.5913** |
| 6. State × Cat | **0.5641** | 0.6660 | **0.5641** |
| 7. State × Dept | **0.5805** | 0.7062 | **0.5805** |
| 8. Store × Cat | **0.6087** | 0.8255 | **0.6087** |
| 9. Store × Dept | **0.6314** | 0.8236 | **0.6314** |
| 10. Item | **0.8007** | 1.1162 | 1.1126 |
| 11. State × Item | **0.7913** | 0.9860 | 0.9779 |
| 12. Store × Item | **0.7879** | 0.8987 | 0.8920 |
| **Overall** | **0.6145** | 0.7533 | 0.6647 |
| **Coherence Error** | 0.0297 | 0.0349 | **0.0188** |

### Key Findings

1. **Bottom-Up achieves the best WRMSSE (0.6145)** — the global LightGBM model captures item-level patterns so well that simple aggregation outperforms formal reconciliation.
2. **MinTrace achieves 37% lower coherence error (0.0188)** — mathematically optimal reconciliation ensures cross-level consistency.
3. **Top-Down loses item-level detail** — disaggregation via historical proportions is too coarse (+22.6% worse WRMSSE).

### Top 5 Features by Importance

| Feature | Importance | Type |
|---|---:|---|
| `item_id` | 35,309 | Identity |
| `day_num` | 2,248 | Temporal |
| `store_id` | 2,229 | Identity |
| `sales_rmean_7` | 2,207 | Rolling |
| `sales_rmean_60` | 2,175 | Rolling |

---

## API Reference

### `GET /health`
Health check and model status.

```json
{"status": "healthy", "model_loaded": true, "n_reference_series": 30490}
```

### `POST /predict`
28-day demand forecast for a store-item pair.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "item_id": "FOODS_1_001"}'
```

```json
{
  "store_id": "CA_1",
  "item_id": "FOODS_1_001",
  "horizon_days": 28,
  "forecasts": [1.37, 1.37, 1.37, ...],
  "method": "LightGBM_BottomUp"
}
```

### `GET /metrics`
WRMSSE results from all reconciliation methods.

### `GET /hierarchy`
Available stores, items, states, categories, departments.

---

## Docker

### Build & Run

```bash
# Build both containers
docker-compose build

# Launch (detached)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Services

| Service | Port | URL |
|---|---|---|
| FastAPI API | 8000 | http://localhost:8000/docs |
| Streamlit Dashboard | 8501 | http://localhost:8501 |

---

## Memory-First Design

The entire pipeline is designed to run on a **laptop with ≤ 8 GB RAM**:

| Phase | Peak RSS |
|---|---|
| Data Audit & Merge | ~620 MB |
| Feature Engineering | ~1,047 MB |
| LightGBM Training | ~2 GB |
| API Inference | ~850 MB |

Key techniques:
- **Aggressive dtype downcasting** — `int8`/`int16` for integers, `float16`/`float32` for floats, `category` for strings
- **500-day sliding window** — only recent history for training, earlier data for feature computation only
- **Explicit `del` + `gc.collect()`** — intermediate DataFrames freed immediately
- **Parquet format** — 5-8× compression vs CSV, columnar access

---

## MLflow Experiment Tracking

```bash
# View experiments in the MLflow UI
mlflow ui --backend-store-uri mlruns/
```

Experiment: `M5_Hierarchical_Forecasting`

| Run | WRMSSE | Coherence Error |
|---|---|---|
| `baseline_lgbm_sku` | 0.6145 | 0.0297 |
| `reconciliation_bottom_up` | 0.6145 | 0.0297 |
| `reconciliation_top_down` | 0.7533 | 0.0349 |
| `reconciliation_mintrace` | 0.6647 | 0.0188 |

---

## Data

This project uses the [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset from Kaggle.

| File | Size | Description |
|---|---|---|
| `calendar.csv` | 101 KB | Date metadata, events, SNAP flags |
| `sales_train_validation.csv` | 114 MB | 30,490 SKUs × 1,913 days |
| `sales_train_evaluation.csv` | 116 MB | Extended to d_1941 |
| `sell_prices.csv` | 194 MB | Weekly prices per store-SKU |

Download and place in the `data/` directory before running the pipeline.

---

## License

This project is for educational and research purposes. The M5 dataset is provided by Walmart under the Kaggle competition terms.
