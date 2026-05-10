"""
FastAPI Inference API for M5 Hierarchical Forecasting.

Endpoints:
    GET  /health          - Health check
    POST /predict          - 28-day forecast for a (store_id, item_id) pair
    GET  /metrics          - WRMSSE results from best MLflow run
    GET  /hierarchy        - Hierarchy breakdown (national -> state -> store)
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.loader import reduce_mem_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_PATH = PROCESSED_DIR / "lgbm_model.txt"
FEATURED_PATH = PROCESSED_DIR / "featured_data.parquet"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="M5 Hierarchical Forecast API",
    description="Retail demand forecasting with hierarchical reconciliation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------
model: Optional[lgb.Booster] = None
feature_data: Optional[pd.DataFrame] = None
wrmsse_results: Dict = {}
hierarchy_info: Dict = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    store_id: str
    item_id: str


class PredictResponse(BaseModel):
    store_id: str
    item_id: str
    horizon_days: int
    forecasts: List[float]
    method: str


class MetricsResponse(BaseModel):
    best_method: str
    wrmsse_overall: float
    coherence_error: float
    level_scores: Dict[str, float]
    methods_comparison: Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_model_and_data():
    """Load LightGBM model and prepare reference data at startup."""
    global model, feature_data, wrmsse_results, hierarchy_info

    logger.info("Loading LightGBM model from %s ...", MODEL_PATH)
    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        return
    model = lgb.Booster(model_file=str(MODEL_PATH))
    logger.info("Model loaded: %d features", model.num_feature())

    # Load a small reference slice for prediction features
    logger.info("Loading feature reference data ...")
    if FEATURED_PATH.exists():
        df = pd.read_parquet(FEATURED_PATH)
        df = reduce_mem_usage(df, verbose=False)  # Match training dtypes
        # Keep only the last day per (store, item) as a feature template
        df["day_num"] = df["d"].astype(str).str.split("_").str[1].astype(int)
        feature_data = (
            df.sort_values("day_num")
            .groupby(["store_id", "item_id"], observed=True)
            .last()
            .reset_index()
        )

        # Build hierarchy info
        hierarchy_info = {
            "stores": sorted(df["store_id"].unique().tolist()),
            "items": sorted(df["item_id"].unique().tolist()),
            "states": sorted(df["state_id"].unique().tolist()),
            "categories": sorted(df["cat_id"].unique().tolist()),
            "departments": sorted(df["dept_id"].unique().tolist()),
            "n_series": int(df.groupby(["store_id", "item_id"]).ngroups),
        }

        del df
        gc.collect()
        logger.info("Reference data loaded: %d series", len(feature_data))

    # Pre-computed WRMSSE results from Phase 3
    wrmsse_results = {
        "best_method": "Bottom-Up",
        "methods_comparison": {
            "Baseline_BU": {
                "wrmsse_overall": 0.6145, "coherence_error": 0.0297,
                "level_1_Total": 0.4643, "level_2_State": 0.4703,
                "level_3_Store": 0.5138, "level_4_Category": 0.5694,
                "level_5_Department": 0.5913, "level_6_State_x_Cat": 0.5641,
                "level_7_State_x_Dept": 0.5805, "level_8_Store_x_Cat": 0.6087,
                "level_9_Store_x_Dept": 0.6314, "level_10_Item": 0.8007,
                "level_11_State_x_Item": 0.7913, "level_12_Store_x_Item": 0.7879,
            },
            "Top_Down": {
                "wrmsse_overall": 0.7533, "coherence_error": 0.0349,
                "level_1_Total": 0.4643, "level_2_State": 0.6124,
                "level_3_Store": 0.7961, "level_4_Category": 0.5299,
                "level_5_Department": 0.6151, "level_6_State_x_Cat": 0.6660,
                "level_7_State_x_Dept": 0.7062, "level_8_Store_x_Cat": 0.8255,
                "level_9_Store_x_Dept": 0.8236, "level_10_Item": 1.1162,
                "level_11_State_x_Item": 0.9860, "level_12_Store_x_Item": 0.8987,
            },
            "MinTrace_OLS": {
                "wrmsse_overall": 0.6647, "coherence_error": 0.0188,
                "level_1_Total": 0.4643, "level_2_State": 0.4703,
                "level_3_Store": 0.5138, "level_4_Category": 0.5694,
                "level_5_Department": 0.5913, "level_6_State_x_Cat": 0.5641,
                "level_7_State_x_Dept": 0.5805, "level_8_Store_x_Cat": 0.6087,
                "level_9_Store_x_Dept": 0.6314, "level_10_Item": 1.1126,
                "level_11_State_x_Item": 0.9779, "level_12_Store_x_Item": 0.8920,
            },
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "n_reference_series": len(feature_data) if feature_data is not None else 0,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Generate 28-day demand forecast for a store-item pair."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if feature_data is None:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    # Find the reference row for this (store, item)
    mask = (
        (feature_data["store_id"] == req.store_id)
        & (feature_data["item_id"] == req.item_id)
    )
    ref = feature_data[mask]
    if ref.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for store={req.store_id}, item={req.item_id}",
        )

    # Build 28-day feature matrix by varying day_num
    ref_row = ref.iloc[0]
    base_day = int(ref_row.get("day_num", 1913))

    rows = []
    for d in range(1, 29):
        row = ref_row.copy()
        row["day_num"] = base_day + d
        rows.append(row)

    pred_df = pd.DataFrame(rows)

    # Select model features
    model_features = model.feature_name()
    for col in model_features:
        if col not in pred_df.columns:
            pred_df[col] = 0

    X = pred_df[model_features]

    # Ensure categoricals
    for col in ["store_id", "item_id", "cat_id", "dept_id", "state_id"]:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Predict
    preds = model.predict(X)
    preds = np.clip(preds, 0, None).tolist()

    return PredictResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        horizon_days=28,
        forecasts=[round(p, 2) for p in preds],
        method="LightGBM_BottomUp",
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Return WRMSSE results from the best-performing run."""
    best = wrmsse_results["methods_comparison"]["Baseline_BU"]
    return MetricsResponse(
        best_method=wrmsse_results["best_method"],
        wrmsse_overall=best["wrmsse_overall"],
        coherence_error=best["coherence_error"],
        level_scores={k: v for k, v in best.items()
                      if k.startswith("level_")},
        methods_comparison=wrmsse_results["methods_comparison"],
    )


@app.get("/hierarchy")
async def hierarchy():
    """Return hierarchy structure info."""
    return hierarchy_info
