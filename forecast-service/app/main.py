from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time

from fastapi import FastAPI, Request
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

from . import schemas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast_api")

MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")

COLUMN_MAP = {
    "gr_liv_area": "GrLivArea",
    "total_bsmt_sf": "TotalBsmtSF",
    "garage_area": "GarageArea",
    "year_built": "YearBuilt",
    "overall_qual": "OverallQual",
    "ms_zoning": "MSZoning",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        app.state.model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        app.state.model_version = str(mv.version)
        stats_path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="data_stats.json"
        )
        app.state.data_stats = json.loads(Path(stats_path).read_text())
        logger.info(f"Loaded {MODEL_NAME}@{MODEL_ALIAS} -> version={mv.version}")
    except Exception as exc:
        logger.critical(f"FAIL-FAST: Could not resolve model {MODEL_NAME}@{MODEL_ALIAS}: {exc}")
        raise RuntimeError(f"FAIL-FAST: Model alias {MODEL_NAME}@{MODEL_ALIAS} could not be resolved") from exc
    yield


app = FastAPI(title="Forecast API", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.model_version}


@app.get("/model/data-stats")
def model_data_stats(request: Request):
    return request.app.state.data_stats


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(payload: schemas.PredictRequest, request: Request):
    t0 = time.perf_counter()
    row = {COLUMN_MAP[k]: v for k, v in payload.model_dump().items()}
    df = pd.DataFrame([row])
    pred = round(float(request.app.state.model.predict(df)[0]), 2)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "predicted_price": pred,
        "currency": "USD",
        "model_version": request.app.state.model_version,
        "latency_ms": latency_ms,
    }
