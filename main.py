"""FastAPI application exposing the AutoValue price model."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app import __version__
from app.schemas import (
    BatchRequest,
    BatchResponse,
    CarFeatures,
    HealthResponse,
    ModelInfo,
    PriceEstimate,
)
from app.service import ModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("autovalue.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    if model_path.exists():
        app.state.service = ModelService.load(model_path)
        logger.info("Loaded model from %s", model_path)
    else:
        app.state.service = None
        logger.warning("Model file %s not found; run `python -m app.train` first", model_path)
    yield


app = FastAPI(
    title="AutoValue - Used Car Price API",
    description="Estimates the resale price of a used car (INR lakh) with an 80% price range.",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def get_service(request: Request) -> ModelService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return service


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.service is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__)


@app.get("/model/info", response_model=ModelInfo, tags=["meta"])
def model_info(request: Request) -> ModelInfo:
    meta = get_service(request).metadata
    return ModelInfo(
        model_name=meta["model_name"],
        trained_at=meta["trained_at"],
        n_train=meta["n_train"],
        n_test=meta["n_test"],
        features=meta["features"],
        test_metrics=meta["test_metrics"],
        cv_r2=meta["cv_r2"],
    )


@app.post("/predict", response_model=PriceEstimate, tags=["prediction"])
def predict(car: CarFeatures, request: Request) -> PriceEstimate:
    return get_service(request).predict([car])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    estimates = get_service(request).predict(payload.cars)
    return BatchResponse(count=len(estimates), estimates=estimates)
