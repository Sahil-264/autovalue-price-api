"""Model loading and inference, independent of the web framework."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd

from app.data import FEATURES
from app.schemas import CarFeatures, PriceEstimate


class ModelService:
    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata
        self.q_low, self.q_high = metadata["interval_log_quantiles"]

    @classmethod
    def load(cls, path: str | Path) -> "ModelService":
        bundle = joblib.load(path)
        return cls(bundle["pipeline"], bundle["metadata"])

    def predict(self, cars: list[CarFeatures]) -> list[PriceEstimate]:
        frame = pd.DataFrame([c.model_dump() for c in cars])[FEATURES]
        preds = self.pipeline.predict(frame)
        results = []
        for p in preds:
            p = float(p)
            results.append(
                PriceEstimate(
                    estimated_price_lakh=round(p, 2),
                    lower_bound_lakh=round(p * math.exp(self.q_low), 2),
                    upper_bound_lakh=round(p * math.exp(self.q_high), 2),
                    estimated_price_inr=int(round(p * 100_000, -2)),
                )
            )
        return results
