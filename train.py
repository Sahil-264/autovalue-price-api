"""Train, compare and persist the price model.

Usage:
    python -m app.train
    python -m app.train --samples 8000 --out models/model.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import __version__
from app.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    generate_car_data,
)

logger = logging.getLogger("autovalue.train")


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + log-target regressor in a single serializable object."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    # Prices are right-skewed and errors are proportional, so model log(price).
    model = TransformedTargetRegressor(regressor=estimator, func=np.log, inverse_func=np.exp)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train(n_samples: int = 5000, seed: int = 42, out_path: str | Path = "models/model.joblib") -> dict:
    df = generate_car_data(n_samples=n_samples, seed=seed)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }

    cv_r2 = {}
    for name, estimator in candidates.items():
        scores = cross_val_score(build_pipeline(estimator), X_train, y_train, cv=5, scoring="r2")
        cv_r2[name] = float(scores.mean())
        logger.info("%s CV R2: %.4f (+/- %.4f)", name, scores.mean(), scores.std())

    best_name = max(cv_r2, key=cv_r2.get)
    pipeline = build_pipeline(candidates[best_name]).fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae_lakh": float(mean_absolute_error(y_test, pred)),
        "rmse_lakh": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mape": float(mean_absolute_percentage_error(y_test, pred)),
    }

    # Empirical 80% prediction interval from held-out log-residuals.
    log_resid = np.log(y_test.to_numpy()) - np.log(pred)
    q_low, q_high = np.quantile(log_resid, [0.10, 0.90])

    bundle = {
        "pipeline": pipeline,
        "metadata": {
            "model_name": best_name,
            "app_version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": FEATURES,
            "cv_r2": cv_r2,
            "test_metrics": metrics,
            "interval_log_quantiles": [float(q_low), float(q_high)],
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    out_path.with_suffix(".metrics.json").write_text(json.dumps(bundle["metadata"], indent=2))
    logger.info("Saved %s to %s | test R2 %.4f", best_name, out_path, metrics["r2"])
    return bundle["metadata"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the AutoValue price model")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()
    meta = train(args.samples, args.seed, args.out)
    print(json.dumps(meta["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
