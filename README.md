# AutoValue: Used Car Price Prediction API

A machine learning service that estimates the resale price of a used car in INR lakh, together with an 80% price range. It covers the full path from data to a deployed endpoint: reproducible training, model comparison, a validated REST API, tests, Docker and CI.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E)
![CI](https://github.com/<your-username>/autovalue-price-api/actions/workflows/ci.yml/badge.svg)

## Features

- **Regression with uncertainty**: every prediction returns a point estimate and a lower/upper bound, derived from the empirical distribution of held-out residuals (80% interval).
- **Log-price modelling**: prices are right-skewed with proportional errors, so models are trained on `log(price)` via `TransformedTargetRegressor` and predictions are returned on the original scale.
- **Automatic model selection**: Ridge, Random Forest and Gradient Boosting are compared with 5-fold cross-validated R²; the best is saved.
- **One serialized `Pipeline`** (scaling + one-hot encoding + model), so serving applies exactly the same preprocessing as training.
- **FastAPI service** with strict Pydantic validation, single and batch estimates, health and model-info endpoints, request timing middleware and Swagger docs.
- **Graceful degradation**: `/health` reports `degraded` and prediction routes return `503` when no model is loaded.
- **Tests** (pytest), **GitHub Actions CI** (lint, tests, Docker build) and a **Docker** image that trains the model at build time.

## Quick start

```bash
git clone https://github.com/<your-username>/autovalue-price-api.git
cd autovalue-price-api

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m app.train                  # trains and writes models/model.joblib
uvicorn app.main:app --reload        # http://localhost:8000/docs
```

### Docker

```bash
docker build -t autovalue-api .
docker run -p 8000:8000 autovalue-api
```

## API

| Method | Endpoint          | Description                                   |
|--------|-------------------|-----------------------------------------------|
| GET    | `/health`         | Liveness and whether a model is loaded        |
| GET    | `/model/info`     | Model name, training date, metrics, features  |
| POST   | `/predict`        | Estimate the price of one car                 |
| POST   | `/predict/batch`  | Estimate up to 1000 cars in one request       |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "hyundai",
    "age_years": 4,
    "km_driven": 45000,
    "engine_cc": 1197,
    "owners": 1,
    "fuel_type": "petrol",
    "transmission": "manual",
    "city": "bhubaneswar"
  }'
```

Response (values from a trained model, seed 42):

```json
{
  "estimated_price_lakh": 5.42,
  "lower_bound_lakh": 4.88,
  "upper_bound_lakh": 6.09,
  "estimated_price_inr": 542100
}
```

Invalid input (unknown brand, negative mileage, missing field) returns `422` with a descriptive error.

## Model results

Trained on 5,000 synthetic listings (80/20 split, seed 42):

| Model             | CV R² |
|-------------------|------:|
| Ridge             | 0.972 |
| Random Forest     | 0.954 |
| Gradient Boosting | 0.953 |

The selected model (Ridge on log-price) scores on the held-out test set:

| Metric | Value |
|--------|------:|
| R²     | 0.973 |
| MAE    | 0.68 lakh |
| RMSE   | 1.46 lakh |
| MAPE   | 7.0%  |

The 80% interval is roughly -10% / +12% around the point estimate. Exact figures are written to `models/model.metrics.json` after each run.

Honest notes on these numbers:

- The data is **synthetic**, generated from a log-linear depreciation formula plus noise. That is why a regularised linear model beats the tree ensembles and why R² is so high. Real listings are messier and would give lower scores, with tree-based models more competitive.
- The interval is a global residual quantile, so its width is the same in percentage terms for every car. A natural improvement is quantile regression or conformal prediction for per-car intervals.

## Project structure

```
app/
  data.py       synthetic listings generator and feature definitions
  train.py      training, cross-validated model selection, interval calibration
  schemas.py    Pydantic request/response models and validation rules
  service.py    model loading and inference (framework independent)
  main.py       FastAPI app, routes, middleware
tests/          pytest suite (API, validation, training)
Dockerfile
.github/workflows/ci.yml
```

## Configuration

| Variable     | Default                | Purpose                    |
|--------------|------------------------|----------------------------|
| `MODEL_PATH` | `models/model.joblib`  | Location of the model file |

## Using real data

Replace `generate_car_data()` in `app/data.py` with a loader that reads a CSV (for example a public used-car dataset from Kaggle) and returns the same columns, converting prices to INR lakh. Training, serving and tests need no other changes.

## Possible extensions

- Conformal prediction or quantile regression for per-car intervals
- SHAP explanations showing what drives each estimate
- Experiment tracking with MLflow
- A small web front end for sellers and buyers
- Deploy to Render, Railway or AWS

## License

MIT
   ![API docs](docs/screenshot.png)
