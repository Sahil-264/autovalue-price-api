import os

import pytest
from fastapi.testclient import TestClient

from app.train import train


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    train(n_samples=1500, seed=7, out_path=path)
    return path


@pytest.fixture(scope="session")
def client(model_path):
    os.environ["MODEL_PATH"] = str(model_path)
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def new_economy_car():
    return {
        "brand": "maruti", "age_years": 1, "km_driven": 8000, "engine_cc": 1200, "owners": 1,
        "fuel_type": "petrol", "transmission": "manual", "city": "delhi",
    }


@pytest.fixture
def old_economy_car():
    return {
        "brand": "maruti", "age_years": 12, "km_driven": 140000, "engine_cc": 1200, "owners": 3,
        "fuel_type": "petrol", "transmission": "manual", "city": "delhi",
    }


@pytest.fixture
def luxury_car():
    return {
        "brand": "bmw", "age_years": 3, "km_driven": 30000, "engine_cc": 2000, "owners": 1,
        "fuel_type": "diesel", "transmission": "automatic", "city": "mumbai",
    }
