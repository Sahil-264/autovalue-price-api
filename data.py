"""Synthetic used-car listings generator (prices in INR lakh).

Deterministic and seeded so results are reproducible and the repo is self-contained.
To use real data (e.g. a Kaggle / CarDekho used-car dataset), replace
`generate_car_data` with a CSV loader returning the same columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["age_years", "km_driven", "engine_cc", "owners"]
CATEGORICAL_FEATURES = ["brand", "fuel_type", "transmission", "city"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price_lakh"

# brand -> (base new-car price in lakh, market share weight, is_luxury)
BRANDS = {
    "maruti": (6.5, 0.28, 0),
    "hyundai": (9.0, 0.18, 0),
    "tata": (9.5, 0.12, 0),
    "honda": (11.0, 0.09, 0),
    "mahindra": (14.0, 0.10, 0),
    "toyota": (16.0, 0.10, 0),
    "bmw": (48.0, 0.04, 1),
    "mercedes": (52.0, 0.04, 1),
    "audi": (49.0, 0.05, 1),
}
FUEL_TYPES = ["petrol", "diesel", "cng"]
TRANSMISSIONS = ["manual", "automatic"]
CITIES = {
    "delhi": 0.00,
    "mumbai": 0.03,
    "bengaluru": 0.02,
    "hyderabad": 0.00,
    "pune": 0.01,
    "chennai": -0.01,
    "kolkata": -0.04,
    "bhubaneswar": -0.05,
}


def generate_car_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    names = list(BRANDS)
    weights = np.array([BRANDS[b][1] for b in names])
    brand = rng.choice(names, size=n_samples, p=weights / weights.sum())
    base = np.array([BRANDS[b][0] for b in brand])
    luxury = np.array([BRANDS[b][2] for b in brand])

    fuel = rng.choice(FUEL_TYPES, size=n_samples, p=[0.55, 0.38, 0.07])
    transmission = np.where(
        rng.random(n_samples) < np.where(luxury == 1, 0.9, 0.28), "automatic", "manual"
    )
    city = rng.choice(list(CITIES), size=n_samples)
    city_effect = np.array([CITIES[c] for c in city])

    age = np.clip(rng.gamma(2.0, 2.6, n_samples), 0, 15).round(1)
    km = np.clip(age * rng.normal(11000, 3500, n_samples) + rng.normal(2000, 1500, n_samples), 500, 250000).round(-2)
    engine = np.clip(rng.normal(1250 + 750 * luxury, 220, n_samples), 800, 3500).round(-1)
    owners = np.clip(1 + rng.poisson(0.35 + 0.08 * age), 1, 4)

    # Log-price model: luxury cars depreciate faster, km and ownership history matter.
    log_price = (
        np.log(base)
        - (0.085 + 0.035 * luxury) * age
        - 0.0000025 * km
        + 0.00018 * (engine - (1250 + 750 * luxury))
        + np.where(fuel == "diesel", 0.08, np.where(fuel == "cng", -0.06, 0.0))
        + np.where(transmission == "automatic", 0.10, 0.0)
        - 0.07 * (owners - 1)
        + city_effect
        + rng.normal(0, 0.08, n_samples)
    )
    price = np.clip(np.exp(log_price), 0.4, None).round(2)

    return pd.DataFrame(
        {
            "age_years": age,
            "km_driven": km.astype(int),
            "engine_cc": engine.astype(int),
            "owners": owners,
            "brand": brand,
            "fuel_type": fuel,
            "transmission": transmission,
            "city": city,
            TARGET: price,
        }
    )
