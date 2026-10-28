"""Pydantic request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CarFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "brand": "hyundai",
                "age_years": 4,
                "km_driven": 45000,
                "engine_cc": 1197,
                "owners": 1,
                "fuel_type": "petrol",
                "transmission": "manual",
                "city": "bhubaneswar",
            }
        }
    )

    brand: Literal["maruti", "hyundai", "tata", "honda", "mahindra", "toyota", "bmw", "mercedes", "audi"]
    age_years: float = Field(..., ge=0, le=30, description="Age of the car in years")
    km_driven: int = Field(..., ge=0, le=500000)
    engine_cc: int = Field(..., ge=600, le=6000)
    owners: int = Field(..., ge=1, le=10, description="Number of previous owners including current")
    fuel_type: Literal["petrol", "diesel", "cng"]
    transmission: Literal["manual", "automatic"]
    city: Literal["delhi", "mumbai", "bengaluru", "hyderabad", "pune", "chennai", "kolkata", "bhubaneswar"]


class PriceEstimate(BaseModel):
    estimated_price_lakh: float = Field(..., description="Point estimate in INR lakh")
    lower_bound_lakh: float = Field(..., description="Lower end of the 80% interval")
    upper_bound_lakh: float = Field(..., description="Upper end of the 80% interval")
    estimated_price_inr: int = Field(..., description="Point estimate in rupees")


class BatchRequest(BaseModel):
    cars: list[CarFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    estimates: list[PriceEstimate]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    model_name: str
    trained_at: str
    n_train: int
    n_test: int
    features: list[str]
    test_metrics: dict[str, float]
    cv_r2: dict[str, float]
