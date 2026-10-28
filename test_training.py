import joblib

from app.data import FEATURES, TARGET, generate_car_data


def test_data_generation_is_deterministic():
    assert generate_car_data(500, seed=1).equals(generate_car_data(500, seed=1))


def test_data_schema_and_sanity():
    df = generate_car_data(2000, seed=3)
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert not df.isna().any().any()
    assert (df[TARGET] > 0).all()
    assert df.groupby("brand")[TARGET].median()["bmw"] > df.groupby("brand")[TARGET].median()["maruti"]


def test_training_artifacts(model_path):
    assert model_path.exists()
    assert model_path.with_suffix(".metrics.json").exists()
    bundle = joblib.load(model_path)
    meta = bundle["metadata"]
    assert meta["test_metrics"]["r2"] > 0.85
    low, high = meta["interval_log_quantiles"]
    assert low < 0 < high
