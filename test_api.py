def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "version": r.json()["version"]}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["test_metrics"]["r2"] > 0.85
    assert "brand" in body["features"]


def test_predict_valid(client, new_economy_car):
    r = client.post("/predict", json=new_economy_car)
    assert r.status_code == 200
    body = r.json()
    assert body["estimated_price_lakh"] > 0
    assert body["lower_bound_lakh"] <= body["estimated_price_lakh"] <= body["upper_bound_lakh"]
    assert "X-Process-Time-ms" in r.headers


def test_newer_car_costs_more_than_older(client, new_economy_car, old_economy_car):
    new = client.post("/predict", json=new_economy_car).json()["estimated_price_lakh"]
    old = client.post("/predict", json=old_economy_car).json()["estimated_price_lakh"]
    assert new > old


def test_luxury_costs_more_than_economy(client, new_economy_car, luxury_car):
    eco = client.post("/predict", json=new_economy_car).json()["estimated_price_lakh"]
    lux = client.post("/predict", json=luxury_car).json()["estimated_price_lakh"]
    assert lux > eco


def test_predict_batch(client, new_economy_car, luxury_car):
    r = client.post("/predict/batch", json={"cars": [new_economy_car, luxury_car]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["estimates"]) == 2


def test_rejects_unknown_brand(client, new_economy_car):
    new_economy_car["brand"] = "tesla"
    assert client.post("/predict", json=new_economy_car).status_code == 422


def test_rejects_negative_km(client, new_economy_car):
    new_economy_car["km_driven"] = -5
    assert client.post("/predict", json=new_economy_car).status_code == 422


def test_rejects_missing_field(client, new_economy_car):
    del new_economy_car["fuel_type"]
    assert client.post("/predict", json=new_economy_car).status_code == 422


def test_rejects_empty_batch(client):
    assert client.post("/predict/batch", json={"cars": []}).status_code == 422
