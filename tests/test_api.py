import importlib

import pytest
from fastapi.testclient import TestClient

from helpers import make_records


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("FORECAST_API_KEY", "geheim")
    import app.config, app.api

    importlib.reload(app.config)
    importlib.reload(app.api)
    return TestClient(app.api.app), app.api


def test_health_open_models_protected(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/v1/models").status_code == 401
    r = c.get("/v1/models", headers={"X-API-Key": "geheim"})
    assert r.status_code == 200
    names = {m["name"]: m for m in r.json()["models"]}
    assert names["seasonal_naive"]["available"] and names["seasonal_naive"]["baseline"]
    assert names["chronos_bolt_small"]["available"] is False  # Gewichte nicht vorhanden


def test_bearer_auth_and_forecast(client):
    c, _ = client
    body = {"series": make_records(2), "h": 6, "n_windows": 2}
    assert c.post("/v1/forecast", json=body).status_code == 401
    r = c.post("/v1/forecast", json=body, headers={"Authorization": "Bearer geheim"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["validated"] is True and len(j["forecast"]) == 12


def test_validation_errors_are_422(client):
    c, _ = client
    h = {"X-API-Key": "geheim"}
    assert c.post("/v1/forecast", json={"series": [], "h": 4}, headers=h).status_code == 422
    r = c.post("/v1/forecast", json={"series": make_records(1), "h": 4, "models": ["gibtsnicht"]}, headers=h)
    assert r.status_code == 422 and "Unbekanntes Modell" in r.json()["detail"]
    r = c.post("/v1/forecast", json={"series": make_records(1), "h": 4, "models": ["chronos_bolt_small"]}, headers=h)
    assert r.status_code == 422 and "nicht verfügbar" in r.json()["detail"]  # Gewichte fehlen -> klare Meldung


def test_int_ids_and_anomalies_and_backtest(client):
    c, _ = client
    h = {"X-API-Key": "geheim"}
    rec = [dict(r, unique_id=int(r["unique_id"][1:])) for r in make_records(anomaly_at=140)]
    r = c.post("/v1/anomalies", json={"series": rec, "model": "auto_ets", "h": 12, "n_windows": 3, "level": 95}, headers=h)
    assert r.status_code == 200 and r.json()["n_anomalies"] >= 1
    r = c.post("/v1/backtest", json={"series": rec, "h": 8, "models": ["naive", "theta"]}, headers=h)
    assert r.status_code == 200 and r.json()["best_model"] in {"naive", "theta", "seasonal_naive"}


def test_explain_off_and_on(client, monkeypatch):
    c, api = client
    h = {"X-API-Key": "geheim"}
    body = {"series": make_records(1), "h": 4, "question": "Warum?"}
    assert c.post("/v1/explain", json=body, headers=h).status_code == 501
    monkeypatch.setattr(api.explain_mod, "enabled", lambda: True)
    captured = {}

    def fake(summary, q):
        captured["s"], captured["q"] = summary, q
        return "Text"

    monkeypatch.setattr(api.explain_mod, "explain", fake)
    r = c.post("/v1/explain", json=body, headers=h)
    assert r.status_code == 200 and r.json()["explanation"] == "Text"
    assert captured["q"] == "Warum?" and captured["s"]["serien"][0]["serie"] == "s0"
