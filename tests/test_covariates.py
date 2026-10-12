"""Kovariaten: Datenfluss vom API-Format bis zum Adapter.

AutoARIMA (statsforecast) läuft hier echt. Chronos-2 und TiRex-2 laufen über Fake-Module und prüfen nur den
Datenfluss (Form der Eingaben); das echte Modell ist hier NICHT getestet, dafür gibt es live/live_covariates.py."""
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from app import core
from app.adapters.base import CovSpec
from app.data import DataError
from app.registry import UnknownModel, resolve_models


def promo_records(n_series=2, n=84, h=6, seed=3):
    """Monatsreihen, deren Wert stark von einer zukunftsbekannten Aktion (promo) abhängt, plus Rest: Temperatur nur Vergangenheit."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2018-01-01", periods=n + h, freq="MS")
    rec, fut = [], []
    for i in range(n_series):
        promo = (rng.random(n + h) < 0.3).astype(float)
        temp = 15 + 8 * np.sin(2 * np.pi * np.arange(n + h) / 12) + rng.normal(0, 1, n + h)
        y = 100 + 40 * promo + 0.2 * np.arange(n + h) + rng.normal(0, 2, n + h)
        for k in range(n + h):
            row = {"unique_id": f"s{i}", "ds": str(ds[k].date()), "promo": float(promo[k])}
            if k < n:
                rec.append({**row, "y": float(y[k]), "temp": float(temp[k])})
            else:
                fut.append(row)
    return rec, fut


def test_arima_with_known_future_covariate_beats_arima_without():
    rec, fut = promo_records()
    r = core.run_forecast(rec, h=6, models=["auto_arima"], n_windows=3, covariates=["promo"], future_covariates=fut, ensemble=False)
    assert r["errors"] == {}
    assert {"auto_arima", "auto_arima@cov"} <= set(r["backtest"])
    assert r["backtest"]["auto_arima@cov"]["relative_mae"] < r["backtest"]["auto_arima"]["relative_mae"] * 0.8
    assert r["selected_model"] == "auto_arima@cov" and r["validated"] is True
    assert r["covariates"] == {"future_known": ["promo"], "past_only": []}
    assert len(r["forecast"]) == 2 * 6
    # Die Prognose folgt der Aktion: Monate mit promo=1 liegen deutlich höher
    f = pd.DataFrame(r["forecast"]).merge(pd.DataFrame(fut)[["unique_id", "ds", "promo"]], on=["unique_id", "ds"])
    assert f[f.promo == 1]["yhat"].mean() > f[f.promo == 0]["yhat"].mean() + 20


def test_past_only_covariates_are_ignored_by_arima_with_warning():
    rec, fut = promo_records()
    r = core.run_forecast(rec, h=6, models=["theta", "auto_arima"], n_windows=2, covariates=["promo", "temp"], future_covariates=[{k: v for k, v in x.items()} for x in fut])
    assert r["covariates"]["past_only"] == ["temp"] and r["covariates"]["future_known"] == ["promo"]
    assert "auto_arima@cov" in r["backtest"]
    assert any("ignoriert: temp" in w for w in r["warnings"]) or r["selected_model"] != "auto_arima@cov"


def test_no_future_values_means_past_only_and_arima_cannot_use_them():
    rec, _ = promo_records()
    r = core.run_forecast(rec, h=6, models=["theta"], n_windows=2, covariates=["promo"])
    assert r["covariates"] == {"future_known": [], "past_only": ["promo"]}
    assert any("Keine Zukunftswerte" in w for w in r["warnings"])
    assert any("kein Modell der Auswahl kann sie nutzen" in w for w in r["warnings"])
    assert not any(n.endswith("@cov") for n in r["backtest"])


def test_covariates_do_not_change_plain_runs():
    rec, _ = promo_records()
    plain = [{k: v for k, v in x.items() if k in ("unique_id", "ds", "y")} for x in rec]
    r = core.run_forecast(plain, h=6, models=["theta"], n_windows=2)
    assert r["covariates"] is None and not any("Kovariat" in w for w in r["warnings"])


def test_validation_errors():
    rec, fut = promo_records()
    with pytest.raises(DataError, match="Kovariatenspalten fehlen"):
        core.run_forecast(rec, h=6, covariates=["gibt_es_nicht"], models=["theta"])
    bad = [dict(x) for x in rec]
    bad[3]["promo"] = None
    with pytest.raises(DataError, match="Kovariate 'promo'"):
        core.run_forecast(bad, h=6, covariates=["promo"], models=["theta"])
    with pytest.raises(DataError, match="heißen"):
        core.run_forecast(rec, h=6, covariates=["y"], models=["theta"])
    with pytest.raises(DataError, match="erwartet genau 6 Zeilen"):
        core.run_forecast(rec, h=6, covariates=["promo"], future_covariates=fut[:-1], models=["theta"])
    shifted = [dict(x, ds=str((pd.Timestamp(x["ds"]) + pd.DateOffset(months=1)).date())) for x in fut]
    with pytest.raises(DataError, match="erwartet genau 6 Zeilen"):
        core.run_forecast(rec, h=6, covariates=["promo"], future_covariates=shifted, models=["theta"])
    with pytest.raises(DataError, match="nicht in covariates"):
        core.run_forecast(rec, h=6, covariates=["temp"], future_covariates=fut, models=["theta"])
    with pytest.raises(DataError, match="ohne covariates"):
        core.run_forecast(rec, h=6, future_covariates=fut, models=["theta"])


def test_explicit_cov_variant_needs_covariates_and_capable_model():
    spec = CovSpec(("promo",), ("promo",))
    assert resolve_models(["auto_arima@cov"], cov=spec) == ["seasonal_naive", "auto_arima@cov"]
    with pytest.raises(UnknownModel, match="Kovariaten"):
        resolve_models(["auto_arima@cov"])
    with pytest.raises(UnknownModel, match="Kovariaten"):
        resolve_models(["theta@cov"], cov=spec)
    assert "auto_arima@cov" not in resolve_models(["auto_arima"], cov=CovSpec(("temp",), ()))  # ohne Zukunftswerte nicht nutzbar


def test_api_accepts_extra_columns_and_future_covariates(monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.delenv("FORECAST_API_KEY", raising=False)
    import app.config, app.api

    importlib.reload(app.config)
    importlib.reload(app.api)
    rec, fut = promo_records(1)
    c = TestClient(app.api.app)
    r = c.post("/v1/forecast", json={"series": rec, "h": 6, "models": ["auto_arima"], "n_windows": 2,
                                     "covariates": ["promo"], "future_covariates": fut})
    assert r.status_code == 200, r.text
    assert r.json()["covariates"]["future_known"] == ["promo"]
    r = c.post("/v1/backtest", json={"series": rec, "h": 6, "models": ["auto_arima"], "n_windows": 2,
                                     "covariates": ["promo"], "future_covariates": fut})
    assert r.status_code == 200 and "auto_arima@cov" in r.json()["backtest"]
    bad = c.post("/v1/forecast", json={"series": rec, "h": 6, "covariates": ["nix"]})
    assert bad.status_code == 422


# ----------------------------------------------------------------------------- Fakes für Chronos-2 und TiRex-2
def _future_df(rec_fut):
    return pd.DataFrame(rec_fut).assign(ds=lambda d: pd.to_datetime(d["ds"]))


def test_chronos2_adapter_builds_covariate_dicts(monkeypatch, tmp_path):
    seen = []
    torch = types.ModuleType("torch")
    torch.tensor = lambda x: np.asarray(x)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class T:
        def __init__(self, a): self.a = np.asarray(a)
        def cpu(self): return self
        def numpy(self): return self.a
        def __getitem__(self, i): return T(self.a[i])

    class Pipe:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            seen.extend(inputs)
            outs = [np.stack([np.full(prediction_length, 1.0 + d) for d in (-1, 0, 1)], axis=-1) for _ in inputs]
            return [T(o[None]) for o in outs], [T(o[None, :, 1]) for o in outs]

    class Base:
        @staticmethod
        def from_pretrained(path, device_map=None): return Pipe()

    ch = types.ModuleType("chronos"); ch.BaseChronosPipeline = Base
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chronos", ch)
    from app.adapters.chronos import ChronosAdapter
    from app.data import to_frame

    rec, fut = promo_records(2, n=40)
    spec = CovSpec(("promo", "temp"), ("promo",))
    df = to_frame(rec, ["promo", "temp"])
    a = ChronosAdapter("c2", str(tmp_path), family="chronos2")
    assert a.supports_covariates and a.can_use(spec)
    assert not ChronosAdapter("b", str(tmp_path), family="bolt").supports_covariates
    out = a.predict(df, 6, "MS", 80, 12, cov=spec, future_df=_future_df(fut)[["unique_id", "ds", "promo"]])
    assert len(out) == 12
    item = seen[0]
    assert set(item) == {"target", "past_covariates", "future_covariates"}
    assert set(item["past_covariates"]) == {"promo", "temp"} and set(item["future_covariates"]) == {"promo"}
    assert len(item["target"]) == 40 and all(len(v) == 40 for v in item["past_covariates"].values())
    assert len(item["future_covariates"]["promo"]) == 6
    s0 = pd.DataFrame(fut)
    np.testing.assert_allclose(item["future_covariates"]["promo"], s0[s0.unique_id == "s0"]["promo"].to_numpy(dtype=np.float32))
    # ohne Kovariaten: reine Zielreihe
    seen.clear()
    a.predict(df, 6, "MS", 80, 12)
    assert not isinstance(seen[0], dict)


def test_tirex2_adapter_builds_covariate_tensors(monkeypatch, tmp_path):
    seen = []
    torch = types.ModuleType("torch")
    torch.tensor = lambda x: np.asarray(x)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    @dataclass
    class TimeseriesType:
        target: object
        past_covariates: object
        future_covariates: object

    class FakeModel:
        quantiles = [0.1, 0.5, 0.9]
        def forecast(self, timeseries, prediction_length, output_type, batch_size):
            seen.extend(timeseries)
            return [np.ones((1, 3, prediction_length)) for _ in timeseries]

    tx = types.ModuleType("tirex2"); tx.TimeseriesType = TimeseriesType
    tx.load_model = lambda path, device, hf_kwargs=None: FakeModel()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "tirex2", tx)
    from app.adapters.tirex import TiRex2Adapter
    from app.data import to_frame

    rec, fut = promo_records(2, n=40)
    df = to_frame(rec, ["promo", "temp"])
    a = TiRex2Adapter("t", str(tmp_path), device="cpu")
    spec = CovSpec(("promo", "temp"), ("promo",))
    a.predict(df, 6, "MS", 80, 12, cov=spec, future_df=_future_df(fut)[["unique_id", "ds", "promo"]])
    ts = seen[0]
    assert ts.target.shape == (1, 40)
    assert ts.past_covariates.shape == (1, 40)  # nur temp (past-only)
    assert ts.future_covariates.shape == (1, 46)  # promo über Historie + Horizont
    s0 = pd.DataFrame(rec); f0 = pd.DataFrame(fut)
    expect = np.concatenate([s0[s0.unique_id == "s0"]["promo"].to_numpy(), f0[f0.unique_id == "s0"]["promo"].to_numpy()])
    np.testing.assert_allclose(ts.future_covariates[0], expect.astype(np.float32))
    # nur Zukunftskovariaten / nur Vergangenheitskovariaten
    seen.clear()
    a.predict(df, 6, "MS", 80, 12, cov=CovSpec(("temp",), ()), future_df=None)
    assert seen[0].future_covariates is None and seen[0].past_covariates.shape == (1, 40)
    seen.clear()
    a.predict(df, 6, "MS", 80, 12)
    assert seen[0].past_covariates is None and seen[0].future_covariates is None


def test_fake_chronos2_runs_in_backtest_with_cov_variant(monkeypatch, tmp_path):
    """Gesamte Pipeline: das Fake-Chronos-2 bekommt im Backtest je Fenster die Zukunftswerte aus den Testzeilen."""
    import app.registry as reg

    seen = []
    torch = types.ModuleType("torch")
    torch.tensor = lambda x: np.asarray(x)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class T:
        def __init__(self, a): self.a = np.asarray(a)
        def cpu(self): return self
        def numpy(self): return self.a
        def __getitem__(self, i): return T(self.a[i])

    class Pipe:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            outs = []
            for it in inputs:
                seen.append(it)
                base = it["target"][-1] if isinstance(it, dict) else it[-1]
                lift = it["future_covariates"]["promo"] * 40 if isinstance(it, dict) and "future_covariates" in it else 0
                q = np.stack([np.broadcast_to(np.asarray(base + lift + d, dtype=float), (prediction_length,)) for d in (-2, 0, 2)], axis=-1)
                outs.append(q)
            return [T(o[None]) for o in outs], [T(o[None, :, 1]) for o in outs]

    class Base:
        @staticmethod
        def from_pretrained(path, device_map=None): return Pipe()

    ch = types.ModuleType("chronos"); ch.BaseChronosPipeline = Base
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chronos", ch)
    reg.get_adapter.cache_clear()
    monkeypatch.setitem(reg.load_config()["models"], "fake_c2", {"type": "chronos", "family": "chronos2", "path": str(tmp_path)})
    rec, fut = promo_records(2, n=60)
    r = core.run_backtest(rec, h=6, models=["fake_c2"], n_windows=3, covariates=["promo"], future_covariates=fut, ensemble=False)
    assert r["errors"] == {} and {"fake_c2", "fake_c2@cov"} <= set(r["backtest"])
    assert any(isinstance(i, dict) and "future_covariates" in i for i in seen)
    reg.get_adapter.cache_clear()
