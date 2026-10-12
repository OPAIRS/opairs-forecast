import numpy as np
import pandas as pd
import pytest

from app import core
from app.data import intermittency, to_frame
from app.registry import resolve_models
from helpers import make_records


def intermittent_records(n_series=6, n=72, seed=1):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2019-01-01", periods=n, freq="MS")
    rec = []
    for i in range(n_series):
        p, size = rng.uniform(0.15, 0.5), rng.uniform(3, 20)
        y = (rng.random(n) < p) * np.round(rng.lognormal(np.log(size), 0.4, n))
        rec += [{"unique_id": f"p{i}", "ds": str(d.date()), "y": float(v)} for d, v in zip(ds, y)]
    return rec


def test_intermittency_detection():
    inter = intermittency(to_frame(intermittent_records(3)))
    assert all(v["intermittent"] for v in inter.values())
    smooth = intermittency(to_frame(make_records(2)))
    assert not any(v["intermittent"] for v in smooth.values())
    # negative Werte: nie intermittierend
    rec = [{"unique_id": "n", "ds": f"2024-{m:02d}-01", "y": v} for m, v in zip(range(1, 13), [0, 0, -1, 0, 0, 2, 0, 0, 0, 3, 0, 0])]
    assert intermittency(to_frame(rec))["n"]["intermittent"] is False


def test_auto_adds_croston_family_only_for_intermittent():
    assert "croston_sba" not in resolve_models(["auto"])
    names = resolve_models(["auto"], intermittent=True)
    assert {"croston_sba", "tsb", "imapa", "adida"} <= set(names)
    # ausdrücklich angefragte Modelle bleiben unverändert
    assert resolve_models(["theta"], intermittent=True) == ["seasonal_naive", "theta"]


def test_intermittent_scored_by_rmse_not_mae():
    """MAE belohnt die Nullprognose, RMSE nicht. Bei intermittierenden Serien zählt der RMSE."""
    y = np.array([0.0] * 14 + [5.0] * 6)

    def bt(yhat):
        return pd.DataFrame({"unique_id": "a", "ds": pd.date_range("2024-01-01", periods=20, freq="D"), "window": 1, "y": y,
                             "yhat": yhat, "lo": np.nan, "hi": np.nan})

    results = {"base": bt(np.full(20, 3.0)), "zero": bt(np.zeros(20)), "mean": bt(np.full(20, y.mean()))}
    plain = core.score_models(results, "base")
    assert plain["zero"]["relative_mae"] < plain["mean"]["relative_mae"]  # MAE: Nullen gewinnen
    inter = core.score_models(results, "base", frozenset({"a"}))
    assert inter["mean"]["relative_mae"] < inter["zero"]["relative_mae"]  # RMSE: der Mittelwert gewinnt
    assert inter["mean"]["n_intermittent"] == 1


def test_forecast_with_intermittent_series():
    r = core.run_forecast(intermittent_records(), h=6, n_windows=3)
    assert r["errors"] == {}
    assert {"croston_sba", "tsb", "imapa", "adida"} <= set(r["backtest"])
    assert any("intermittierend" in w for w in r["warnings"])
    assert len(r["forecast"]) == 6 * 6
    assert r["backtest"]["seasonal_naive"]["relative_mae"] == pytest.approx(1.0)


def test_regular_series_unchanged():
    r = core.run_forecast(make_records(2), h=8, n_windows=2)
    assert "croston_sba" not in r["backtest"]
    assert not any("intermittierend" in w for w in r["warnings"])
    assert all("n_intermittent" not in s for s in r["backtest"].values())
