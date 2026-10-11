import numpy as np
import pandas as pd
import pytest

from app import config, core
from app.adapters import stats as stats_mod
from app.registry import get_adapter
from helpers import make_records


def _df(n_series=6, n=104):
    ds = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    rng = np.random.default_rng(1)
    rows = []
    for s in range(n_series):
        y = 100 + s * 10 + 8 * np.sin(np.arange(n) * 2 * np.pi / 52) + rng.normal(0, 2, n)
        rows.append(pd.DataFrame({"unique_id": f"s{s}", "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_jobs_rule(monkeypatch):
    monkeypatch.setattr(config, "N_JOBS", 4)
    j = stats_mod.StatsAdapter._jobs
    assert j(2) == 1 and j(3) == 1  # zu wenige Serien lohnen sich nicht
    assert j(4) == 4 and j(100) == 4
    monkeypatch.setattr(config, "N_JOBS", 1)
    assert j(100) == 1
    monkeypatch.setattr(config, "N_JOBS", -1)
    assert j(100) >= 1


def test_parallel_matches_serial(monkeypatch):
    df = _df()
    ad = get_adapter("auto_ets")
    monkeypatch.setattr(config, "N_JOBS", 1)
    a = ad.predict(df, 8, "W-MON", 80, 52).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    monkeypatch.setattr(config, "N_JOBS", 2)
    b = ad.predict(df, 8, "W-MON", 80, 52).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-6)


def test_parallel_failure_falls_back_to_serial(monkeypatch):
    df = _df()
    monkeypatch.setattr(config, "N_JOBS", 2)
    import statsforecast

    orig = statsforecast.StatsForecast.forecast

    def flaky(self, *a, **k):
        if self.n_jobs > 1:
            raise RuntimeError("Pool kaputt")
        return orig(self, *a, **k)

    monkeypatch.setattr(statsforecast.StatsForecast, "forecast", flaky)
    out = get_adapter("theta").predict(df, 4, "W-MON", 80, 52)
    assert len(out) == 6 * 4 and out["yhat"].notna().all()


def test_run_forecast_with_jobs(monkeypatch):
    monkeypatch.setattr(config, "N_JOBS", 2)
    r = core.run_forecast(make_records(5), h=8, models=["seasonal_naive", "theta"], n_windows=2)
    assert r["selected_model"] in {"theta", "seasonal_naive"} and len(r["forecast"]) == 5 * 8
