import pytest

from app import core
from app.data import DataError, default_season_length
from helpers import make_records


def test_auto_selects_and_validates():
    r = core.run_forecast(make_records(), h=8, n_windows=2)
    assert r["baseline"] == "seasonal_naive"
    assert r["selected_model"] != "seasonal_naive"
    assert r["validated"] is True
    assert r["backtest"][r["selected_model"]]["relative_mae"] < 1
    assert r["backtest"]["seasonal_naive"]["relative_mae"] == pytest.approx(1.0)
    assert len(r["forecast"]) == 3 * 8
    row = r["forecast"][0]
    assert set(row) == {"unique_id", "ds", "yhat", "lo", "hi"} and row["lo"] <= row["yhat"] <= row["hi"]
    assert r["freq"] == "W-MON" and r["season_length"] == 52


def test_refuses_when_baseline_wins():
    # Reine Wiederholung der Saison: Seasonal-Naive ist praktisch perfekt, Kandidaten können nicht besser sein
    import numpy as np, pandas as pd
    ds = pd.date_range("2022-01-03", periods=156, freq="W-MON")
    y = np.tile(np.arange(52, dtype=float) * 3 + 10, 3)
    rec = [{"unique_id": "a", "ds": str(d.date()), "y": float(v)} for d, v in zip(ds, y)]
    r = core.run_forecast(rec, h=8, models=["naive", "historic_average"], n_windows=2)
    assert r["selected_model"] == "seasonal_naive"
    assert r["validated"] is False
    assert any("schlägt die Baseline" in w for w in r["warnings"])


def test_no_backtest_single_model():
    r = core.run_forecast(make_records(1), h=4, models=["theta"], n_windows=0)
    assert r["selected_model"] == "theta" and r["validated"] is None and r["backtest"] == {}
    with pytest.raises(DataError):
        core.run_forecast(make_records(1), h=4, models=["theta", "naive"], n_windows=0)


def test_isolated_model_failure(monkeypatch):
    from app.adapters.stats import StatsAdapter

    orig = StatsAdapter.predict

    def boom(self, *a, **k):
        if self.name == "theta":
            raise RuntimeError("kaputt")
        return orig(self, *a, **k)

    monkeypatch.setattr(StatsAdapter, "predict", boom)
    r = core.run_forecast(make_records(2), h=8, models=["naive", "theta", "auto_ets"], n_windows=2)
    assert "theta" in r["errors"] and "theta" not in r["backtest"]
    assert r["selected_model"] in {"auto_ets", "naive", "seasonal_naive", "ensemble(auto_ets+naive)"}


def test_data_errors():
    rec = make_records(1, n=60)
    with pytest.raises(DataError, match="Lücken"):
        core.run_forecast(rec[:20] + rec[22:], h=4, freq="W-MON")
    with pytest.raises(DataError, match="Doppelter"):
        core.run_forecast(rec + [rec[0]], h=4)
    bad = [dict(r) for r in rec]; bad[3]["y"] = None
    with pytest.raises(DataError, match="y-Werte"):
        core.run_forecast(bad, h=4)
    with pytest.raises(DataError, match="Spalten fehlen"):
        core.run_forecast([{"unique_id": "a", "ds": "2024-01-01"}], h=4)
    with pytest.raises(DataError, match="lang genug"):
        core.run_forecast(make_records(1, n=30), h=8, n_windows=2)


def test_anomalies_find_injected_spike():
    a = core.run_anomalies(make_records(anomaly_at=140), model="auto_ets", h=12, n_windows=3, level=95)
    hits = [x for x in a["anomalies"] if x["unique_id"] == "s2" and x["direction"] == "high"]
    assert any(x["y"] > x["hi"] for x in hits)
    # die Spitze liegt in Woche 140 -> 2022-01-03 + 140 Wochen = 2024-09-09
    assert any(x["ds"] == "2024-09-09" for x in hits)


def test_anomalies_need_intervals():
    with pytest.raises(DataError, match="Prognoseintervalle"):
        core.run_anomalies(make_records(1), model="croston", h=8, n_windows=2)


def test_season_length_defaults():
    assert [default_season_length(f) for f in ("D", "W-MON", "MS", "ME", "QS", "h", "15min", "B", "YS")] == [
        7, 52, 12, 12, 4, 24, 96, 5, 1]


def test_daily_and_monthly():
    r = core.run_forecast(make_records(2, n=120, freq="MS", start="2015-01-01"), h=6, n_windows=2)
    assert r["freq"] == "MS" and r["season_length"] == 12 and len(r["forecast"]) == 12
    r = core.run_forecast(make_records(1, n=200, freq="D", start="2024-01-01"), h=14, n_windows=2)
    assert r["season_length"] == 7 and r["forecast"][0]["ds"] == "2024-07-19"


def test_ensemble_is_scored_and_blends_members():
    import pandas as pd
    from app.registry import get_adapter

    recs = make_records(2)
    r = core.run_backtest(recs, h=8, models=["seasonal_naive", "auto_ets", "theta", "naive"], n_windows=2)
    ens = [n for n in r["backtest"] if n.startswith("ensemble(")]
    assert len(ens) == 1
    members = r["backtest"][ens[0]]["members"]
    assert len(members) == 2 and all(r["backtest"][m]["beats_baseline"] for m in members)
    assert r["backtest"][ens[0]]["relative_mae"] is not None

    f = core.run_forecast(recs, h=8, models=["seasonal_naive", "auto_ets", "theta"], n_windows=2)
    if f["selected_model"].startswith("ensemble("):
        mem = core.ensemble_members(f["selected_model"])
        prep = core.prepare(recs, None, None, 8)
        fr = [get_adapter(m).predict(prep.df, 8, prep.freq, 80, prep.season_length) for m in mem]
        got = pd.DataFrame(f["forecast"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)
        want = sum(x.sort_values(["unique_id", "ds"]).reset_index(drop=True)["yhat"] for x in fr) / len(fr)
        assert (got["yhat"] - want).abs().max() < 1e-6
        assert any("Ensemble" in w for w in f["warnings"])


def test_ensemble_can_be_disabled_and_needs_two_winners():
    recs = make_records(2)
    r = core.run_forecast(recs, h=8, models=["seasonal_naive", "auto_ets", "theta"], n_windows=2, ensemble=False)
    assert not any(n.startswith("ensemble(") for n in r["backtest"])
    assert not r["selected_model"].startswith("ensemble(")
    r1 = core.run_backtest(recs, h=8, models=["seasonal_naive", "auto_ets"], n_windows=2)
    assert not any(n.startswith("ensemble(") for n in r1["backtest"])


def test_ensemble_final_failure_falls_back(monkeypatch):
    recs = make_records(2)
    base = core.run_forecast(recs, h=8, models=["seasonal_naive", "auto_ets", "theta"], n_windows=2)
    if not base["selected_model"].startswith("ensemble("):
        pytest.skip("Ensemble wurde hier nicht gewählt")
    orig = core._predict_final

    def boom(selected, *a, **k):
        if selected.startswith("ensemble("):
            raise RuntimeError("kaputt")
        return orig(selected, *a, **k)

    monkeypatch.setattr(core, "_predict_final", boom)
    r = core.run_forecast(recs, h=8, models=["seasonal_naive", "auto_ets", "theta"], n_windows=2)
    assert r["selected_model"] == "seasonal_naive" and r["validated"] is False
    assert base["selected_model"] in r["errors"]
