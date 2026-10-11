"""TiRex-2-Adapter ohne echtes torch/Gewichte: Fake-Module prüfen nur den Datenfluss
(Eingabeform (1, T), Quantilauswahl aus model.quantiles, Zukunftsdaten, Batching, Randfälle).
Das echte Modell ist hier NICHT getestet."""
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from helpers import make_records


def install_fakes(monkeypatch, quantiles, calls):
    torch = types.ModuleType("torch")
    torch.tensor = lambda x: np.asarray(x)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    @dataclass
    class TimeseriesType:
        target: object
        past_covariates: object
        future_covariates: object

    class FakeForecastModel:
        def __init__(self):
            self.quantiles = quantiles

        def forecast(self, timeseries, prediction_length, output_type, batch_size):
            calls.append({"n": len(timeseries), "shape": timeseries[0].target.shape, "out": output_type, "bs": batch_size})
            q = np.asarray(quantiles)
            outs = []
            for ts in timeseries:
                last = float(ts.target[0, -1])
                # Wert = letzter Istwert + 100 * Quantilniveau -> Quantile sind im Ergebnis ablesbar
                outs.append(np.stack([np.full(prediction_length, last + 100 * lv) for lv in q])[None])  # (1, Q, h)
            return outs

    tx = types.ModuleType("tirex2")
    tx.TimeseriesType = TimeseriesType
    tx.load_model = lambda path, device, hf_kwargs=None: (calls.append({"load": path, "device": device, "hf": hf_kwargs}), FakeForecastModel())[1]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "tirex2", tx)


@pytest.fixture()
def model_dir(tmp_path):
    (tmp_path / "model-config.yaml").write_text("x: 1")
    (tmp_path / "model.ckpt").write_bytes(b"0")
    return str(tmp_path)


def test_available_checks_files(tmp_path, model_dir, monkeypatch):
    from app.adapters.tirex import TiRex2Adapter

    install_fakes(monkeypatch, [0.1, 0.5, 0.9], [])
    assert TiRex2Adapter("t", model_dir).available() == (True, "")
    ok, why = TiRex2Adapter("t", str(tmp_path / "nix")).available()
    assert not ok and "model-config.yaml" in why


def test_dataflow_and_quantile_pick(monkeypatch, model_dir):
    from app.adapters.tirex import TiRex2Adapter
    from app.data import to_frame

    calls = []
    levels = [round(0.1 * i, 1) for i in range(1, 10)]
    install_fakes(monkeypatch, levels, calls)
    a = TiRex2Adapter("t", model_dir, device="cpu", batch_size=7)
    df = to_frame(make_records(2, n=60))
    out = a.predict(df, h=5, freq="W-MON", level=80, season_length=52)
    assert list(out.columns) == ["unique_id", "ds", "yhat", "lo", "hi"] and len(out) == 10
    assert calls[0] == {"load": model_dir, "device": "cpu", "hf": {"local_files_only": True}}
    assert calls[1]["n"] == 2 and calls[1]["shape"] == (1, 60) and calls[1]["out"] == "numpy" and calls[1]["bs"] == 7
    s0 = out[out.unique_id == "s0"]
    last = df[df.unique_id == "s0"].iloc[-1]
    assert s0["ds"].iloc[0] == last["ds"] + pd.Timedelta(days=7)
    assert s0["yhat"].iloc[0] == pytest.approx(last["y"] + 50)   # Median 0.5
    assert s0["lo"].iloc[0] == pytest.approx(last["y"] + 10)     # 0.1
    assert s0["hi"].iloc[0] == pytest.approx(last["y"] + 90)     # 0.9
    assert a.notes(80) == []


def test_interpolation_and_clamp_note(monkeypatch, model_dir):
    from app.adapters.tirex import TiRex2Adapter
    from app.data import to_frame

    calls = []
    install_fakes(monkeypatch, [round(0.1 * i, 1) for i in range(1, 10)], calls)
    a = TiRex2Adapter("t", model_dir, device="cpu")
    df = to_frame(make_records(1, n=60))
    out = a.predict(df, h=3, freq="W-MON", level=60, season_length=52)  # alpha=0.2 -> exakt Quantil 0.2/0.8
    last = df.iloc[-1]["y"]
    assert out["lo"].iloc[0] == pytest.approx(last + 20) and out["hi"].iloc[0] == pytest.approx(last + 80)
    out = a.predict(df, h=3, freq="W-MON", level=95, season_length=52)  # alpha=0.025 -> auf 0.1 begrenzt
    assert out["lo"].iloc[0] == pytest.approx(last + 10)
    assert a.notes(95) and "begrenzt" in a.notes(95)[0]


def test_registry_and_full_pipeline(monkeypatch, model_dir):
    import app.registry as reg
    from app import core

    install_fakes(monkeypatch, [round(0.1 * i, 1) for i in range(1, 10)], [])
    reg.get_adapter.cache_clear()
    monkeypatch.setitem(reg.load_config()["models"], "tirex_2", {"type": "tirex2", "path": model_dir, "device": "cpu"})
    assert [m for m in reg.describe_models() if m["name"] == "tirex_2"][0]["available"]
    r = core.run_forecast(make_records(2), h=6, models=["tirex_2", "naive"], n_windows=2)
    assert "tirex_2" in r["backtest"] and r["errors"] == {}
    reg.get_adapter.cache_clear()


def test_not_available_is_skipped_in_auto(monkeypatch):
    import app.registry as reg

    reg.get_adapter.cache_clear()
    assert "tirex_2" not in reg.resolve_models(["auto"])  # /models/tirex-2 existiert hier nicht
    with pytest.raises(reg.UnknownModel, match="nicht verfügbar"):
        reg.resolve_models(["tirex_2"])
    reg.get_adapter.cache_clear()
