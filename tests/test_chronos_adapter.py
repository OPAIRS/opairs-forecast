"""Chronos-Adapter ohne echtes torch/Gewichte: Fake-Module prüfen nur Datenfluss
(Quantil-Reihenfolge, Zukunftsdaten, Batching, Chronos-2-Listenformat). Das echte Modell ist hier NICHT getestet."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from helpers import make_records


class T:  # Mini-Tensor
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def __getitem__(self, i): return T(self.a[i])


def install_fakes(monkeypatch, chronos2=False, calls=None):
    torch = types.ModuleType("torch")
    torch.tensor = lambda x: np.asarray(x)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class Pipe:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            if calls is not None: calls.append((len(inputs), quantile_levels))
            outs = []
            for ctx in inputs:
                last = ctx[-1]
                q = np.stack([np.full(prediction_length, last + d) for d in (-5, 0, 5)], axis=-1)  # (h, 3)
                outs.append(q)
            if chronos2:
                return [T(o[None]) for o in outs], [T(o[None, :, 1]) for o in outs]
            return T(np.stack(outs)), T(np.stack([o[:, 1] for o in outs]))

    class Base:
        @staticmethod
        def from_pretrained(path, device_map=None):
            return Pipe()

    ch = types.ModuleType("chronos"); ch.BaseChronosPipeline = Base
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chronos", ch)


@pytest.mark.parametrize("chronos2", [False, True])
def test_adapter_dataflow(monkeypatch, tmp_path, chronos2):
    calls = []
    install_fakes(monkeypatch, chronos2, calls)
    from app.adapters.chronos import ChronosAdapter
    from app.data import to_frame

    a = ChronosAdapter("fake", str(tmp_path), batch_size=2, family="chronos2" if chronos2 else "bolt")
    assert a.available()[0]
    df = to_frame(make_records(3, n=60))
    out = a.predict(df, h=5, freq="W-MON", level=80, season_length=52)
    assert list(out.columns) == ["unique_id", "ds", "yhat", "lo", "hi"] and len(out) == 15
    assert [c[0] for c in calls] == [2, 1]  # Batching
    assert calls[0][1] == [0.1, 0.5, 0.9]
    s0 = out[out.unique_id == "s0"]
    last = df[df.unique_id == "s0"].iloc[-1]
    assert s0["ds"].iloc[0] == last["ds"] + pd.Timedelta(days=7)
    assert (s0["lo"] < s0["yhat"]).all() and (s0["yhat"] < s0["hi"]).all()
    assert s0["yhat"].iloc[0] == pytest.approx(last["y"])


def test_adapter_in_full_pipeline_and_notes(monkeypatch, tmp_path):
    install_fakes(monkeypatch)
    import app.registry as reg
    from app import core
    from app.adapters.chronos import ChronosAdapter

    reg.get_adapter.cache_clear()
    monkeypatch.setitem(reg.load_config()["models"], "fake_bolt", {"type": "chronos", "family": "bolt", "path": str(tmp_path)})
    r = core.run_forecast(make_records(2), h=6, models=["fake_bolt", "naive"], n_windows=2, level=95)
    assert "fake_bolt" in r["backtest"] and r["errors"] == {}
    assert ChronosAdapter("x", ".", family="bolt").notes(95) and not ChronosAdapter("x", ".", family="chronos2").notes(99)
    reg.get_adapter.cache_clear()
