"""Timestral-Adapter ohne echtes torch/Gewichte: Fake-Module prüfen nur den Datenfluss
(Eingabeform (B, 1, T), Auffüllen ungleich langer Reihen mit NaN, Quantilauswahl, Kovariatenformen,
Fähigkeiten aus config.json, Registry). Das echte Modell ist hier NICHT getestet – siehe live/live_timestral.py."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from helpers import make_records
from test_covariates import promo_records, _future_df

LEVELS = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def install_fakes(monkeypatch, calls, covariate_enabled=True, known=True, levels=LEVELS, max_ctx=None, with_torch=True):
    torch = types.ModuleType("torch")
    torch.tensor = lambda x: np.asarray(x)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class Cfg:
        def __init__(self):
            self.covariate_enabled = covariate_enabled
            self.supports_known_covariates = covariate_enabled and known
            self.quantile_knots = tuple(levels)

        @classmethod
        def load(cls, path):
            calls.append({"config": str(path)})
            return cls()

    class FakeForecaster:
        quantile_levels = tuple(levels)
        max_context = max_ctx

        @classmethod
        def from_pretrained(cls, path, device="cpu", local_files_only=False, **kw):
            calls.append({"load": path, "device": device, "local": local_files_only})
            return cls()

        def forecast(self, context, prediction_length, past_covariates=None, known_covariates=None, symmetric_averaging=True, **kw):
            calls.append(
                {
                    "ctx": np.array(context),
                    "h": prediction_length,
                    "past": None if past_covariates is None else np.array(past_covariates),
                    "known": None if known_covariates is None else np.array(known_covariates),
                    "sym": symmetric_averaging,
                }
            )
            b = context.shape[0]
            out = np.zeros((b, 1, prediction_length, len(levels)))
            for i in range(b):
                last = float(context[i, 0][~np.isnan(context[i, 0])][-1])
                # Wert = letzter Istwert + 100 * Quantilniveau -> Quantile sind im Ergebnis ablesbar
                for qi, lv in enumerate(levels):
                    out[i, 0, :, qi] = last + 100 * lv
            return SimpleNamespace(quantiles=out, quantile_levels=tuple(levels))

    pkg = types.ModuleType("timestral")
    pkg.TimestralForecaster = FakeForecaster
    cfgmod = types.ModuleType("timestral.config")
    cfgmod.TimestralConfig = Cfg
    pkg.config = cfgmod
    if with_torch:  # statsforecast braucht das echte torch; der Registry-Test lässt die Attrappe weg
        monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "timestral", pkg)
    monkeypatch.setitem(sys.modules, "timestral.config", cfgmod)


@pytest.fixture()
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"0")
    return str(tmp_path)


def test_available_checks_files(tmp_path, model_dir, monkeypatch):
    from app.adapters.timestral import TimestralAdapter

    install_fakes(monkeypatch, [])
    assert TimestralAdapter("t", model_dir).available() == (True, "")
    ok, why = TimestralAdapter("t", str(tmp_path / "nix")).available()
    assert not ok and "config.json" in why
    monkeypatch.delitem(sys.modules, "timestral")
    ok, why = TimestralAdapter("t", model_dir).available()
    assert not ok and "timestral nicht installiert" in why


def test_dataflow_quantile_pick_and_batching(monkeypatch, model_dir):
    from app.adapters.timestral import TimestralAdapter
    from app.data import to_frame

    calls = []
    install_fakes(monkeypatch, calls)
    a = TimestralAdapter("t", model_dir, device="cpu", batch_size=2, symmetric_averaging=False)
    df = to_frame(make_records(3, n=60))
    out = a.predict(df, h=5, freq="W-MON", level=80, season_length=52)
    assert list(out.columns) == ["unique_id", "ds", "yhat", "lo", "hi"] and len(out) == 15
    assert list(out["unique_id"].unique()) == ["s0", "s1", "s2"]  # Reihenfolge der Eingabe bleibt
    load = next(c for c in calls if "load" in c)
    assert load == {"load": model_dir, "device": "cpu", "local": True}
    fcs = [c for c in calls if "ctx" in c]
    assert [c["ctx"].shape for c in fcs] == [(2, 1, 60), (1, 1, 60)]  # batch_size=2 -> zwei Aufrufe
    assert all(c["sym"] is False and c["past"] is None and c["known"] is None and c["h"] == 5 for c in fcs)
    s0 = out[out.unique_id == "s0"]
    last = df[df.unique_id == "s0"].iloc[-1]
    assert s0["ds"].iloc[0] == last["ds"] + pd.Timedelta(days=7)
    assert s0["yhat"].iloc[0] == pytest.approx(last["y"] + 50)   # Median 0.5
    assert s0["lo"].iloc[0] == pytest.approx(last["y"] + 10)     # 0.1
    assert s0["hi"].iloc[0] == pytest.approx(last["y"] + 90)     # 0.9


def test_unequal_lengths_are_left_padded_with_nan(monkeypatch, model_dir):
    from app.adapters.timestral import TimestralAdapter
    from app.data import to_frame

    calls = []
    install_fakes(monkeypatch, calls)
    short, long_ = make_records(1, n=30), make_records(2, n=60)
    short = [{**r, "unique_id": "kurz"} for r in short]
    df = to_frame(short + [{**r, "unique_id": "lang"} for r in long_[:60]])
    a = TimestralAdapter("t", model_dir, device="cpu", batch_size=8)
    out = a.predict(df, h=3, freq="W-MON", level=80, season_length=52)
    assert list(out["unique_id"].unique()) == ["kurz", "lang"]
    ctx = next(c["ctx"] for c in calls if "ctx" in c)
    assert ctx.shape == (2, 1, 60)
    assert np.isnan(ctx[0, 0, :30]).all() and not np.isnan(ctx[0, 0, 30:]).any()  # kurze Reihe: links aufgefüllt
    assert not np.isnan(ctx[1, 0]).any()
    # Ergebnis gehört trotz Sortierung nach Länge zur richtigen Reihe
    last_kurz = df[df.unique_id == "kurz"]["y"].iloc[-1]
    assert out[out.unique_id == "kurz"]["yhat"].iloc[0] == pytest.approx(last_kurz + 50)


def test_context_is_trimmed_to_max_context(monkeypatch, model_dir):
    from app.adapters.timestral import TimestralAdapter
    from app.data import to_frame

    calls = []
    install_fakes(monkeypatch, calls, max_ctx=40)
    df = to_frame(make_records(1, n=100))
    TimestralAdapter("t", model_dir, device="cpu").predict(df, h=4, freq="W-MON", level=80, season_length=52)
    ctx = next(c["ctx"] for c in calls if "ctx" in c)
    assert ctx.shape == (1, 1, 40)
    np.testing.assert_allclose(ctx[0, 0], df["y"].to_numpy(dtype=np.float32)[-40:])


def test_covariate_tensors(monkeypatch, model_dir):
    from app.adapters.base import CovSpec
    from app.adapters.timestral import TimestralAdapter
    from app.data import to_frame

    calls = []
    install_fakes(monkeypatch, calls)
    rec, fut = promo_records(2, n=40)
    df = to_frame(rec, ["promo", "temp"])
    a = TimestralAdapter("t", model_dir, device="cpu")
    spec = CovSpec(("promo", "temp"), ("promo",))
    assert a.supports_covariates and a.can_use(spec) and a.cov_notes(spec) == []
    a.predict(df, 6, "MS", 80, 12, cov=spec, future_df=_future_df(fut)[["unique_id", "ds", "promo"]])
    c = next(c for c in calls if "ctx" in c)
    assert c["ctx"].shape == (2, 1, 40)
    assert c["past"].shape == (2, 1, 40)    # nur temp (nur Vergangenheit bekannt)
    assert c["known"].shape == (2, 1, 46)   # promo über Historie + Horizont
    s0, f0 = pd.DataFrame(rec), pd.DataFrame(fut)
    expect = np.concatenate([s0[s0.unique_id == "s0"]["promo"].to_numpy(), f0[f0.unique_id == "s0"]["promo"].to_numpy()])
    np.testing.assert_allclose(c["known"][0, 0], expect.astype(np.float32))
    np.testing.assert_allclose(c["past"][0, 0], s0[s0.unique_id == "s0"]["temp"].to_numpy(dtype=np.float32))
    # ohne Kovariaten: reine Zielreihe
    calls.clear()
    a.predict(df, 6, "MS", 80, 12)
    c = next(c for c in calls if "ctx" in c)
    assert c["past"] is None and c["known"] is None


def test_checkpoint_without_known_future_uses_them_as_past(monkeypatch, model_dir):
    from app.adapters.base import CovSpec
    from app.adapters.timestral import TimestralAdapter
    from app.data import to_frame

    calls = []
    install_fakes(monkeypatch, calls, known=False)
    rec, fut = promo_records(1, n=40)
    df = to_frame(rec, ["promo", "temp"])
    a = TimestralAdapter("t", model_dir, device="cpu")
    spec = CovSpec(("promo", "temp"), ("promo",))
    assert a.supports_covariates and "promo" in a.cov_notes(spec)[0]
    a.predict(df, 6, "MS", 80, 12, cov=spec, future_df=_future_df(fut)[["unique_id", "ds", "promo"]])
    c = next(c for c in calls if "ctx" in c)
    assert c["known"] is None and c["past"].shape == (1, 2, 40)


def test_checkpoint_without_covariates(monkeypatch, model_dir):
    from app.adapters.base import CovSpec
    from app.adapters.timestral import TimestralAdapter

    install_fakes(monkeypatch, [], covariate_enabled=False)
    a = TimestralAdapter("t", model_dir)
    assert not a.supports_covariates and not a.can_use(CovSpec(("x",), ("x",)))


def test_interval_outside_native_quantiles_is_noted(monkeypatch, model_dir):
    from app.adapters.timestral import TimestralAdapter

    install_fakes(monkeypatch, [])
    a = TimestralAdapter("t", model_dir)
    assert a.notes(80) == []
    assert "außerhalb der Modellquantile" in a.notes(99)[0]   # alpha 0,005 < 0,01


def test_registry_builds_timestral_and_auto_plus_extra(monkeypatch, tmp_path):
    import app.registry as reg
    install_fakes(monkeypatch, [], with_torch=False)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"0")
    cfg = {
        "baseline": "seasonal_naive",
        "default_candidates": ["seasonal_naive", "theta"],
        "intermittent_candidates": [],
        "models": {
            "seasonal_naive": {"type": "stats", "class": "SeasonalNaive"},
            "theta": {"type": "stats", "class": "AutoTheta"},
            "timestral_28m": {"type": "timestral", "path": str(tmp_path), "device": "cpu", "batch_size": 4, "symmetric_averaging": False},
            "timestral_fehlt": {"type": "timestral", "path": str(tmp_path / "nix")},
        },
    }
    monkeypatch.setattr(reg, "load_config", lambda: cfg)
    reg.get_adapter.cache_clear()
    try:
        ad = reg.get_adapter("timestral_28m")
        assert ad.kind == "timestral" and ad.batch_size == 4 and ad.symmetric_averaging is False
        assert reg.resolve_models(["auto"]) == ["seasonal_naive", "theta"]
        assert reg.resolve_models(["auto", "timestral_28m"]) == ["seasonal_naive", "theta", "timestral_28m"]
        assert reg.resolve_models(["timestral_28m"]) == ["seasonal_naive", "timestral_28m"]
        with pytest.raises(reg.UnknownModel, match="nicht verfügbar"):
            reg.resolve_models(["auto", "timestral_fehlt"])
        rows = {r["name"]: r for r in reg.describe_models()}
        assert rows["timestral_28m"]["available"] and rows["timestral_28m"]["supports_covariates"]
        assert not rows["timestral_28m"]["default_candidate"] and not rows["timestral_fehlt"]["available"]
    finally:
        reg.get_adapter.cache_clear()


def test_models_yaml_has_timestral_entries_not_in_defaults():
    import yaml

    with open("models.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for n in ("timestral_28m", "timestral_209m"):
        assert cfg["models"][n]["type"] == "timestral" and cfg["models"][n]["device"] == "cpu"
        assert n not in cfg["default_candidates"]
