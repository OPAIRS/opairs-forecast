#!/usr/bin/env python3
"""Rauchtest für den Timestral-Adapter mit dem ECHTEN Paket `timestral` und PyTorch (kein Netzwerk nötig).

Die Unit-Tests (tests/test_timestral_adapter.py) nutzen Fake-Module. Dieses Skript prüft die Schnittstelle gegen den
echten Code des Pakets: Formen, NaN-Auffüllen ungleich langer Reihen, Kontextkürzung, Quantile, Kovariaten, blockweise
Prognose bei langem Horizont.

  # ohne Gewichte: baut einen winzigen, ZUFÄLLIG initialisierten Checkpoint (Ergebnisse sind Unsinn, es zählt der Datenfluss)
  docker exec opairs-forecast-api python live/live_timestral.py
  # mit echten Gewichten (Verzeichnis mit config.json + model.safetensors):
  docker exec opairs-forecast-api python live/live_timestral.py --path /models/timestral-28m

Das Skript ersetzt KEINEN Genauigkeitstest: ob Timestral auf euren Daten gut ist, zeigt live/benchmark.py
(--models ...,timestral_28m) und live/live_covariates.py (--models timestral_28m).
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.adapters.base import CovSpec  # noqa: E402
from app.adapters.timestral import TimestralAdapter  # noqa: E402


def build_tiny(path: str, known: bool) -> None:
    import torch
    from safetensors.torch import save_file
    from timestral.config import TimestralConfig
    from timestral.model import TimestralModel

    levels = tuple(round(x, 2) for x in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    cfg = TimestralConfig(
        patch_size=8, d_model=32, num_heads=4, mlp_hidden_dim=64,
        block_schedule=("time_variate",) * 3 if known else ("time",) * 3,
        quantile_knots=levels, block_size=4, context_length=128, covariate_enabled=True,
    )
    torch.manual_seed(0)
    model = TimestralModel(cfg)
    os.makedirs(path, exist_ok=True)
    cfg.save(os.path.join(path, "config.json"))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(path, "model.safetensors"))


def series(uid, n, rng, start="2022-01-03", freq="W-MON", cov=False):
    ds = pd.date_range(start, periods=n, freq=freq)
    y = 100 + 0.3 * np.arange(n) + 10 * np.sin(2 * np.pi * np.arange(n) / 52) + rng.normal(0, 2, n)
    d = pd.DataFrame({"unique_id": uid, "ds": ds, "y": y.astype("float32")})
    if cov:
        d["promo"] = (rng.random(n) < 0.3).astype("float32")
        d["temp"] = rng.normal(15, 3, n).astype("float32")
    return d


def check(out, uids, h):
    assert list(out["unique_id"].unique()) == uids and len(out) == h * len(uids), "falsche Zeilenzahl/Reihenfolge"
    assert out[["yhat", "lo", "hi"]].notna().all().all(), "NaN im Ergebnis"
    assert (out["lo"] <= out["yhat"]).all() and (out["yhat"] <= out["hi"]).all(), "Quantile nicht geordnet"


def run(path: str, expect_known: bool | None) -> None:
    rng = np.random.default_rng(1)
    ad = TimestralAdapter("timestral", path, device="cpu", batch_size=2)
    ok, why = ad.available()
    assert ok, why
    print(f"Checkpoint {path}: Kovariaten={ad.supports_covariates}, Known-Future={ad.supports_known}")
    if expect_known is not None:
        assert ad.supports_known == expect_known

    df = pd.concat([series("a", 200, rng), series("b", 90, rng), series("c", 40, rng)], ignore_index=True)
    t0 = time.time()
    out = ad.predict(df, 6, "W-MON", 80, 52)
    check(out, ["a", "b", "c"], 6)
    print(f"1) ohne Kovariaten, ungleich lange Reihen, mehrere Batches: OK ({time.time() - t0:.1f} s)")

    if ad.supports_covariates:
        h, n = 6, 120
        hist = pd.concat([series("a", n, rng, cov=True), series("b", n, rng, cov=True)], ignore_index=True)
        fut = pd.concat(
            [
                pd.DataFrame({
                    "unique_id": u,
                    "ds": pd.date_range(hist[hist.unique_id == u]["ds"].iloc[-1], periods=h + 1, freq="W-MON")[1:],
                    "promo": (rng.random(h) < 0.3).astype("float32"),
                })
                for u in ("a", "b")
            ],
            ignore_index=True,
        )
        spec = CovSpec(("promo", "temp"), ("promo",))
        with_cov = ad.predict(hist, h, "W-MON", 80, 52, cov=spec, future_df=fut)
        check(with_cov, ["a", "b"], h)
        print(f"2) mit Kovariaten (promo zukunftsbekannt, temp nur Vergangenheit): OK {ad.cov_notes(spec) or ''}")

    long_h = ad.predict(series("m", 60, rng, start="2021-01-01", freq="MS"), 40, "MS", 80, 12)
    check(long_h, ["m"], 40)
    print("3) langer Horizont (mehrere Blöcke): OK")
    print("ALLES OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="", help="Verzeichnis mit config.json + model.safetensors (leer = Zufallsmodell)")
    a = ap.parse_args()
    if a.path:
        run(a.path, None)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for known in (True, False):
            p = os.path.join(tmp, "known" if known else "noknown")
            build_tiny(p, known)
            run(p, known)


if __name__ == "__main__":
    main()
