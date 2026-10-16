#!/usr/bin/env python3
"""Prüft, ob der Timestral-Adapter bekannte Zukunfts-Kovariaten richtig ausrichtet und nutzt.

Synthetische Monatsreihen, bei denen y direkt von einer Kovariate abhängt (y = Grundniveau + Saison + 40 * promo + Rauschen).
Die letzten 12 Monate sind der Test. Ein Modell, das die Kovariate wirklich liest, muss mit ihr deutlich besser sein als ohne.
Zum Gegenprobe-Lauf wird die Kovariate um einen Schritt verschoben: dann darf der Vorteil weitgehend verschwinden.

  # echte Gewichte (empfohlen, so wie auf dem Server):
  docker run --rm -v "$PWD":/work -w /work -v /opt/opairs-assets/llm-models/timeseries:/models:ro \
      opairs-forecast:foundation python live/live_timestral_cov_check.py --path /models/timestral-28m
  # ohne --path: winziges Zufallsmodell, prüft nur, dass das Skript läuft (die Zahlen sind dann Unsinn)

Ergebnis: MAE ohne Kovariate / mit Kovariate / mit um einen Schritt verschobener Kovariate, je Checkpoint.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.adapters.base import CovSpec  # noqa: E402
from app.adapters.timestral import TimestralAdapter  # noqa: E402

H = 12
N_HIST = 60


def make(seed: int, n_series: int = 4):
    rng = np.random.default_rng(seed)
    hist, fut, truth = [], [], []
    for k in range(n_series):
        n = N_HIST + H
        ds = pd.date_range("2021-01-01", periods=n, freq="MS")
        promo = (rng.random(n) < 0.3).astype("float32")
        y = 100 + 10 * k + 10 * np.sin(2 * np.pi * np.arange(n) / 12) + 40 * promo + rng.normal(0, 2, n)
        uid = f"s{k}"
        hist.append(pd.DataFrame({"unique_id": uid, "ds": ds[:N_HIST], "y": y[:N_HIST].astype("float32"), "promo": promo[:N_HIST]}))
        fut.append(pd.DataFrame({"unique_id": uid, "ds": ds[N_HIST:], "promo": promo[N_HIST:]}))
        truth.append(pd.DataFrame({"unique_id": uid, "ds": ds[N_HIST:], "y": y[N_HIST:]}))
    return pd.concat(hist, ignore_index=True), pd.concat(fut, ignore_index=True), pd.concat(truth, ignore_index=True)


def mae(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    m = pred.merge(truth, on=["unique_id", "ds"])
    assert len(m) == len(truth), "Zeilenzahl passt nicht"
    return float((m["yhat"] - m["y"]).abs().mean())


def run(path: str) -> None:
    ad = TimestralAdapter("timestral", path, device="cpu")
    ok, why = ad.available()
    assert ok, why
    print(f"Checkpoint {path}: Kovariaten={ad.supports_covariates}, Known-Future={ad.supports_known}")
    if not (ad.supports_covariates and ad.supports_known):
        print("Dieser Checkpoint kann keine bekannten Zukunfts-Kovariaten: Test entfällt.")
        return
    res = {"ohne": [], "mit": [], "verschoben": []}
    for seed in range(5):
        hist, fut, truth = make(seed)
        spec = CovSpec(("promo",), ("promo",))
        res["ohne"].append(mae(ad.predict(hist.drop(columns="promo"), H, "MS", 80, 12), truth))
        res["mit"].append(mae(ad.predict(hist, H, "MS", 80, 12, cov=spec, future_df=fut), truth))
        sh = fut.copy()
        sh["promo"] = sh.groupby("unique_id")["promo"].shift(1).fillna(0).astype("float32")  # um einen Schritt nach hinten
        res["verschoben"].append(mae(ad.predict(hist, H, "MS", 80, 12, cov=spec, future_df=sh), truth))
    o, m, s = (float(np.mean(res[k])) for k in ("ohne", "mit", "verschoben"))
    print(f"MAE (Mittel über 5 Datensätze):  ohne Kovariate {o:6.2f}   mit {m:6.2f}   verschoben {s:6.2f}")
    if m < 0.6 * o and s > m:
        print("ERGEBNIS: Kovariate wird gelesen und richtig ausgerichtet (mit << ohne, verschoben schlechter als mit).")
    elif m >= o:
        print("ERGEBNIS: Kovariate hilft NICHT (mit >= ohne). Entweder liest das Modell sie nicht wie erwartet oder die Ausrichtung im Adapter stimmt nicht.")
    else:
        print("ERGEBNIS: unklar (Vorteil gering oder verschobene Kovariate nicht schlechter). Bitte Zahlen an Entwicklung weitergeben.")


def tiny(tmp: str) -> str:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from live_timestral import build_tiny

    p = os.path.join(tmp, "known")
    build_tiny(p, True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="", help="Verzeichnis mit config.json + model.safetensors (leer = Zufallsmodell)")
    a = ap.parse_args()
    if a.path:
        run(a.path)
        return
    with tempfile.TemporaryDirectory() as tmp:
        run(tiny(tmp))


if __name__ == "__main__":
    main()
