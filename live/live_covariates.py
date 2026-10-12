#!/usr/bin/env python3
"""Live-Test für Kovariaten mit den echten Modellen (Chronos-2, TiRex-2, AutoARIMA) über LiteLLM (nur Standardbibliothek).

Die Unit-Tests prüfen für Chronos-2 und TiRex-2 nur den Datenfluss mit Fake-Modulen. Dieses Skript zeigt, ob die echten
Modelle die Kovariaten sinnvoll nutzen: Synthetische Monatsreihen, deren Wert von einer zukunftsbekannten Aktion
(promo) und einer nur in der Vergangenheit bekannten Größe (temp) abhängt. Erwartung: `<modell>@cov` schlägt `<modell>`
deutlich, wenn die Modelle die Kovariaten wirklich verarbeiten.

  LITELLM_KEY=$(grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2-) python3 live/live_covariates.py
  ... --models chronos_2,tirex_2,auto_arima   (Standard)   --url http://localhost:4000
chronos_2 braucht die Gewichte in /models/chronos-2 (siehe README), sonst bleibt es weg.
"""
import argparse
import json
import math
import os
import random
import sys
import urllib.request
from datetime import date


def make(n_series, n, h, seed):
    rnd = random.Random(seed)
    rec, fut = [], []
    for s in range(n_series):
        level, trend = rnd.uniform(80, 200), rnd.uniform(0, 0.5)
        for i in range(n + h):
            promo = 1.0 if rnd.random() < 0.3 else 0.0
            temp = 15 + 8 * math.sin(2 * math.pi * i / 12) + rnd.gauss(0, 1)
            y = level + trend * i + 0.8 * promo * level * 0.3 + 1.5 * (temp - 15) + rnd.gauss(0, 2)
            d = date(2016 + i // 12, i % 12 + 1, 1).isoformat()
            row = {"unique_id": f"s{s:02d}", "ds": d, "promo": promo}
            if i < n:
                rec.append({**row, "y": round(y, 3), "temp": round(temp, 3)})
            else:
                fut.append(row)
    return rec, fut


def post(url, key, path, body, timeout=1800):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode() if body is not None else None,
        method="POST" if body is not None else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4000")
    ap.add_argument("--series", type=int, default=6)
    ap.add_argument("--points", type=int, default=84)
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--models", default="chronos_2,tirex_2,auto_arima")
    a = ap.parse_args()
    key = os.environ.get("LITELLM_KEY", "")
    url = a.url.rstrip("/")
    raw = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/mcp/server", headers={"Authorization": f"Bearer {key}"})).read())
    items = raw if isinstance(raw, list) else raw.get("servers") or raw.get("data") or [raw]
    sid = next((x.get("server_id") for x in items if isinstance(x, dict) and "opairs_forecast" in json.dumps(x)), None)
    if not sid:
        sys.exit("server_id nicht gefunden")

    available = {m["name"] for m in json.loads("".join(c.get("text", "") for c in post(url, key, "/mcp-rest/tools/call", {"server_id": sid, "name": "list_models", "arguments": {}}).get("content", [])))["models"] if m["available"]}
    models = [m for m in a.models.split(",") if m in available]
    print("Modelle:", ", ".join(models) or "(keine verfügbar)", "| nicht verfügbar:", ", ".join(m for m in a.models.split(",") if m not in available) or "-")
    if not models:
        sys.exit(1)
    rec, fut = make(a.series, a.points, a.h, 7)
    args = {"series": rec, "h": a.h, "models": models, "n_windows": a.windows, "ensemble": False,
            "covariates": ["promo", "temp"], "future_covariates": fut}
    r = post(url, key, "/mcp-rest/tools/call", {"server_id": sid, "name": "backtest", "arguments": args})
    text = "".join(c.get("text", "") for c in r.get("content", []))
    if r.get("isError"):
        sys.exit("FEHLER: " + text[:400])
    res = json.loads(text)
    print(f"\nKovariaten: {res.get('covariates')}\n")
    bt = res["backtest"]
    print(f"{'Modell':<22} {'relative MAE':>13}")
    for n, s in sorted(bt.items(), key=lambda kv: kv[1]["relative_mae"]):
        print(f"{n:<22} {s['relative_mae']:>13.3f}")
    print("\nOhne vs. mit Kovariaten (kleiner ist besser):")
    for m in models:
        if m in bt and f"{m}@cov" in bt:
            w, c = bt[m]["relative_mae"], bt[f"{m}@cov"]["relative_mae"]
            print(f"  {m:<14} {w:.3f} -> {c:.3f}   {'Kovariaten helfen' if c < w * 0.9 else 'kaum Wirkung: Adapter/Datenfluss prüfen'}")
    for w in res.get("warnings", []):
        print("Hinweis:", w)
    if res.get("errors"):
        print("Fehler je Modell:", res["errors"])


if __name__ == "__main__":
    main()
