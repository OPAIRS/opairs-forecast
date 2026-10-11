#!/usr/bin/env python3
"""Einfacher Last- und Hardwaretest über LiteLLM (nur Standardbibliothek).

Schickt R gleichartige Prognoseanfragen (synthetische Monatsreihen, alle Modelle per 'auto' mit Backtest) mit P parallelen
Clients und misst Gesamtdauer, Latenz je Anfrage und Durchsatz. Beispiel:

  LITELLM_KEY=$(grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2-) python3 loadtest.py --series 20 --parallel 1,2,4

Parallelität wirkt nur, wenn FORECAST_MAX_CONCURRENCY im MCP-Container >= P ist. Währenddessen in einem zweiten Terminal:
  docker stats --no-stream opairs-forecast-mcp
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date


def make_series(n_series, points, seed):
    rnd = random.Random(seed)
    out = []
    for s in range(n_series):
        level, trend, amp = rnd.uniform(50, 500), rnd.uniform(-0.2, 1.0), rnd.uniform(5, 60)
        for i in range(points):
            y = level + trend * i + amp * math.sin(2 * math.pi * i / 12) + rnd.gauss(0, amp * 0.3)
            y_m = date(2015 + (i // 12), i % 12 + 1, 1).isoformat()
            out.append({"unique_id": f"s{s:03d}", "ds": y_m, "y": round(y, 3)})
    return out


def post(url, key, path, body, timeout=1800):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode() if body is not None else None,
        method="POST" if body is not None else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4000")
    ap.add_argument("--series", type=int, default=20, help="Serien je Anfrage")
    ap.add_argument("--points", type=int, default=120, help="Monate je Serie")
    ap.add_argument("--h", type=int, default=12)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--parallel", default="1,2,4", help="Liste paralleler Clients, z. B. 1,2,4")
    ap.add_argument("--requests", type=int, default=0, help="Anfragen je Stufe (Standard: 2 x größte Parallelität, mind. 4)")
    ap.add_argument("--scale", default="", help="Skalierungstest: Liste von Serienzahlen, je eine Anfrage, z. B. 50,200,500")
    ap.add_argument("--models", default="", help="optional, z. B. tirex_2 oder seasonal_naive,theta")
    a = ap.parse_args()
    key = os.environ.get("LITELLM_KEY", "")
    url = a.url.rstrip("/")

    raw = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/mcp/server", headers={"Authorization": f"Bearer {key}"})).read())
    items = raw if isinstance(raw, list) else raw.get("servers") or raw.get("data") or [raw]
    sid = next((x.get("server_id") for x in items if isinstance(x, dict) and "opairs_forecast" in json.dumps(x)), None)
    if not sid:
        sys.exit("server_id nicht gefunden")

    if a.scale:
        models = [m for m in a.models.split(",") if m] or ["auto"]
        print("Skalierungstest: je eine Anfrage, Modelle=" + ",".join(models) + f", {a.points} Monate, h={a.h}, Fenster={a.windows}")
        print("Aufwärmen …", flush=True)
        post(url, key, "/mcp-rest/tools/call", {"server_id": sid, "name": "forecast", "arguments": {"h": a.h, "models": models, "n_windows": a.windows, "series": make_series(3, a.points, 1)}})
        print(f"\n{'Serien':>7} {'Sekunden':>9} {'Serien/min':>11} {'Last(1min)':>11} {'Ergebnis':>10}")
        for n in [int(x) for x in a.scale.split(",") if x]:
            body = {"server_id": sid, "name": "forecast", "arguments": {"h": a.h, "models": models, "n_windows": a.windows, "series": make_series(n, a.points, n)}}
            t0 = time.time()
            try:
                r = post(url, key, "/mcp-rest/tools/call", body)
                text = "".join(c.get("text", "") for c in r.get("content", []))
                ok = "ok" if not r.get("isError") else "FEHLER: " + text[:80]
                if ok == "ok":
                    ok = "ok/" + str(json.loads(text).get("selected_model", "?"))[:22]
            except Exception as e:
                ok = "FEHLER: " + str(e)[:80]
            dt = time.time() - t0
            try:
                load = open("/proc/loadavg").read().split()[0]
            except Exception:
                load = "-"
            print(f"{n:>7} {dt:>9.1f} {60 * n / dt:>11.1f} {load:>11} {ok:>10}", flush=True)
        return

    levels = [int(x) for x in a.parallel.split(",") if x]
    nreq = a.requests or max(4, 2 * max(levels))
    models = [m for m in a.models.split(",") if m] or ["auto"]
    args_base = {"h": a.h, "models": models, "n_windows": a.windows}

    def one(i):
        args = dict(args_base, series=make_series(a.series, a.points, seed=i))
        t0 = time.time()
        r = post(url, key, "/mcp-rest/tools/call", {"server_id": sid, "name": "forecast", "arguments": args})
        dt = time.time() - t0
        if r.get("isError"):
            return dt, "FEHLER: " + "".join(c.get("text", "") for c in r.get("content", []))[:120]
        return dt, None

    print("Wärme den Dienst auf (erster Aufruf lädt die Modelle) …", flush=True)
    t0 = time.time()
    _, err = one(999)
    print(f"  Aufwärm-Anfrage: {time.time() - t0:.1f} s" + (f"  {err}" if err else ""))
    print(f"\n{a.series} Serien x {a.points} Monate, h={a.h}, Fenster={a.windows}, Modelle={','.join(models)}, {nreq} Anfragen je Stufe\n")
    print(f"{'Parallel':>8} {'Gesamt s':>9} {'Ø Latenz s':>11} {'max s':>7} {'Anfr./min':>10} {'Fehler':>7}")
    for p in levels:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=p) as ex:
            res = list(ex.map(one, range(nreq)))
        wall = time.time() - t0
        lat = [r[0] for r in res if r[1] is None]
        errs = [r[1] for r in res if r[1]]
        print(
            f"{p:>8} {wall:>9.1f} {statistics.mean(lat) if lat else float('nan'):>11.1f} {max(lat) if lat else float('nan'):>7.1f} "
            f"{60 * len(lat) / wall:>10.1f} {len(errs):>7}"
        )
        if errs:
            print("   ", errs[0])


if __name__ == "__main__":
    main()
