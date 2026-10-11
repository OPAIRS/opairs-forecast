#!/usr/bin/env python3
"""Rückblick-Benchmark über öffentliche Zeitreihen (nur Standardbibliothek).

Ablauf je Datensatz: die letzten h Punkte werden zurückgehalten, der Service prognostiziert nur mit den
übrigen Daten, danach wird gegen die echten Werte verglichen.
  * 'auto'  = der Service wählt das Modell selbst per Backtest (so nutzt es die Pipe)
  * Ens2 = Mittelwert aus Chronos und TiRex-2, Top2 = Mittelwert der zwei besten Backtest-Modelle (falls zwei die Baseline schlagen)
  * je Modell einzeln (ohne Backtest) = wie gut wäre dieses Modell gewesen
Kennzahl: relative MAE auf dem Rückhalte-Zeitraum, geteilt durch die Baseline (seasonal_naive); < 1 ist besser.

Aufruf (Schlüssel nicht ausgeben):
  LITELLM_KEY=$(grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2-) python3 benchmark.py
Optionen: --url http://localhost:4000  --only airline,autos  --windows 4  --models a,b,c
Die Datensätze werden von GitHub geladen (der Host braucht Internet, der Container nicht).
"""
import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
from collections import OrderedDict
from datetime import date, datetime, timedelta

JB = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/"
ETT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
CACHE = os.environ.get("BENCH_CACHE", "/tmp/fc-bench")


def fetch(url):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, url.replace("/", "_").replace(":", ""))
    if not os.path.exists(p):
        with urllib.request.urlopen(url, timeout=60) as r:
            open(p, "wb").write(r.read())
    return open(p, encoding="utf-8").read()


def rows(url):
    return list(csv.reader(io.StringIO(fetch(url))))[1:]


def monthly(url, year_fix=None):
    out = []
    for m, v in rows(url):
        if year_fix:
            m = year_fix(m)
        out.append((m + "-01", float(v)))
    return {"serie": out}


def daily(url, regular=False):
    pts = [(d, float(v)) for d, v in rows(url)]
    if regular:  # Quelle hat Lücken (Schalttage): fortlaufende Tage vergeben, das Datum ist für den Test nebensächlich
        d0 = date.fromisoformat(pts[0][0])
        pts = [((d0 + timedelta(days=i)).isoformat(), v) for i, (_, v) in enumerate(pts)]
    return {"serie": pts}


def ett_daily(cols):
    data = rows(ETT)
    head = list(csv.reader(io.StringIO(fetch(ETT))))[0]
    idx = {c: head.index(c) for c in cols}
    days = OrderedDict()
    for r in data:
        days.setdefault(r[0][:10], []).append(r)
    out = {}
    for c, i in idx.items():
        out[c] = [(d, sum(float(x[i]) for x in v) / len(v)) for d, v in days.items()]
    return out


def ett_hourly(cols, n):
    data = rows(ETT)[-n:]
    head = list(csv.reader(io.StringIO(fetch(ETT))))[0]
    return {c: [(r[0], float(r[head.index(c)])) for r in data] for c in cols}


def quarterly(url):
    out = []
    for _, t, v in rows(url):
        t = float(t)
        y = int(t + 1e-9)
        q = int(round((t - y) * 4))
        out.append((f"{y}-{1 + 3 * q:02d}-01", float(v)))
    return {"serie": out}


def weekly(url, start=date(1991, 2, 1)):
    return {"serie": [((start + timedelta(weeks=i)).isoformat(), float(v)) for i, (_, _, v) in enumerate(rows(url))]}


# name, loader, last_n (None = alle), h, beschreibung
SPECS = [
    ("airline", lambda: monthly(JB + "airline-passengers.csv"), None, 12, "Passagiere, monatlich"),
    ("autos", lambda: monthly(JB + "monthly-car-sales.csv"), None, 12, "Autoverkäufe, monatlich"),
    ("sonnenfl", lambda: monthly(JB + "monthly-sunspots.csv"), 400, 24, "Sonnenflecken, monatlich"),
    ("temp_mon", lambda: monthly(JB + "monthly-mean-temp.csv"), None, 12, "Mitteltemperatur, monatlich"),
    ("raub", lambda: monthly(JB + "monthly-robberies.csv"), None, 12, "Raubüberfälle, monatlich, verrauscht"),
    ("shampoo", lambda: monthly(JB + "shampoo.csv", lambda m: "200" + m), None, 6, "Shampoo, 36 Monate (kurz)"),
    ("geburten", lambda: daily(JB + "daily-total-female-births.csv"), None, 14, "Geburten, täglich, Rauschen"),
    ("temp_tag", lambda: daily(JB + "daily-min-temperatures.csv", regular=True), 1100, 28, "Tagestiefsttemperatur"),
    ("trafo_tag", lambda: ett_daily(["HUFL", "MUFL", "OT"]), 365, 14, "Trafo-Last/Temp., täglich, 3 Serien"),
    ("trafo_std", lambda: ett_hourly(["OT", "HUFL"], 1056), None, 48, "Trafo, stündlich, 2 Serien"),
    ("bier_q", lambda: quarterly(RD + "fpp2/ausbeer.csv"), 160, 8, "Bierproduktion, quartalsweise"),
    ("benzin_w", lambda: weekly(RD + "fpp2/gasoline.csv"), 520, 26, "US-Benzin, wöchentlich"),
]


class Client:
    def __init__(self, url, key, server_name="opairs_forecast"):
        self.url, self.key = url.rstrip("/"), key
        raw = self._req("GET", "/v1/mcp/server")
        items = raw if isinstance(raw, list) else raw.get("servers") or raw.get("data") or [raw]
        self.sid = next((x.get("server_id") for x in items if isinstance(x, dict) and server_name in json.dumps(x)), None)
        if not self.sid:
            sys.exit(f"server_id für {server_name} nicht gefunden: {json.dumps(raw)[:300]}")

    def _req(self, method, path, body=None):
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def call(self, name, arguments):
        j = self._req("POST", "/mcp-rest/tools/call", {"server_id": self.sid, "name": name, "arguments": arguments})
        text = "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text")
        if j.get("isError"):
            raise RuntimeError(text.split(":", 1)[-1].strip()[:200])
        return json.loads(text)


def mae_by_series(pred, actual):
    """pred: [{unique_id, ds, yhat}], actual: {uid: {ds: y}} -> {uid: MAE}"""
    err = {}
    for r in pred:
        a = actual.get(r["unique_id"], {}).get(str(r["ds"])[:19].replace("T", " "))
        if a is None:
            a = actual.get(r["unique_id"], {}).get(str(r["ds"])[:10])
        if a is not None:
            err.setdefault(r["unique_id"], []).append(abs(a - r["yhat"]))
    return {u: sum(v) / len(v) for u, v in err.items() if v}


FOUNDATION = ("chronos_bolt_small", "tirex_2")


def mean_fc(lists):
    acc = {}
    for lst in lists:
        for r in lst:
            acc.setdefault((r["unique_id"], r["ds"]), []).append(r["yhat"])
    return [{"unique_id": u, "ds": d, "yhat": sum(v) / len(v)} for (u, d), v in acc.items()]


def norm_ds(s):
    return s if len(s) <= 10 else s.replace("T", " ")[:19]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4000")
    ap.add_argument("--only", default="")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--models", default="seasonal_naive,auto_ets,theta,chronos_bolt_small,tirex_2")
    a = ap.parse_args()
    key = os.environ.get("LITELLM_KEY", "")
    cl = Client(a.url, key)
    models = [m for m in a.models.split(",") if m]
    base = "seasonal_naive"
    if base not in models:
        models.insert(0, base)
    only = {x for x in a.only.split(",") if x}
    short = {"seasonal_naive": "sNaiv", "auto_ets": "ETS", "theta": "Theta", "chronos_bolt_small": "Chron", "tirex_2": "TiRex"}

    results = []
    print(f"{'Datensatz':<10}{'h':>3} {'Auto wählt':<10}{'Auto':>6} " + " ".join(f"{short.get(m, m[:6]):>6}" for m in models) + "   Ens2 Top2")
    for name, loader, last_n, h, desc in SPECS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            data = loader()
            train, actual = [], {}
            for uid, pts in data.items():
                pts = pts[-last_n:] if last_n else pts
                for d, v in pts[:-h]:
                    train.append({"unique_id": uid, "ds": norm_ds(d), "y": round(v, 4)})
                actual[uid] = {norm_ds(d): v for d, v in pts[-h:]}
            auto = cl.call("forecast", {"series": train, "h": h, "models": ["auto"], "n_windows": a.windows})
            sel = auto["selected_model"]
            rel = {}
            per = {}
            fcs = {}
            for m in models:
                try:
                    r = cl.call("forecast", {"series": train, "h": h, "models": [m], "n_windows": 0, "freq": auto["freq"]})
                    fcs[m] = r["forecast"]
                    per[m] = mae_by_series(r["forecast"], actual)
                except Exception as e:  # Modell nicht verfügbar o. Ä.
                    per[m] = None
            # Ensembles (Mittelwert der Punktprognosen), clientseitig berechnet
            per["__ens2__"] = mae_by_series(mean_fc([fcs[m] for m in FOUNDATION if m in fcs]), actual) if all(m in fcs for m in FOUNDATION) else None
            ranked = sorted(
                (k for k, v in (auto.get("backtest") or {}).items() if k in fcs and k != base and v.get("relative_mae", 9) < 1),
                key=lambda k: auto["backtest"][k]["relative_mae"],
            )[:2]
            per["__top2__"] = mae_by_series(mean_fc([fcs[m] for m in ranked]), actual) if len(ranked) == 2 else per.get(ranked[0] if ranked else base)
            bmae = per.get(base)
            ra = mae_by_series(auto["forecast"], actual)
            for m in models + ["__ens2__", "__top2__", "__auto__"]:
                mm = ra if m == "__auto__" else per.get(m)
                if not mm or not bmae:
                    rel[m] = None
                    continue
                vals = [mm[u] / bmae[u] for u in mm if u in bmae and bmae[u] > 0]
                rel[m] = sum(vals) / len(vals) if vals else None
            f = lambda x: "   n/a" if x is None else f"{x:6.2f}"
            flag = "" if auto.get("validated") else "*"
            print(f"{name:<10}{h:>3} {("Ensemble" if sel.startswith("ensemble(") else short.get(sel, sel[:9])) + flag:<10}{f(rel['__auto__'])} " + " ".join(f(rel[m]) for m in models) + "  " + f(rel["__ens2__"]) + f(rel["__top2__"]))
            results.append((name, sel, auto.get("validated"), rel))
        except Exception as e:
            print(f"{name:<10}{h:>3} FEHLER: {str(e)[:90]}")

    if not results:
        return
    print("\n* = Service fand kein besseres Modell als die Baseline und lieferte diese.")
    print("Werte = relative MAE im Rückblick (Baseline = 1.00, kleiner ist besser).\n")
    ok = [r for r in results if r[3].get("__auto__") is not None]
    if ok:
        def avg(k):
            v = [r[3][k] for r in ok if r[3].get(k) is not None]
            return sum(v) / len(v) if v else None
        print(f"Datensätze gewertet: {len(ok)}")
        fmt = lambda x: "n/a (Modell nicht verfügbar)" if x is None else f"{x:.3f}"
        print(f"Mittel Auto-Wahl : {fmt(avg('__auto__'))}")
        for m in models:
            n = sum(1 for r in ok if r[3].get(m) is not None)
            print(f"Mittel {short.get(m, m):<9}: {fmt(avg(m))}" + (f"  ({n} Sätze)" if n and n < len(ok) else ""))
        print(f"Mittel Ens2 (Chronos+TiRex)         : {fmt(avg('__ens2__'))}")
        print(f"Mittel Top2 (beste 2 aus Backtest)  : {fmt(avg('__top2__'))}")
        best = 0
        for r in ok:
            cands = [v for k, v in r[3].items() if not k.startswith("__") and v is not None]
            if cands and r[3]["__auto__"] <= min(cands) * 1.05:
                best += 1
        print(f"Auto liegt bei {best} von {len(ok)} Sätzen höchstens 5 % hinter dem besten Einzelmodell.")
        beat = sum(1 for r in ok if r[3]["__auto__"] < 1.0)
        print(f"Auto schlägt die Baseline in {beat} von {len(ok)} Sätzen.")


if __name__ == "__main__":
    main()
