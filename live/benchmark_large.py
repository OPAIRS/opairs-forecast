#!/usr/bin/env python3
"""Größerer, robusterer Benchmark: M4-Teilstichproben (offizielle Trainings-/Testdaten), eigene CSV-Daten und
synthetische intermittierende Reihen.

Was dieser Benchmark gegenüber live/benchmark.py ändert:
  * Hunderte statt zwölf Reihen (Stichprobe je M4-Frequenz, feste Zufallszahl), offizieller M4-Holdout.
  * Robustere Kennzahlen: geometrisches Mittel der relativen Fehler (Verhältnisse gehören geometrisch gemittelt),
    Median, Gewinnrate gegen die Baseline, MASE, sMAPE, Intervallabdeckung.
  * Kalibrierung von `validated`: Wie oft schlägt eine als validiert gemeldete Auswahl die Baseline auch im echten Holdout?
  * Wahlweise lokal (ohne Server, nur die in diesem Repo installierten Modelle) oder gegen einen laufenden Dienst.

Beispiele:
  python live/benchmark_large.py --source m4 --freqs Quarterly,Weekly,Hourly --n 50                 # lokal, Statistikmodelle
  python live/benchmark_large.py --source m4 --n 100 --url http://localhost:8200 --key $FORECAST_API_KEY   # REST
  LITELLM_KEY=... python live/benchmark_large.py --source m4 --n 100 --url http://localhost:4000 --via mcp  # über LiteLLM
  python live/benchmark_large.py --source csv --csv daten.csv --id-col artikel --ds-col monat --y-col menge --h 6 --freq MS
  python live/benchmark_large.py --source intermittent --n 60                                              # synthetisch

Grenzen: M4 hat keine echten Zeitstempel; es werden regelmäßige Daten vergeben (für die Modelle nur die Frequenz relevant).
Die Modellwahl des Dienstes ist global je Aufruf; `--group-size` bestimmt, wie viele Reihen sich eine Wahl teilen.
M4 ist eine allgemeine Sammlung und kein Ersatz für eigene Fertigungsdaten: dafür `--source csv`.
"""
import argparse
import csv
import io
import json
import math
import os
import random
import sys
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta

M4_BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/"
# Frequenz -> (pandas-Frequenz, Horizont laut M4, Saisonlänge)
M4 = {
    "Yearly": ("YS", 6, 1),
    "Quarterly": ("QS", 8, 4),
    "Monthly": ("MS", 18, 12),
    "Weekly": ("W-MON", 13, 52),
    "Daily": ("D", 14, 7),
    "Hourly": ("h", 48, 24),
}
CACHE = os.environ.get("BENCH_CACHE", "/tmp/fc-bench")


# ----------------------------------------------------------------------------------------------- Daten
def fetch_lines(url):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, url.replace("/", "_").replace(":", ""))
    if not os.path.exists(p):
        with urllib.request.urlopen(url, timeout=300) as r, open(p, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    with open(p, encoding="utf-8") as f:
        next(f)  # Kopfzeile
        for line in f:
            yield line


def make_dates(freq, n, start=None):
    if freq == "W-MON":
        d0 = date(2000, 1, 3)
        return [(d0 + timedelta(days=7 * i)).isoformat() for i in range(n)]
    if freq == "D":
        d0 = date(2000, 1, 1)
        return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]
    if freq == "h":
        t0 = datetime(2000, 1, 1)
        return [(t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(n)]
    if freq == "MS":
        return [date(2000 + i // 12, i % 12 + 1, 1).isoformat() for i in range(n)]
    if freq == "QS":
        return [date(2000 + i // 4, (i % 4) * 3 + 1, 1).isoformat() for i in range(n)]
    if freq == "YS":
        return [date(1900 + i, 1, 1).isoformat() for i in range(n)]
    raise ValueError(freq)


def parse_row(line):
    f = line.rstrip("\r\n").split(",")
    sid = f[0].strip('"')
    vals = [float(x.strip().strip('"')) for x in f[1:] if x.strip().strip('"') != ""]
    return sid, vals


def load_m4(name, n, seed, windows):
    freq, h, season = M4[name]
    min_len = max(2 * season, 8) + h  # damit mindestens ein Backtest-Fenster passt
    rnd = random.Random(seed)
    keep, seen = [], 0
    for line in fetch_lines(M4_BASE + f"Train/{name}-train.csv"):
        sid, vals = parse_row(line)
        if len(vals) < min_len:
            continue
        seen += 1
        if len(keep) < n:
            keep.append((sid, vals))
        else:
            j = rnd.randrange(seen)
            if j < n:
                keep[j] = (sid, vals)
    test = {sid: vals for sid, vals in (parse_row(l) for l in fetch_lines(M4_BASE + f"Test/{name}-test.csv"))}
    series = [(sid, vals, test[sid][:h]) for sid, vals in sorted(keep)]
    return {"freq": freq, "h": h, "season": season, "series": series, "label": f"M4 {name}"}


def load_csv(path, id_col, ds_col, y_col, h, freq, season):
    by = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r[id_col], []).append((r[ds_col], float(r[y_col])))
    series = []
    for sid, pts in by.items():
        pts.sort()
        if len(pts) <= h + 8:
            continue
        series.append((sid, [v for _, v in pts[:-h]], [v for _, v in pts[-h:]], [d for d, _ in pts]))
    return {"freq": freq, "h": h, "season": season, "series": series, "label": f"CSV {os.path.basename(path)}", "dated": True}


def load_intermittent(n, seed):
    rnd = random.Random(seed)
    h, length = 6, 72
    series = []
    for i in range(n):
        p, size = rnd.uniform(0.15, 0.5), rnd.uniform(3, 20)
        y = [round(rnd.lognormvariate(math.log(size), 0.4)) if rnd.random() < p else 0.0 for _ in range(length + h)]
        series.append((f"p{i:03d}", y[:length], y[length:]))
    return {"freq": "MS", "h": h, "season": 12, "series": series, "label": "synthetisch intermittent (Monate)"}


def to_records(ds, series):
    recs = []
    for item in series:
        sid, train = item[0], item[1]
        dates = item[3][: len(train)] if len(item) > 3 else make_dates(ds["freq"], len(train))
        recs += [{"unique_id": sid, "ds": d, "y": v} for d, v in zip(dates, train)]
    return recs


# ----------------------------------------------------------------------------------------------- Engines
class Local:
    def __init__(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app import core

        self.core = core
        from app.registry import describe_models

        self.models = {m["name"] for m in describe_models() if m["available"]}

    def forecast(self, recs, h, models, n_windows, freq, season, level=80):
        return self.core.run_forecast(recs, h, freq, season, models, n_windows, level)


class Remote:
    def __init__(self, url, key, via):
        self.url, self.key, self.via = url.rstrip("/"), key, via
        self.sid = None
        if via == "mcp":
            raw = json.loads(self._req("GET", "/v1/mcp/server", None))
            items = raw if isinstance(raw, list) else raw.get("servers") or raw.get("data") or [raw]
            self.sid = next((x.get("server_id") for x in items if isinstance(x, dict) and "opairs_forecast" in json.dumps(x)), None)
            if not self.sid:
                sys.exit("server_id nicht gefunden")
            res = self._mcp("list_models", {})
            self.models = {m["name"] for m in res["models"] if m["available"]}
        else:
            self.models = {m["name"] for m in json.loads(self._req("GET", "/v1/models", None))["models"] if m["available"]}

    def _req(self, method, path, body):
        hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None, method=method, headers=hdr)
        with urllib.request.urlopen(req, timeout=3600) as r:
            return r.read()

    def _mcp(self, name, args):
        r = json.loads(self._req("POST", "/mcp-rest/tools/call", {"server_id": self.sid, "name": name, "arguments": args}))
        text = "".join(c.get("text", "") for c in r.get("content", []))
        if r.get("isError"):
            raise RuntimeError(text[:300])
        return json.loads(text)

    def forecast(self, recs, h, models, n_windows, freq, season, level=80):
        args = {"series": recs, "h": h, "models": models, "n_windows": n_windows, "level": level, "freq": freq, "season_length": season}
        if self.via == "mcp":
            return self._mcp("forecast", args)
        return json.loads(self._req("POST", "/v1/forecast", args))


# ----------------------------------------------------------------------------------------------- Kennzahlen
def mase_scale(train, m):
    m = m if len(train) > 2 * m else 1
    d = [abs(train[i] - train[i - m]) for i in range(m, len(train))]
    return sum(d) / len(d) if d else 0.0


def smape(a, f):
    num = sum(abs(x - y) * 2 for x, y in zip(a, f))
    den = sum(abs(x) + abs(y) for x, y in zip(a, f))
    return 100 * num / den if den else 0.0


def metrics(actual, fc, train, season):
    n = len(actual)
    mae = sum(abs(a - f) for a, f in zip(actual, fc)) / n
    rmse = math.sqrt(sum((a - f) ** 2 for a, f in zip(actual, fc)) / n)
    sc = mase_scale(train, season)
    return {"mae": mae, "rmse": rmse, "mase": mae / sc if sc else float("nan"), "smape": smape(actual, fc)}


def rel(x, b):
    """Fehler relativ zur Baseline. Ist die Baseline im Holdout fehlerfrei (b = 0), ist das Verhältnis
    nicht definiert: die Reihe zählt dann nicht in die relativen Kennzahlen (None)."""
    if b <= 1e-9:
        return None
    return min(max(x / b, 1e-3), 1e3)


def geo(v):
    v = [max(x, 1e-6) for x in v if x is not None]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def median(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2 if v else float("nan")


def mean(v):
    v = [x for x in v if x is not None and x == x]
    return sum(v) / len(v) if v else float("nan")


# ----------------------------------------------------------------------------------------------- Lauf
def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_variant(engine, ds, series, models, n_windows, group_size, level=80):
    """Gibt {sid: forecast-Liste, ...}, {sid: gewähltes Modell}, {sid: validated}, {sid: (lo, hi)} zurück."""
    fcs, picked, valid, bands = {}, {}, {}, {}
    for grp in chunks(series, group_size):
        recs = to_records(ds, grp)
        res = engine.forecast(recs, ds["h"], models, n_windows, ds["freq"], ds["season"], level)
        rows = {}
        for r in res["forecast"]:
            rows.setdefault(str(r["unique_id"]), []).append(r)
        for item in grp:
            sid = item[0]
            rs = sorted(rows[sid], key=lambda r: r["ds"])
            fcs[sid] = [r["yhat"] for r in rs]
            bands[sid] = [(r["lo"], r["hi"]) for r in rs]
            picked[sid], valid[sid] = res["selected_model"], res.get("validated")
    return fcs, picked, valid, bands


def evaluate(engine, ds, args):
    series = ds["series"]
    base_fc, *_ = run_variant(engine, ds, series, ["seasonal_naive"], 0, args.group_size)
    singles = [m for m in args.models.split(",") if m and m != "seasonal_naive"]
    singles = [m for m in singles if m in engine.models]
    variants = {"auto": ["auto"]} | {m: [m] for m in singles}
    per = {}
    for name, models in variants.items():
        try:
            fc, picked, valid, bands = run_variant(engine, ds, series, models, args.windows if name == "auto" else 0, args.group_size)
        except Exception as e:  # ein defektes Modell soll den Lauf nicht beenden
            print(f"  {name}: übersprungen ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
            continue
        rows = []
        for item in series:
            sid, train, actual = item[0], item[1], item[2]
            m, b = metrics(actual, fc[sid], train, ds["season"]), metrics(actual, base_fc[sid], train, ds["season"])
            cov = None
            lo_hi = bands[sid]
            if all(x[0] is not None and x[1] is not None for x in lo_hi):
                cov = sum(lo <= a <= hi for a, (lo, hi) in zip(actual, lo_hi)) / len(actual)
            rows.append({"rel_mae": rel(m["mae"], b["mae"]), "rel_rmse": rel(m["rmse"], b["rmse"]), "mase": m["mase"],
                         "smape": m["smape"], "cov": cov, "valid": valid[sid], "picked": picked[sid]})
        per[name] = rows
    return per


def summarize(per):
    out = {}
    for name, rows in per.items():
        v = [r for r in rows if r["valid"] is True]
        covs = [r["cov"] for r in rows if r["cov"] is not None]
        rel_rows = [r for r in rows if r["rel_mae"] is not None]
        out[name] = {
            "n": len(rows),
            "geo_mae": geo([r["rel_mae"] for r in rows]),
            "mean_mae": mean([r["rel_mae"] for r in rows]),
            "median_mae": median([r["rel_mae"] for r in rows]),
            "geo_rmse": geo([r["rel_rmse"] for r in rows]),
            "n_rel": len(rel_rows),
            "win": (sum(r["rel_mae"] < 1 for r in rel_rows) / len(rel_rows)) if rel_rows else float("nan"),
            "picked": dict(sorted(Counter(r["picked"] for r in rows).items(), key=lambda kv: -kv[1])),
            "mase": mean([r["mase"] for r in rows]),
            "smape": mean([r["smape"] for r in rows]),
            "cov": mean(covs) if covs else float("nan"),
            "valid_n": len(v),
            "valid_hold": (sum(r["rel_mae"] < 1 for r in v if r["rel_mae"] is not None) / max(1, sum(r["rel_mae"] is not None for r in v))) if v else float("nan"),
        }
    return out


def f(x, d=3):
    return "–" if x != x else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["m4", "csv", "intermittent"], default="m4")
    ap.add_argument("--freqs", default="Yearly,Quarterly,Monthly,Weekly,Daily,Hourly", help="M4-Frequenzen")
    ap.add_argument("--n", type=int, default=50, help="Reihen je Frequenz (M4) bzw. Anzahl synthetischer Reihen")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--windows", type=int, default=3, help="Backtest-Fenster der automatischen Wahl")
    ap.add_argument("--group-size", type=int, default=25, help="Reihen, die sich eine Modellwahl teilen")
    ap.add_argument("--models", default="theta,auto_ets,auto_arima,croston_sba,tsb,imapa,adida,chronos_bolt_small,chronos_2,tirex_2", help="Einzelmodelle zum Vergleich (nur verfügbare)")
    ap.add_argument("--url", default="", help="Dienst-URL (leer = lokal ohne Server)")
    ap.add_argument("--key", default=os.environ.get("FORECAST_API_KEY", os.environ.get("LITELLM_KEY", "")))
    ap.add_argument("--via", choices=["rest", "mcp"], default="rest")
    ap.add_argument("--csv", default="")
    ap.add_argument("--id-col", default="unique_id")
    ap.add_argument("--ds-col", default="ds")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--h", type=int, default=6, help="Horizont für --source csv")
    ap.add_argument("--freq", default="MS", help="Frequenz für --source csv (pandas-Notation)")
    ap.add_argument("--season", type=int, default=12, help="Saisonlänge für --source csv")
    ap.add_argument("--out", default="", help="Ergebnis zusätzlich als JSON speichern")
    args = ap.parse_args()

    engine = Remote(args.url, args.key, args.via) if args.url else Local()
    print(f"Modelle verfügbar: {', '.join(sorted(engine.models))}\n")

    sets = []
    if args.source == "m4":
        for name in [x for x in args.freqs.split(",") if x]:
            print(f"Lade M4 {name} …", file=sys.stderr, flush=True)
            sets.append(load_m4(name, args.n, args.seed, args.windows))
    elif args.source == "csv":
        sets.append(load_csv(args.csv, args.id_col, args.ds_col, args.y_col, args.h, args.freq, args.season))
    else:
        sets.append(load_intermittent(args.n, args.seed))

    results, pooled = {}, {}
    for ds in sets:
        print(f"{ds['label']}: {len(ds['series'])} Reihen, h={ds['h']}, Saison {ds['season']} …", file=sys.stderr, flush=True)
        per = evaluate(engine, ds, args)
        results[ds["label"]] = summarize(per)
        for k, rows in per.items():
            pooled.setdefault(k, []).extend(rows)
    results["Alle Reihen zusammen"] = summarize(pooled)

    print("Relativer Fehler gegen `seasonal_naive` im echten Holdout (1,00 = Baseline, kleiner ist besser).")
    print("geo = geometrisches Mittel (Hauptkennzahl), Median, Gewinnrate = Anteil der Reihen besser als die Baseline.")
    print("Reihen, deren Baseline im Holdout fehlerfrei ist (Verhältnis undefiniert), zählen nicht in die relativen Kennzahlen (Spalte „rel. Reihen“).\n")
    for label, summ in results.items():
        print(f"### {label}\n")
        print("| Modell | Reihen | rel. Reihen | rel. MAE geo | Median | arithm. | rel. RMSE geo | Gewinnrate | MASE | sMAPE % | Abdeckung 80 % |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, s in sorted(summ.items(), key=lambda kv: kv[1]["geo_mae"]):
            print(f"| {name} | {s['n']} | {s['n_rel']} | {f(s['geo_mae'])} | {f(s['median_mae'])} | {f(s['mean_mae'])} | {f(s['geo_rmse'])} | {f(100 * s['win'], 0)} % | {f(s['mase'], 2)} | {f(s['smape'], 1)} | {f(100 * s['cov'], 0) if s['cov'] == s['cov'] else '–'} |")
        a = summ.get("auto")
        if a:
            print(f"\n`auto`: {a['valid_n']} von {a['n']} Reihen als validiert gemeldet; davon schlagen im Holdout {f(100 * a['valid_hold'], 0)} % die Baseline.")
            print("Gewählte Modelle: " + ", ".join(f"{k} ×{v}" for k, v in a["picked"].items()) + "\n")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False, default=lambda x: None)
        print(f"Gespeichert: {args.out}")


if __name__ == "__main__":
    main()
