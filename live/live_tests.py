#!/usr/bin/env python3
"""Live-Tests gegen einen laufenden Forecast-Service. Nur Standardbibliothek.

Aufruf:  KEY=... BASE=http://localhost:18200 python3 live_tests.py [--quick] [--all]
  --quick  ohne Skalierungs-, Parallelitäts- und Modellvergleichstest
  --all    Modellvergleich auch mit auto_arima (bei Wochendaten mit Saison 52 sehr langsam)
"""
import datetime
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.environ.get("BASE", "http://localhost:18200").rstrip("/")
KEY = os.environ.get("KEY", "")
QUICK = "--quick" in sys.argv
ALL = "--all" in sys.argv


class Skip(Exception):
    pass


def req(method, path, body=None, key=None, timeout=1800):
    key = KEY if key is None else key
    headers = {"content-type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    r = urllib.request.Request(
        BASE + path, data=None if body is None else json.dumps(body).encode(), headers=headers, method=method
    )
    t = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None), time.time() - t
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            j = json.loads(raw)
        except Exception:
            j = {"detail": raw.decode(errors="replace")}
        return e.code, j, time.time() - t


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def detail(j):
    return str((j or {}).get("detail", ""))


# ----------------------------------------------------------------- Daten
def make_dates(n, freq):
    if freq == "W-MON":
        s = datetime.date(2022, 1, 3)
        return [(s + datetime.timedelta(weeks=i)).isoformat() for i in range(n)]
    if freq == "D":
        s = datetime.date(2024, 1, 1)
        return [(s + datetime.timedelta(days=i)).isoformat() for i in range(n)]
    if freq == "H":
        s = datetime.datetime(2024, 1, 1)
        return [(s + datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]
    if freq == "MS":
        out, y, m = [], 2014, 1
        for _ in range(n):
            out.append(f"{y}-{m:02d}-01")
            m += 1
            if m > 12:
                m, y = 1, y + 1
        return out
    raise ValueError(freq)


def make_series(uid, n, freq, period, seed, kind="seasonal", trend=0.3, amp=10.0, noise=2.0, level=100.0,
                spike_at=None, spike=0.0):
    rng = random.Random(seed)
    walk, y = level, []
    for i in range(n):
        if kind == "seasonal":
            v = level + trend * i + amp * math.sin(2 * math.pi * i / period) + rng.gauss(0, noise)
        elif kind == "walk":
            walk += rng.gauss(0, 1)
            v = walk
        elif kind == "intermittent":
            v = float(rng.choice([0, 0, 0, 0, 5, 8, 12]))
        elif kind == "constant":
            v = level
        else:
            raise ValueError(kind)
        if spike_at is not None and i == spike_at:
            v += spike
        y.append(v)
    return [{"unique_id": uid, "ds": d, "y": v} for d, v in zip(make_dates(n, freq), y)]


def weekly(n_series=3, n=150, seed=0, **kw):
    rows = []
    for i in range(n_series):
        rows += make_series(f"s{i}", n, "W-MON", 52, seed + i, **kw)
    return rows


def spike_data():
    rows = weekly(2)
    rows += make_series("s2", 150, "W-MON", 52, 2, spike_at=140, spike=60.0)
    return rows, make_dates(150, "W-MON")[140]


def fc(rows, **kw):
    body = {"series": rows, "h": kw.pop("h", 8), "n_windows": kw.pop("n_windows", 2)}
    body.update(kw)
    return req("POST", "/v1/forecast", body)


def models_info():
    s, j, _ = req("GET", "/v1/models")
    ok(s == 200, f"/v1/models Status {s}")
    return {m["name"]: m for m in j["models"]}


def avail(name):
    m = models_info().get(name)
    return bool(m and m["available"])


def check_rows(j, n_series, h):
    rows = j["forecast"]
    ok(len(rows) == n_series * h, f"{len(rows)} Zeilen statt {n_series * h}")
    for r in rows:
        ok(r["yhat"] is not None and math.isfinite(r["yhat"]), "yhat fehlt oder ist nicht endlich")
        if r["lo"] is not None and r["hi"] is not None:
            ok(r["lo"] <= r["yhat"] + 1e-6 and r["yhat"] <= r["hi"] + 1e-6, f"Intervall unstimmig: {r}")


# ----------------------------------------------------------------- Tests
def t_health():
    s, j, _ = req("GET", "/health", key="")
    ok(s == 200 and j["status"] == "ok", f"{s} {j}")
    return f"Version {j['version']}"


def t_auth():
    if not KEY:
        raise Skip("kein KEY gesetzt, Service evtl. ohne Auth")
    s1, _, _ = req("GET", "/v1/models", key="falsch")
    s2, _, _ = req("POST", "/v1/forecast", {"series": [], "h": 1}, key="")
    ok(s1 == 401 and s2 == 401, f"falscher Key {s1}, ohne Key {s2}")
    return "401 bei falschem und fehlendem Key"


def t_models():
    m = models_info()
    ok(m["seasonal_naive"]["available"] and m["seasonal_naive"]["baseline"], "Baseline fehlt")
    on = [n for n, v in m.items() if v["available"]]
    off = [n for n, v in m.items() if not v["available"]]
    return f"verfügbar: {', '.join(on)} | nicht: {', '.join(off) or '-'}"


def t_weekly():
    s, j, dt = fc(weekly(3), h=8)
    ok(s == 200, f"{s} {detail(j)}")
    check_rows(j, 3, 8)
    ok(j["validated"] is True, f"nicht validiert: {j['warnings']}")
    ok(j["freq"] == "W-MON" and j["season_length"] == 52, f"freq/season: {j['freq']} {j['season_length']}")
    last = datetime.date.fromisoformat(make_dates(150, "W-MON")[-1])
    first = [r for r in j["forecast"] if r["unique_id"] == "s0"][0]["ds"]
    ok(first == (last + datetime.timedelta(weeks=1)).isoformat(), f"erste Prognosewoche {first}")
    return f"{j['selected_model']} (rel. MAE {j['backtest'][j['selected_model']]['relative_mae']:.3f}), {dt:.1f}s"


def t_daily():
    rows = make_series("d0", 400, "D", 7, 1) + make_series("d1", 400, "D", 7, 2)
    s, j, dt = fc(rows, h=14)
    ok(s == 200, f"{s} {detail(j)}")
    ok(j["season_length"] == 7, f"season_length {j['season_length']}")
    check_rows(j, 2, 14)
    return f"{j['selected_model']}, validated={j['validated']}, {dt:.1f}s"


def t_monthly():
    rows = make_series("m0", 120, "MS", 12, 1) + make_series("m1", 120, "MS", 12, 2)
    s, j, dt = fc(rows, h=6)
    ok(s == 200, f"{s} {detail(j)}")
    ok(j["freq"] == "MS" and j["season_length"] == 12, f"{j['freq']} {j['season_length']}")
    check_rows(j, 2, 6)
    return f"{j['selected_model']}, validated={j['validated']}, {dt:.1f}s"


def t_hourly():
    rows = make_series("h0", 500, "H", 24, 1, trend=0.01)
    s, j, dt = fc(rows, h=24)
    ok(s == 200, f"{s} {detail(j)}")
    ok(j["season_length"] == 24 and j["freq"].lower() == "h", f"{j['freq']} {j['season_length']}")
    check_rows(j, 1, 24)
    return f"{j['selected_model']}, validated={j['validated']}, {dt:.1f}s"


def t_long_horizon():
    s, j, dt = fc(weekly(2), h=26)
    ok(s == 200, f"{s} {detail(j)}")
    check_rows(j, 2, 26)
    return f"h=26 ok, {j['selected_model']}, {dt:.1f}s"


def t_random_walk():
    rows = []
    for i in range(3):
        rows += make_series(f"w{i}", 150, "W-MON", 52, 10 + i, kind="walk")
    s, j, dt = fc(rows, h=8)
    ok(s == 200, f"{s} {detail(j)}")
    if j["validated"] is False:
        ok(j["selected_model"] == j["baseline"] and j["warnings"], "validated=False ohne Baseline/Warnung")
    elif j["validated"] is True:
        ok(j["backtest"][j["selected_model"]]["relative_mae"] < 1, "validated=True aber rel. MAE >= 1")
    return f"{j['selected_model']}, validated={j['validated']} (konsistent), {dt:.1f}s"


def t_constant():
    rows = make_series("c0", 150, "W-MON", 52, 1, kind="constant")
    s, j, dt = fc(rows, h=8)
    ok(s == 200, f"{s} {detail(j)}")
    ok(j["selected_model"] == j["baseline"], f"{j['selected_model']} schlägt eine perfekte Baseline?")
    return f"Baseline gewählt, validated={j['validated']}, errors={list(j['errors'])}"


def t_intermittent():
    rows = make_series("i0", 150, "W-MON", 52, 1, kind="intermittent")
    s, j, dt = fc(rows, h=8, models=["croston", "adida", "naive"])
    ok(s == 200, f"{s} {detail(j)}")
    check_rows(j, 1, 8)
    return f"{j['selected_model']}, validated={j['validated']}, errors={list(j['errors'])}"


def t_no_backtest():
    s, j, _ = fc(weekly(1), h=4, models=["theta"], n_windows=0)
    ok(s == 200 and j["validated"] is None and j["backtest"] == {}, f"{s} {j.get('validated')}")
    s2, j2, _ = fc(weekly(1), h=4, models=["theta", "naive"], n_windows=0)
    ok(s2 == 422, f"n_windows=0 mit zwei Modellen: {s2}")
    return "ohne Backtest: validated=None; zwei Modelle ohne Backtest -> 422"


def t_backtest_endpoint():
    s, j, dt = req("POST", "/v1/backtest", {"series": weekly(3), "h": 8, "n_windows": 3})
    ok(s == 200, f"{s} {detail(j)}")
    ok(j["best_model"] in j["backtest"] and "forecast" not in j, "Antwortform")
    return f"best={j['best_model']}, validated={j['validated']}, {dt:.1f}s"


def t_anomalies_stats():
    rows, spike_ds = spike_data()
    s, j, dt = req("POST", "/v1/anomalies", {"series": rows, "model": "auto_ets", "h": 12, "n_windows": 3, "level": 95})
    ok(s == 200, f"{s} {detail(j)}")
    hit = [a for a in j["anomalies"] if a["unique_id"] == "s2" and a["ds"] == spike_ds and a["direction"] == "high"]
    ok(hit, f"Ausreißer {spike_ds} nicht gefunden: {j['anomalies'][:3]}")
    return f"auto_ets: {j['n_anomalies']} von {j['n_points']} Punkten, Spitze gefunden, {dt:.1f}s"


def anomalies_with(model):
    if not avail(model):
        raise Skip(f"{model} nicht verfügbar")
    rows, spike_ds = spike_data()
    s, j, dt = req("POST", "/v1/anomalies", {"series": rows, "model": model, "h": 12, "n_windows": 3, "level": 80})
    ok(s == 200, f"{s} {detail(j)}")
    hit = [a for a in j["anomalies"] if a["unique_id"] == "s2" and a["ds"] == spike_ds and a["direction"] == "high"]
    ok(hit, f"Ausreißer nicht gefunden: {j['anomalies'][:3]}")
    return f"{model}: {j['n_anomalies']} von {j['n_points']} Punkten, Spitze gefunden, {dt:.1f}s"


def t_anomalies_no_intervals():
    s, j, _ = req("POST", "/v1/anomalies", {"series": weekly(1), "model": "croston", "h": 8, "n_windows": 2})
    ok(s == 422 and "Prognoseintervalle" in detail(j), f"{s} {detail(j)}")
    return "Croston ohne Intervalle -> klare 422"


def t_level_note():
    if not avail("chronos_bolt_small"):
        raise Skip("chronos_bolt_small nicht verfügbar")
    s, j, dt = fc(weekly(2), h=8, models=["chronos_bolt_small"], level=95)
    ok(s == 200, f"{s} {detail(j)}")
    if j["selected_model"] == "chronos_bolt_small":
        ok(any("begrenzt" in w for w in j["warnings"]), f"keine Hinweis-Warnung: {j['warnings']}")
    return f"Level 95 bei Bolt: Warnung vorhanden ({dt:.1f}s)"


def t_foundation_forecast():
    names = [n for n in ("chronos_bolt_small", "tirex_2") if avail(n)]
    if not names:
        raise Skip("keine Foundation-Modelle verfügbar")
    s, j, dt = fc(weekly(3), h=8, models=names)
    ok(s == 200 and not j["errors"], f"{s} errors={j.get('errors')}")
    for n in names:
        ok(n in j["backtest"], f"{n} fehlt im Backtest")
    check_rows(j, 3, 8)
    return f"{', '.join(names)} im Backtest, gewählt: {j['selected_model']}, {dt:.1f}s"


def t_errors():
    base = weekly(1, n=60)
    cases = {
        "zu kurz": (fc(weekly(1, n=30), h=8), 422, "lang genug"),
        "Lücken": (fc(base[:20] + base[22:], h=4, freq="W-MON"), 422, "Lücken"),
        "Duplikat": (fc(base + [base[0]], h=4), 422, "Doppelter"),
        "y=null": (fc([dict(r, y=None) if i == 3 else r for i, r in enumerate(base)], h=4), 422, "y-Werte"),
        "unbekanntes Modell": (fc(base, h=4, models=["gibtsnicht"]), 422, "Unbekanntes Modell"),
        "h zu groß": (fc(base, h=9999), 422, ""),
        "leer": (fc([], h=4), 422, ""),
        "Spalten": (fc([{"unique_id": "a", "ds": "2024-01-01"}], h=4), 422, ""),
        "Datum kaputt": (fc([{"unique_id": "a", "ds": "kein-datum", "y": 1.0}] * 5, h=2), 422, ""),
    }
    m = models_info()
    if not m["chronos_2"]["available"]:
        cases["nicht verfügbares Modell"] = (fc(base, h=4, models=["chronos_2"]), 422, "nicht verfügbar")
    bad = []
    for name, ((s, j, _), want, frag) in cases.items():
        if s != want or (frag and frag not in detail(j)):
            bad.append(f"{name}: {s} {detail(j)[:120]}")
    ok(not bad, " | ".join(bad))
    return f"{len(cases)} Fehlerfälle liefern saubere 422"


def t_int_ids():
    rows = [dict(r, unique_id=int(r["unique_id"][1:])) for r in weekly(2)]
    s, j, _ = fc(rows, h=4)
    ok(s == 200, f"{s} {detail(j)}")
    return "ganzzahlige unique_id akzeptiert"


def t_explain():
    s, j, _ = req("POST", "/v1/explain", {"series": weekly(1), "h": 4})
    ok(s in (200, 501), f"{s} {detail(j)}")
    return "aus (501)" if s == 501 else "aktiv (200)"


MATRIX_ROWS = []


def t_matrix():
    rows = weekly(4, seed=100)
    info = models_info()
    names = [n for n, v in info.items() if v["available"] and (ALL or n != "auto_arima")]
    for n in names:
        s, j, dt = fc(rows, h=8, models=[n], n_windows=3)
        if s != 200:
            MATRIX_ROWS.append((n, None, None, dt, f"HTTP {s}: {detail(j)[:60]}"))
            continue
        b = j["backtest"].get(n)
        MATRIX_ROWS.append((n, b and b["relative_mae"], b and b["interval_coverage"], dt, "; ".join(j["errors"].values())[:60]))
    bad = [r for r in MATRIX_ROWS if r[1] is None and r[0] != "croston"]
    ok(not bad, f"ohne Ergebnis: {[b[0] for b in bad]}")
    return f"{len(names)} Modelle verglichen (Tabelle unten)"


def t_scale():
    rows = []
    for i in range(50):
        rows += make_series(f"p{i}", 120, "W-MON", 52, 1000 + i, level=50 + i)
    s, j, dt = fc(rows, h=8)
    ok(s == 200, f"{s} {detail(j)}")
    check_rows(j, 50, 8)
    return f"50 Serien x 120 Punkte, {j['selected_model']}, {dt:.1f}s"


def t_parallel():
    res, t0 = [], time.time()

    def one(i):
        res.append(fc(weekly(3, seed=i * 7), h=8)[0])

    th = [threading.Thread(target=one, args=(i,)) for i in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    ok(res == [200, 200, 200], f"Status {res}")
    return f"3 parallele Anfragen alle 200, zusammen {time.time() - t0:.1f}s"


TESTS = [
    ("Health", t_health), ("Auth", t_auth), ("Modellliste", t_models),
    ("Wochendaten auto", t_weekly), ("Tagesdaten", t_daily), ("Monatsdaten", t_monthly),
    ("Stundendaten", t_hourly), ("Langer Horizont", t_long_horizon),
    ("Random Walk (Konsistenz)", t_random_walk), ("Konstante Serie", t_constant),
    ("Sporadische Nachfrage", t_intermittent), ("Ohne Backtest", t_no_backtest),
    ("Backtest-Endpunkt", t_backtest_endpoint), ("Ganzzahlige IDs", t_int_ids),
    ("Foundation-Prognose", t_foundation_forecast), ("Intervall-Hinweis Bolt", t_level_note),
    ("Anomalien Statistik", t_anomalies_stats),
    ("Anomalien chronos_bolt_small", lambda: anomalies_with("chronos_bolt_small")),
    ("Anomalien tirex_2", lambda: anomalies_with("tirex_2")),
    ("Anomalien ohne Intervalle", t_anomalies_no_intervals),
    ("Fehlerfälle", t_errors), ("Explain-Schalter", t_explain),
]
if not QUICK:
    TESTS += [("Modellvergleich", t_matrix), ("Skalierung 50 Serien", t_scale), ("Parallelität", t_parallel)]


def main():
    print(f"Live-Tests gegen {BASE}  (quick={QUICK}, key={'ja' if KEY else 'nein'})\n")
    passed = failed = skipped = 0
    for name, fn in TESTS:
        t = time.time()
        try:
            msg = fn()
            status, passed = "PASS", passed + 1
        except Skip as e:
            status, msg, skipped = "SKIP", str(e), skipped + 1
        except AssertionError as e:
            status, msg, failed = "FAIL", str(e), failed + 1
        except Exception as e:
            status, msg, failed = "FAIL", f"{type(e).__name__}: {e}", failed + 1
        print(f"{status:4}  {name:30} {time.time() - t:6.1f}s  {msg}", flush=True)
    if MATRIX_ROWS:
        print("\nModellvergleich (4 Wochenserien, h=8, 3 Fenster; rel. MAE < 1 = besser als Seasonal-Naive)")
        print(f"{'Modell':22}{'rel. MAE':>10}{'Abdeckung':>11}{'Zeit s':>9}  Hinweis")
        for n, rel, cov, dt, note in sorted(MATRIX_ROWS, key=lambda r: (r[1] is None, r[1] or 0)):
            print(f"{n:22}{'-' if rel is None else format(rel, '.3f'):>10}"
                  f"{'-' if cov is None else format(cov, '.2f'):>11}{dt:9.1f}  {note}")
    print(f"\nErgebnis: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    sys.exit(1 if failed else 0)


main()
