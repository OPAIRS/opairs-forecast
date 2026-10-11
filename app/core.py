"""Kernlogik: Rolling-Origin-Backtest, Modellauswahl ("muss die Baseline schlagen"), Prognose, Anomalien.

Alles ist modellunabhängig: jedes Modell läuft über dieselbe Adapter-Schnittstelle,
Backtest und Metriken sind einheitlich und kommen nicht aus einer Fremdbibliothek.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ENSEMBLE_TOLERANCE, MAX_HORIZON, MAX_CONCURRENCY
from .data import DataError, default_season_length, infer_freq, to_frame
from .registry import baseline_name, get_adapter, resolve_models

log = logging.getLogger("forecast")
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)


# --------------------------------------------------------------------------- helpers
def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = df[cols].copy()
    if "ds" in out.columns:
        has_time = (out["ds"].dt.normalize() != out["ds"]).any()
        out["ds"] = out["ds"].dt.strftime("%Y-%m-%dT%H:%M:%S" if has_time else "%Y-%m-%d")
    out = out.astype(object).where(out.notna(), None)
    for c in cols:
        if c in ("yhat", "lo", "hi", "y", "expected"):
            out[c] = out[c].map(lambda v: None if v is None else float(v))
    return out.to_dict("records")


@dataclass
class Prepared:
    df: pd.DataFrame
    freq: str
    season_length: int
    warnings: list[str] = field(default_factory=list)


def prepare(records: list[dict], freq: str | None, season_length: int | None, h: int) -> Prepared:
    if not 1 <= h <= MAX_HORIZON:
        raise DataError(f"h muss zwischen 1 und {MAX_HORIZON} liegen")
    df = to_frame(records)
    f = infer_freq(df, freq)
    sl = season_length or default_season_length(f)
    if sl < 1:
        raise DataError("season_length muss >= 1 sein")
    return Prepared(df, f, sl)


# --------------------------------------------------------------------------- backtest
def _windows(df: pd.DataFrame, h: int, n_windows: int, min_train: int):
    """Liefert {j: (train, test)} mit j = Fensternummer vom Ende (j=1 ist das letzte Fenster)
    und eine Liste übersprungener Serien. Pro Serie so viele Fenster wie passen (max n_windows)."""
    per_j: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
    skipped = []
    for uid, g in df.groupby("unique_id", sort=False):
        L = len(g)
        k = min(n_windows, (L - min_train) // h)
        if k < 1:
            skipped.append(uid)
            continue
        for j in range(1, k + 1):
            cut = L - j * h
            per_j.setdefault(j, []).append((g.iloc[:cut], g.iloc[cut : cut + h]))
    out = {}
    for j, pairs in per_j.items():
        out[j] = (
            pd.concat([p[0] for p in pairs], ignore_index=True),
            pd.concat([p[1] for p in pairs], ignore_index=True),
        )
    return out, skipped


def backtest_model(name: str, prep: Prepared, h: int, n_windows: int, level: int) -> tuple[pd.DataFrame, list[str]]:
    """Backtest eines Modells. Gibt (Zeilen mit y/yhat/lo/hi, übersprungene Serien) zurück."""
    adapter = get_adapter(name)
    min_train = max(2 * prep.season_length, 8)
    wins, skipped = _windows(prep.df, h, n_windows, min_train)
    if not wins:
        raise DataError(
            f"Keine Serie lang genug für Backtest (nötig: mind. {min_train + h} Punkte bei h={h}, "
            f"season_length={prep.season_length}). h oder n_windows verkleinern."
        )
    frames = []
    for j, (train, test) in wins.items():
        pred = adapter.predict(train, h, prep.freq, level, prep.season_length)
        m = test.merge(pred, on=["unique_id", "ds"], how="left")
        if m["yhat"].isna().any():
            raise RuntimeError(f"{name}: Prognose unvollständig (Fenster {j})")
        m["window"] = j
        frames.append(m)
    return pd.concat(frames, ignore_index=True), skipped


def _mae_by_series(bt: pd.DataFrame) -> pd.Series:
    return (bt["y"] - bt["yhat"]).abs().groupby(bt["unique_id"]).mean()


def score_models(results: dict[str, pd.DataFrame], baseline: str) -> dict[str, dict]:
    """Relative MAE gegen die Baseline (pro Serie, dann gemittelt). <1 heißt besser als Baseline.

    Trifft die Baseline eine Serie praktisch perfekt (MAE ~ 0), kann kein Modell sie schlagen:
    gleich perfekt -> Verhältnis 1, sonst sehr großes Verhältnis."""
    base_bt = results[baseline]
    base_mae = _mae_by_series(base_bt)
    floor = 1e-9 * base_bt["y"].abs().groupby(base_bt["unique_id"]).mean() + 1e-12
    scores = {}
    for name, bt in results.items():
        mae = _mae_by_series(bt)
        common = base_mae.index.intersection(mae.index)
        ratios = []
        for u in common:
            if base_mae[u] <= floor[u] and mae[u] <= floor[u]:
                ratios.append(1.0)
            else:
                ratios.append(float(mae[u] / max(base_mae[u], floor[u])))
        rel = float(np.mean(ratios)) if ratios else None
        cov = None
        if bt["lo"].notna().all():
            cov = float(((bt["y"] >= bt["lo"]) & (bt["y"] <= bt["hi"])).mean())
        scores[name] = {
            "mae": float(mae.mean()),
            "relative_mae": rel,
            "beats_baseline": bool(rel is not None and rel < 1.0 and name != baseline),
            "interval_coverage": cov,
            "n_series": int(len(mae)),
        }
    return scores


def select_model(scores: dict[str, dict], baseline: str) -> tuple[str, bool | None, str | None]:
    """(Modell, validiert, Warnung). Kein Kandidat besser als Baseline -> Baseline, validated=False."""
    cands = {n: s for n, s in scores.items() if n != baseline and s["relative_mae"] is not None}
    if not cands:
        return baseline, None, "Kein Vergleichsmodell auswertbar; es wird die Baseline verwendet."
    best = min(cands, key=lambda n: cands[n]["relative_mae"])
    if cands[best]["relative_mae"] < 1.0:
        return best, True, None
    return (
        baseline,
        False,
        f"Kein Modell schlägt die Baseline '{baseline}' im Backtest; es wird die Baseline geliefert. "
        "Ergebnis mit Vorsicht verwenden.",
    )


# --------------------------------------------------------------------------- ensemble
ENS_PREFIX = "ensemble("


def ensemble_name(members: list[str]) -> str:
    return f"{ENS_PREFIX}{'+'.join(members)})"


def ensemble_members(name: str) -> list[str] | None:
    if name.startswith(ENS_PREFIX) and name.endswith(")"):
        return name[len(ENS_PREFIX) : -1].split("+")
    return None


def _blend(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Mittelwert von yhat/lo/hi über Modelle (Intervalle: Mittel der Grenzen, eine Näherung)."""
    out = frames[0].copy()
    idx = out.set_index(keys).index
    for c in ("yhat", "lo", "hi"):
        if all(c in f.columns for f in frames):
            out[c] = np.mean([f.set_index(keys)[c].reindex(idx).to_numpy(dtype=float) for f in frames], axis=0)
    return out


def _add_ensemble(results: dict[str, pd.DataFrame], scores: dict[str, dict], base: str):
    """Fügt das Ensemble der zwei besten Modelle (beide schlagen die Baseline) als weiteren Kandidaten hinzu.
    Gibt (results, scores, name|None) zurück."""
    good = sorted(
        (n for n, s in scores.items() if n != base and s["relative_mae"] is not None and s["relative_mae"] < 1.0),
        key=lambda n: scores[n]["relative_mae"],
    )[:2]
    if len(good) < 2:
        return results, scores, None
    name = ensemble_name(good)
    results = {**results, name: _blend([results[m] for m in good], ["unique_id", "ds", "window"])}
    scores = score_models(results, base)
    scores[name]["members"] = good
    return results, scores, name


def select_final(scores: dict[str, dict], base: str, ens: str | None) -> tuple[str, bool | None, str | None]:
    singles = {n: s for n, s in scores.items() if n != ens}
    best, validated, warn = select_model(singles, base)
    if ens and validated and scores[ens]["relative_mae"] <= singles[best]["relative_mae"] * ENSEMBLE_TOLERANCE:
        return ens, True, None
    return best, validated, warn


def _predict_final(selected: str, prep: Prepared, h: int, level: int) -> tuple[pd.DataFrame, list[str]]:
    members = ensemble_members(selected)
    if members:
        frames = [get_adapter(m).predict(prep.df, h, prep.freq, level, prep.season_length) for m in members]
        notes = [n for m in members for n in get_adapter(m).notes(level)]
        notes.append("Ensemble: Mittel der Teilprognosen; die Intervallgrenzen sind das Mittel der Grenzen (Näherung).")
        return _blend(frames, ["unique_id", "ds"]), notes
    return get_adapter(selected).predict(prep.df, h, prep.freq, level, prep.season_length), get_adapter(selected).notes(level)


# --------------------------------------------------------------------------- forecast
def run_forecast(
    records: list[dict],
    h: int,
    freq: str | None = None,
    season_length: int | None = None,
    models: list[str] | None = None,
    n_windows: int = 2,
    level: int = 80,
    ensemble: bool = True,
) -> dict:
    prep = prepare(records, freq, season_length, h)
    names = resolve_models(models)
    base = baseline_name()
    warnings: list[str] = []
    errors: dict[str, str] = {}
    scores: dict[str, dict] = {}
    skipped: list[str] = []

    with _SLOTS:
        if n_windows < 1:
            cands = [n for n in names if n != base] or [base]
            if len(cands) != 1:
                raise DataError("n_windows=0 (ohne Backtest) geht nur mit genau einem Modell")
            selected, validated = cands[0], None
            warnings.append("Kein Backtest durchgeführt: Modell nicht validiert.")
        else:
            results: dict[str, pd.DataFrame] = {}
            for n in names:
                try:
                    results[n], sk = backtest_model(n, prep, h, n_windows, level)
                    skipped = sorted(set(skipped) | set(sk))
                except DataError:
                    raise
                except Exception as e:  # ein defektes Modell darf den Lauf nicht kippen
                    log.exception("Backtest %s fehlgeschlagen", n)
                    errors[n] = f"{type(e).__name__}: {e}"
            if base not in results:
                raise RuntimeError(f"Baseline '{base}' fehlgeschlagen: {errors.get(base)}")
            scores = score_models(results, base)
            ens = None
            if ensemble:
                results, scores, ens = _add_ensemble(results, scores, base)
            selected, validated, warn = select_final(scores, base, ens)
            if warn:
                warnings.append(warn)
            if skipped:
                warnings.append(f"Für den Backtest zu kurz (nicht bewertet): {', '.join(skipped[:10])}")

        try:
            fc, notes = _predict_final(selected, prep, h, level)
        except Exception as e:
            if selected == base:
                raise
            log.exception("Endprognose %s fehlgeschlagen", selected)
            errors[selected] = f"{type(e).__name__}: {e}"
            warnings.append(f"Modell '{selected}' fiel bei der Endprognose aus; Baseline verwendet.")
            selected, validated = base, False
            fc, notes = _predict_final(base, prep, h, level)

    warnings += notes
    return {
        "selected_model": selected,
        "validated": validated,
        "baseline": base,
        "warnings": warnings,
        "backtest": scores,
        "errors": errors,
        "n_series": int(prep.df["unique_id"].nunique()),
        "freq": prep.freq,
        "season_length": prep.season_length,
        "h": h,
        "level": level,
        "forecast": _records(fc, ["unique_id", "ds", "yhat", "lo", "hi"]),
    }


def run_backtest(
    records: list[dict],
    h: int,
    freq: str | None = None,
    season_length: int | None = None,
    models: list[str] | None = None,
    n_windows: int = 3,
    level: int = 80,
    include_rows: bool = False,
    ensemble: bool = True,
) -> dict:
    """Nur der Modellvergleich, ohne Endprognose."""
    prep = prepare(records, freq, season_length, h)
    names = resolve_models(models)
    base = baseline_name()
    results, errors = {}, {}
    with _SLOTS:
        for n in names:
            try:
                results[n], _ = backtest_model(n, prep, h, max(n_windows, 1), level)
            except DataError:
                raise
            except Exception as e:
                log.exception("Backtest %s fehlgeschlagen", n)
                errors[n] = f"{type(e).__name__}: {e}"
    if base not in results:
        raise RuntimeError(f"Baseline '{base}' fehlgeschlagen: {errors.get(base)}")
    scores = score_models(results, base)
    ens = None
    if ensemble:
        results, scores, ens = _add_ensemble(results, scores, base)
    best, validated, warn = select_final(scores, base, ens)
    out = {
        "baseline": base,
        "best_model": best,
        "validated": validated,
        "warnings": [warn] if warn else [],
        "backtest": scores,
        "errors": errors,
        "freq": prep.freq,
        "season_length": prep.season_length,
        "h": h,
        "n_windows": n_windows,
    }
    if include_rows:
        out["rows"] = {
            n: _records(bt, ["unique_id", "ds", "window", "y", "yhat", "lo", "hi"])
            for n, bt in results.items()
        }
    return out


# --------------------------------------------------------------------------- anomalies
def run_anomalies(
    records: list[dict],
    model: str | None = None,
    h: int = 12,
    n_windows: int = 4,
    freq: str | None = None,
    season_length: int | None = None,
    level: int = 99,
) -> dict:
    """Punkte, die in rollierenden Out-of-sample-Fenstern außerhalb des Prognoseintervalls liegen.

    Geprüft werden die letzten n_windows*h Punkte jeder Serie (soweit die Historie reicht)."""
    prep = prepare(records, freq, season_length, h)
    name = model or baseline_name()
    adapter = get_adapter(name)
    ok, why = adapter.available()
    if not ok:
        raise DataError(f"Modell '{name}' nicht verfügbar: {why}")
    with _SLOTS:
        bt, skipped = backtest_model(name, prep, h, max(n_windows, 1), level)
    if bt["lo"].isna().any():
        raise DataError(f"Modell '{name}' liefert keine Prognoseintervalle; für Anomalien ungeeignet.")
    flag = (bt["y"] < bt["lo"]) | (bt["y"] > bt["hi"])
    an = bt[flag].sort_values(["unique_id", "ds"]).rename(columns={"yhat": "expected"})
    an = an.assign(direction=np.where(an["y"] > an["hi"], "high", "low"))
    warnings = adapter.notes(level)
    if skipped:
        warnings.append(f"Zu kurz für die Prüfung: {', '.join(skipped[:10])}")
    return {
        "model": name,
        "level": level,
        "h": h,
        "n_windows": n_windows,
        "n_points": int(len(bt)),
        "n_anomalies": int(flag.sum()),
        "warnings": warnings,
        "anomalies": _records(an, ["unique_id", "ds", "y", "expected", "lo", "hi", "direction"]),
    }
