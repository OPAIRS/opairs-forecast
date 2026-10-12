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
from .adapters.base import CovSpec
from .data import DataError, default_season_length, infer_freq, intermittency, to_frame, to_future_frame
from .registry import baseline_name, get_adapter, resolve_models, split_name

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
    intermittent: frozenset = frozenset()  # Serien mit sporadischer Nachfrage (ADI >= 1,32)
    cov: CovSpec | None = None  # Kovariaten (Spalten in df); cov.future = auch für den Horizont bekannt
    future_df: pd.DataFrame | None = None  # Zukunftswerte der bekannten Kovariaten, h Zeilen je Serie


def prepare(
    records: list[dict],
    freq: str | None,
    season_length: int | None,
    h: int,
    covariates: list[str] | None = None,
    future_records: list[dict] | None = None,
) -> Prepared:
    if not 1 <= h <= MAX_HORIZON:
        raise DataError(f"h muss zwischen 1 und {MAX_HORIZON} liegen")
    df = to_frame(records, covariates)
    f = infer_freq(df, freq)
    sl = season_length or default_season_length(f)
    if sl < 1:
        raise DataError("season_length muss >= 1 sein")
    inter = frozenset(u for u, v in intermittency(df).items() if v["intermittent"])
    cov, fut = None, None
    if covariates:
        covs = tuple(dict.fromkeys(covariates))
        future_cols: tuple[str, ...] = ()
        if future_records:
            fut = to_future_frame(future_records, list(covs), df, f, h)
            future_cols = tuple(c for c in fut.columns if c not in ("unique_id", "ds"))
        cov = CovSpec(covs, future_cols)
    elif future_records:
        raise DataError("future_covariates ohne covariates: die Kovariatenspalten in covariates benennen")
    return Prepared(df, f, sl, intermittent=inter, cov=cov, future_df=fut)


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
    base_name, use_cov = split_name(name)
    adapter = get_adapter(base_name)
    min_train = max(2 * prep.season_length, 8)
    wins, skipped = _windows(prep.df, h, n_windows, min_train)
    if not wins:
        raise DataError(
            f"Keine Serie lang genug für Backtest (nötig: mind. {min_train + h} Punkte bei h={h}, "
            f"season_length={prep.season_length}). h oder n_windows verkleinern."
        )
    frames = []
    for j, (train, test) in wins.items():
        if use_cov:
            fut = test[["unique_id", "ds", *prep.cov.future]] if prep.cov.future else None
            pred = adapter.predict(train, h, prep.freq, level, prep.season_length, cov=prep.cov, future_df=fut)
        else:
            pred = adapter.predict(train[["unique_id", "ds", "y"]], h, prep.freq, level, prep.season_length)
        m = test[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
        if m["yhat"].isna().any():
            raise RuntimeError(f"{name}: Prognose unvollständig (Fenster {j})")
        m["window"] = j
        frames.append(m)
    return pd.concat(frames, ignore_index=True), skipped


def _mae_by_series(bt: pd.DataFrame) -> pd.Series:
    return (bt["y"] - bt["yhat"]).abs().groupby(bt["unique_id"]).mean()


def _rmse_by_series(bt: pd.DataFrame) -> pd.Series:
    return np.sqrt(((bt["y"] - bt["yhat"]) ** 2).groupby(bt["unique_id"]).mean())


def _error_by_series(bt: pd.DataFrame, intermittent: frozenset = frozenset()) -> pd.Series:
    """Fehler je Serie: MAE, bei intermittierenden Serien RMSE. Bei sporadischer Nachfrage ist die MAE-optimale
    Prognose die Null (der Median); der MAE würde Nullprognosen gegenüber Croston & Co. belohnen."""
    mae = _mae_by_series(bt)
    if not intermittent:
        return mae
    rmse = _rmse_by_series(bt)
    return pd.Series({u: (rmse[u] if u in intermittent else mae[u]) for u in mae.index})


def score_models(results: dict[str, pd.DataFrame], baseline: str, intermittent: frozenset = frozenset()) -> dict[str, dict]:
    """Relative MAE gegen die Baseline (pro Serie, dann gemittelt). <1 heißt besser als Baseline.

    Für intermittierende Serien wird statt des MAE der RMSE verglichen (Feld bleibt aus Kompatibilität `relative_mae`).
    Trifft die Baseline eine Serie praktisch perfekt (Fehler ~ 0), kann kein Modell sie schlagen:
    gleich perfekt -> Verhältnis 1, sonst sehr großes Verhältnis."""
    base_bt = results[baseline]
    base_mae = _error_by_series(base_bt, intermittent)
    floor = 1e-9 * base_bt["y"].abs().groupby(base_bt["unique_id"]).mean() + 1e-12
    scores = {}
    for name, bt in results.items():
        mae = _error_by_series(bt, intermittent)
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
            "mae": float(_mae_by_series(bt).mean()),
            "relative_mae": rel,
            "beats_baseline": bool(rel is not None and rel < 1.0 and name != baseline),
            "interval_coverage": cov,
            "n_series": int(len(mae)),
        }
        if intermittent:
            scores[name]["n_intermittent"] = int(len(intermittent & set(mae.index)))
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


def _add_ensemble(results: dict[str, pd.DataFrame], scores: dict[str, dict], base: str, intermittent: frozenset = frozenset()):
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
    scores = score_models(results, base, intermittent)
    scores[name]["members"] = good
    return results, scores, name


def select_final(scores: dict[str, dict], base: str, ens: str | None) -> tuple[str, bool | None, str | None]:
    singles = {n: s for n, s in scores.items() if n != ens}
    best, validated, warn = select_model(singles, base)
    if ens and validated and scores[ens]["relative_mae"] <= singles[best]["relative_mae"] * ENSEMBLE_TOLERANCE:
        return ens, True, None
    return best, validated, warn


def _predict_one(name: str, prep: Prepared, h: int, level: int) -> tuple[pd.DataFrame, list[str]]:
    base_name, use_cov = split_name(name)
    ad = get_adapter(base_name)
    if use_cov:
        fc = ad.predict(prep.df, h, prep.freq, level, prep.season_length, cov=prep.cov, future_df=prep.future_df)
        return fc, ad.notes(level) + ad.cov_notes(prep.cov)
    return ad.predict(prep.df[["unique_id", "ds", "y"]], h, prep.freq, level, prep.season_length), ad.notes(level)


def _predict_final(selected: str, prep: Prepared, h: int, level: int) -> tuple[pd.DataFrame, list[str]]:
    members = ensemble_members(selected)
    if members:
        parts = [_predict_one(m, prep, h, level) for m in members]
        notes = [n for _, ns in parts for n in ns]
        notes.append("Ensemble: Mittel der Teilprognosen; die Intervallgrenzen sind das Mittel der Grenzen (Näherung).")
        return _blend([f for f, _ in parts], ["unique_id", "ds"]), notes
    return _predict_one(selected, prep, h, level)


def _intermittent_warnings(prep: Prepared, n_windows: int) -> list[str]:
    if not prep.intermittent:
        return []
    ids = sorted(prep.intermittent)
    msg = f"{len(ids)} von {prep.df['unique_id'].nunique()} Serien sind intermittierend (viele Nullen), z. B. {', '.join(ids[:5])}."
    if n_windows >= 1:
        msg += " Bei auto kommen Croston-Verfahren (SBA, TSB, IMAPA, ADIDA) dazu; für diese Serien wird der RMSE statt des MAE verglichen."
    return [msg]


def _cov_info(prep: Prepared) -> dict | None:
    if prep.cov is None:
        return None
    return {"future_known": list(prep.cov.future), "past_only": list(prep.cov.past_only)}


def _cov_warnings(prep: Prepared, names: list[str], selected: str | None = None, validated: bool | None = None) -> list[str]:
    if prep.cov is None:
        return []
    out = []
    if not any(split_name(n)[1] for n in names):
        out.append(
            "Kovariaten übergeben, aber kein Modell der Auswahl kann sie nutzen (z. B. chronos_2, tirex_2 oder auto_arima mit "
            "bekannten Zukunftswerten). Die Prognose nutzt nur die Zielreihe."
        )
    elif selected is not None and validated and "@cov" not in selected:
        out.append("Die Kovariaten verbessern den Backtest nicht: das gewählte Modell nutzt nur die Zielreihe.")
    if not prep.cov.future:
        out.append(
            "Keine Zukunftswerte übergeben (future_covariates): Alle Kovariaten gelten als nur in der Vergangenheit bekannt."
        )
    return out


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
    covariates: list[str] | None = None,
    future_covariates: list[dict] | None = None,
) -> dict:
    prep = prepare(records, freq, season_length, h, covariates, future_covariates)
    names = resolve_models(models, intermittent=bool(prep.intermittent), cov=prep.cov)
    base = baseline_name()
    warnings: list[str] = _intermittent_warnings(prep, n_windows)
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
            warnings += _cov_warnings(prep, names)
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
            scores = score_models(results, base, prep.intermittent)
            ens = None
            if ensemble:
                results, scores, ens = _add_ensemble(results, scores, base, prep.intermittent)
            selected, validated, warn = select_final(scores, base, ens)
            warnings += _cov_warnings(prep, names, selected, validated)
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
        "covariates": _cov_info(prep),
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
    covariates: list[str] | None = None,
    future_covariates: list[dict] | None = None,
) -> dict:
    """Nur der Modellvergleich, ohne Endprognose. Mit Kovariaten: jedes fähige Modell mit (`@cov`) und ohne."""
    prep = prepare(records, freq, season_length, h, covariates, future_covariates)
    names = resolve_models(models, intermittent=bool(prep.intermittent), cov=prep.cov)
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
    scores = score_models(results, base, prep.intermittent)
    ens = None
    if ensemble:
        results, scores, ens = _add_ensemble(results, scores, base, prep.intermittent)
    best, validated, warn = select_final(scores, base, ens)
    out = {
        "baseline": base,
        "best_model": best,
        "validated": validated,
        "warnings": _intermittent_warnings(prep, n_windows) + _cov_warnings(prep, names, best, validated) + ([warn] if warn else []),
        "covariates": _cov_info(prep),
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
