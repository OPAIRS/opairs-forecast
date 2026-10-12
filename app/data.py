"""Eingabevalidierung: Records -> sauberes Long-Format-DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_POINTS, MAX_SERIES


class DataError(ValueError):
    pass


def to_frame(records: list[dict], covariates: list[str] | None = None) -> pd.DataFrame:
    """Records -> Long-Format. `covariates` benennt zusätzliche numerische Spalten (Kovariaten), die erhalten bleiben."""
    if not records:
        raise DataError("Keine Daten übergeben")
    if len(records) > MAX_POINTS:
        raise DataError(f"Zu viele Datenpunkte ({len(records)} > {MAX_POINTS})")
    df = pd.DataFrame.from_records(records)
    missing = {"unique_id", "ds", "y"} - set(df.columns)
    if missing:
        raise DataError(f"Spalten fehlen: {', '.join(sorted(missing))} (erwartet: unique_id, ds, y)")
    covs = list(dict.fromkeys(covariates or []))
    clash = [c for c in covs if c in ("unique_id", "ds", "y")]
    if clash:
        raise DataError(f"Kovariate darf nicht heißen: {', '.join(clash)}")
    absent = [c for c in covs if c not in df.columns]
    if absent:
        raise DataError(f"Kovariatenspalten fehlen in series: {', '.join(absent)}")
    df = df[["unique_id", "ds", "y", *covs]].copy()
    for c in covs:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any() or not np.isfinite(df[c].to_numpy()).all():
            raise DataError(f"Kovariate '{c}' hat fehlende, nicht numerische oder unendliche Werte (nur Zahlen ohne Lücken)")
    df["unique_id"] = df["unique_id"].astype(str)
    try:
        df["ds"] = pd.to_datetime(df["ds"])
    except Exception as e:
        raise DataError(f"Spalte ds nicht als Datum lesbar: {e}") from e
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if df["y"].isna().any():
        bad = df.loc[df["y"].isna(), "unique_id"].unique()[:5]
        raise DataError(f"Fehlende oder nicht numerische y-Werte (z. B. in: {', '.join(bad)})")
    if not np.isfinite(df["y"].to_numpy()).all():
        raise DataError("y enthält unendliche Werte")
    if df["unique_id"].nunique() > MAX_SERIES:
        raise DataError(f"Zu viele Serien ({df['unique_id'].nunique()} > {MAX_SERIES})")
    if df.duplicated(["unique_id", "ds"]).any():
        d = df[df.duplicated(["unique_id", "ds"], keep=False)].iloc[0]
        raise DataError(f"Doppelter Zeitstempel in Serie '{d['unique_id']}' ({d['ds']:%Y-%m-%d})")
    return df.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)


def infer_freq(df: pd.DataFrame, freq: str | None) -> str:
    """Gemeinsame Frequenz bestimmen und Lücken melden (keine stille Reparatur)."""
    if freq:
        try:
            pd.tseries.frequencies.to_offset(freq)
        except Exception as e:
            raise DataError(f"Ungültige Frequenz '{freq}'") from e
        found = freq
    else:
        freqs = {}
        for uid, g in df.groupby("unique_id", sort=False):
            if len(g) < 3:
                raise DataError(f"Serie '{uid}' hat weniger als 3 Punkte, Frequenz nicht ableitbar (freq angeben)")
            f = pd.infer_freq(g["ds"])
            if f is None:
                raise DataError(
                    f"Serie '{uid}': unregelmäßige Zeitstempel oder Lücken. freq angeben oder Lücken füllen."
                )
            freqs[uid] = f
        uniq = set(freqs.values())
        if len(uniq) > 1:
            raise DataError(f"Serien haben unterschiedliche Frequenzen: {sorted(uniq)}")
        found = uniq.pop()
    # Lückenprüfung gegen die Frequenz
    off = pd.tseries.frequencies.to_offset(found)
    for uid, g in df.groupby("unique_id", sort=False):
        expected = pd.date_range(g["ds"].iloc[0], g["ds"].iloc[-1], freq=off)
        if len(expected) < len(g):
            raise DataError(
                f"Serie '{uid}': Zeitstempel liegen nicht auf dem Raster von freq={found} "
                f"(erster Wert {g['ds'].iloc[0]:%Y-%m-%d}, {g['ds'].iloc[0]:%A}). "
                "freq weglassen (wird abgeleitet) oder passend setzen, z. B. W-MON statt W."
            )
        if len(expected) != len(g):
            raise DataError(
                f"Serie '{uid}' hat Lücken ({len(expected) - len(g)} fehlende Perioden bei freq={found}). "
                "Lücken vor dem Aufruf füllen."
            )
    return found


def default_season_length(freq: str) -> int:
    off = pd.tseries.frequencies.to_offset(freq)
    n = max(int(off.n), 1)
    o = pd.offsets
    if isinstance(off, o.Hour):
        return max(24 // n, 1)
    if isinstance(off, o.Minute):
        return max(1440 // n, 1)
    if isinstance(off, o.Second):
        return max(86400 // n, 1)
    if isinstance(off, o.BusinessDay):
        return 5
    if isinstance(off, o.Day):
        return max(7 // n, 1)
    if isinstance(off, o.Week):
        return max(52 // n, 1)
    if isinstance(off, (o.MonthBegin, o.MonthEnd)):
        return max(12 // n, 1)
    if isinstance(off, (o.QuarterBegin, o.QuarterEnd)):
        return max(4 // n, 1)
    return 1


# Grenzwert nach Syntetos/Boylan: durchschnittlicher Abstand zwischen Nachfragen (ADI) ab 1,32 gilt als intermittierend.
INTERMITTENT_ADI = 1.32


def intermittency(df: pd.DataFrame) -> dict[str, dict]:
    """Je Serie: ADI (Perioden pro Nachfrage), CV² der Nachfragemengen und ob die Serie intermittierend ist.

    Intermittierend heißt hier: nicht negativ und ADI >= 1,32 (mindestens etwa ein Viertel Nullen).
    Typisch für Ersatzteile und C-Teile. Für solche Reihen belohnt der MAE Nullprognosen, siehe core.score_models."""
    out: dict[str, dict] = {}
    for uid, g in df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        nz = y[y != 0]
        adi = float(len(y) / len(nz)) if len(nz) else float("inf")
        cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 and nz.mean() != 0 else 0.0
        out[str(uid)] = {
            "adi": adi,
            "cv2": cv2,
            "intermittent": bool((y >= 0).all() and adi >= INTERMITTENT_ADI),
        }
    return out


def to_future_frame(records: list[dict], covariates: list[str], df: pd.DataFrame, freq: str, h: int) -> pd.DataFrame:
    """Zukunftswerte bekannter Kovariaten prüfen: je Serie genau h Zeilen auf dem Raster direkt nach dem letzten ds.

    Nur Spalten aus `covariates` sind erlaubt. Gibt (unique_id, ds, <Spalten>) zurück."""
    if not records:
        raise DataError("future_covariates ist leer")
    fut = pd.DataFrame.from_records(records)
    if not {"unique_id", "ds"} <= set(fut.columns):
        raise DataError("future_covariates braucht die Spalten unique_id und ds")
    cols = [c for c in fut.columns if c not in ("unique_id", "ds")]
    if not cols:
        raise DataError("future_covariates enthält keine Kovariatenspalten")
    unknown = [c for c in cols if c not in covariates]
    if unknown:
        raise DataError(f"future_covariates enthält Spalten, die nicht in covariates stehen: {', '.join(unknown)}")
    fut["unique_id"] = fut["unique_id"].astype(str)
    try:
        fut["ds"] = pd.to_datetime(fut["ds"])
    except Exception as e:
        raise DataError(f"future_covariates: ds nicht als Datum lesbar: {e}") from e
    for c in cols:
        fut[c] = pd.to_numeric(fut[c], errors="coerce")
        if fut[c].isna().any() or not np.isfinite(fut[c].to_numpy()).all():
            raise DataError(f"future_covariates: '{c}' hat fehlende, nicht numerische oder unendliche Werte")
    if fut.duplicated(["unique_id", "ds"]).any():
        raise DataError("future_covariates enthält doppelte Zeitstempel")
    off = pd.tseries.frequencies.to_offset(freq)
    parts = []
    for uid, g in df.groupby("unique_id", sort=False):
        last = g["ds"].iloc[-1]
        want = pd.date_range(start=last, periods=h + 1, freq=off)[1:]
        have = fut[fut["unique_id"] == str(uid)].sort_values("ds")
        if len(have) != h or not (have["ds"].to_numpy() == want.to_numpy()).all():
            raise DataError(
                f"future_covariates für Serie '{uid}': erwartet genau {h} Zeilen von {want[0]:%Y-%m-%d} bis {want[-1]:%Y-%m-%d} "
                f"(gefunden: {len(have)})"
            )
        parts.append(have[["unique_id", "ds", *cols]])
    extra = set(fut["unique_id"]) - set(df["unique_id"].astype(str))
    if extra:
        raise DataError(f"future_covariates enthält unbekannte Serien: {', '.join(sorted(extra)[:5])}")
    return pd.concat(parts, ignore_index=True)
