"""Eingabevalidierung: Records -> sauberes Long-Format-DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_POINTS, MAX_SERIES


class DataError(ValueError):
    pass


def to_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        raise DataError("Keine Daten übergeben")
    if len(records) > MAX_POINTS:
        raise DataError(f"Zu viele Datenpunkte ({len(records)} > {MAX_POINTS})")
    df = pd.DataFrame.from_records(records)
    missing = {"unique_id", "ds", "y"} - set(df.columns)
    if missing:
        raise DataError(f"Spalten fehlen: {', '.join(sorted(missing))} (erwartet: unique_id, ds, y)")
    df = df[["unique_id", "ds", "y"]].copy()
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
