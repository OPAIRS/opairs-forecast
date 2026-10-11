"""Einheitliche Schnittstelle für alle Prognosemodelle.

Jeder Adapter bekommt ein Long-Format-DataFrame (unique_id, ds, y) und liefert
ein DataFrame (unique_id, ds, yhat, lo, hi) mit genau `h` Zeilen pro Serie.
lo/hi gehören zum Intervall `level` (z. B. 80 -> 10 %- und 90 %-Quantil).
Fehlen Intervalle, sind lo/hi NaN.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

OUT_COLS = ["unique_id", "ds", "yhat", "lo", "hi"]


class Adapter(ABC):
    kind: str = "base"

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: int,
        season_length: int,
    ) -> pd.DataFrame:
        """Prognose für alle Serien in df."""

    def available(self) -> tuple[bool, str]:
        """(ist nutzbar, Grund falls nicht). Darf nichts Schweres laden."""
        return True, ""

    def notes(self, level: int) -> list[str]:
        """Hinweise zu Einschränkungen (z. B. begrenzte Intervallniveaus)."""
        return []


def future_index(last_ds: pd.Timestamp, h: int, freq: str) -> pd.DatetimeIndex:
    off = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=last_ds, periods=h + 1, freq=off)[1:]


def apply_torch_threads(torch) -> None:
    """Begrenzt die Torch-Threads des aufrufenden Threads (FORECAST_TORCH_THREADS; 0 = Torch-Standard)."""
    from .. import config

    n = config.TORCH_THREADS
    if n > 0 and torch.get_num_threads() != n:
        torch.set_num_threads(n)
