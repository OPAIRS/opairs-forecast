"""Einheitliche Schnittstelle für alle Prognosemodelle.

Jeder Adapter bekommt ein Long-Format-DataFrame (unique_id, ds, y) und liefert
ein DataFrame (unique_id, ds, yhat, lo, hi) mit genau `h` Zeilen pro Serie.
lo/hi gehören zum Intervall `level` (z. B. 80 -> 10 %- und 90 %-Quantil).
Fehlen Intervalle, sind lo/hi NaN.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

OUT_COLS = ["unique_id", "ds", "yhat", "lo", "hi"]


@dataclass(frozen=True)
class CovSpec:
    """Kovariaten einer Anfrage. `past` = alle Kovariatenspalten in der Historie (df),
    `future` = die Teilmenge, deren Werte auch für den Prognosehorizont bekannt sind (future_df)."""

    past: tuple[str, ...]
    future: tuple[str, ...] = ()

    @property
    def past_only(self) -> tuple[str, ...]:
        return tuple(c for c in self.past if c not in self.future)


class Adapter(ABC):
    kind: str = "base"
    supports_covariates: bool = False  # kann das Modell Kovariaten nutzen (Variante `<name>@cov`)?

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
        cov: CovSpec | None = None,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Prognose für alle Serien in df.

        Ohne `cov` nutzt ein Adapter nur die Spalten unique_id, ds, y. Mit `cov` enthält df zusätzlich die
        Kovariatenspalten (cov.past) und future_df die Zukunftswerte der bekannten Kovariaten (cov.future)
        mit genau h Zeilen je Serie."""

    def available(self) -> tuple[bool, str]:
        """(ist nutzbar, Grund falls nicht). Darf nichts Schweres laden."""
        return True, ""

    def notes(self, level: int) -> list[str]:
        """Hinweise zu Einschränkungen (z. B. begrenzte Intervallniveaus)."""
        return []

    def can_use(self, cov: CovSpec) -> bool:
        """Kann dieser Adapter die Kovariaten dieser Anfrage nutzen? Standard: Modell unterstützt Kovariaten."""
        return self.supports_covariates

    def cov_notes(self, cov: CovSpec) -> list[str]:
        """Hinweise, welche Kovariaten genutzt oder ignoriert werden."""
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
