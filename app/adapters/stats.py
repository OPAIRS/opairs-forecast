"""Klassische Statistikmodelle über statsforecast (Nixtla, Apache-2.0)."""
from __future__ import annotations

import inspect
import logging
import os
import threading

import numpy as np
import pandas as pd

from .. import config
from .base import Adapter, CovSpec, OUT_COLS

log = logging.getLogger("forecast")
_WARM: set[str] = set()
_WARM_LOCK = threading.Lock()
MIN_SERIES_FOR_PARALLEL = 4
# statsforecast-Modelle mit exogenen Regressoren (brauchen Zukunftswerte, X_df)
EXOG_CLASSES = {"AutoARIMA"}


class StatsAdapter(Adapter):
    kind = "stats"

    def __init__(self, name: str, cls: str, params: dict | None = None):
        super().__init__(name)
        self.cls = cls
        self.params = params or {}

    @property
    def supports_covariates(self) -> bool:  # type: ignore[override]
        return self.cls in EXOG_CLASSES

    def can_use(self, cov: CovSpec) -> bool:
        # Regressoren brauchen bekannte Zukunftswerte; reine Vergangenheitskovariaten kann AutoARIMA nicht nutzen
        return self.supports_covariates and bool(cov.future)

    def cov_notes(self, cov: CovSpec) -> list[str]:
        if cov.past_only:
            return [f"{self.name}: nutzt nur Kovariaten mit bekannten Zukunftswerten ({', '.join(cov.future)}); ignoriert: {', '.join(cov.past_only)}."]
        return []

    def _model_class(self):
        import statsforecast.models as sm

        if not hasattr(sm, self.cls):
            raise ValueError(f"statsforecast kennt keine Klasse '{self.cls}'")
        return getattr(sm, self.cls)

    def available(self):
        try:
            self._model_class()
            return True, ""
        except Exception as e:  # ImportError oder unbekannte Klasse
            return False, str(e)

    @staticmethod
    def _jobs(n_series: int) -> int:
        nj = config.N_JOBS
        if nj == 0 or nj == 1 or n_series < MIN_SERIES_FOR_PARALLEL:
            return 1
        if nj < 0:
            nj = os.cpu_count() or 1
        return max(1, min(nj, n_series))

    def _warm(self, cls, kwargs, freq, season_length, level):
        """Kompiliert das Modell einmalig im Hauptprozess, damit die abgespaltenen Arbeitsprozesse den
        kompilierten Code erben und nicht jeder erneut kompiliert (Numba-Kompilierung dauert Sekunden)."""
        from statsforecast import StatsForecast

        key = f"{cls.__name__}:{sorted(kwargs.items())}"
        with _WARM_LOCK:
            if key in _WARM:
                return
            try:
                n = max(3 * season_length + 8, 24)
                ds = pd.date_range("2000-01-01", periods=n, freq=freq)
                rng = np.random.default_rng(0)
                tiny = pd.concat(
                    [
                        pd.DataFrame({"unique_id": u, "ds": ds, "y": 100 + rng.normal(0, 3, n).cumsum() * 0.1 + np.sin(np.arange(n))})
                        for u in ("w1", "w2")
                    ]
                )
                sf = StatsForecast(models=[cls(**kwargs)], freq=freq, n_jobs=1)
                try:
                    sf.forecast(df=tiny, h=2, level=[level])
                except Exception:
                    sf.forecast(df=tiny, h=2)
            except Exception:
                log.warning("Aufwärmen von %s vor der Parallelisierung fehlgeschlagen", self.name, exc_info=True)
            _WARM.add(key)

    def predict(self, df, h, freq, level, season_length, cov=None, future_df=None):
        from statsforecast import StatsForecast

        cls = self._model_class()
        kwargs = dict(self.params)
        if "season_length" in inspect.signature(cls.__init__).parameters and "season_length" not in kwargs:
            kwargs["season_length"] = season_length
        model = cls(**kwargs)
        exog = list(cov.future) if (cov is not None and self.can_use(cov)) else []
        data = df[["unique_id", "ds", "y", *exog]]
        x_df = future_df[["unique_id", "ds", *exog]] if exog else None
        extra = {"X_df": x_df} if exog else {}

        def run(nj):
            sf = StatsForecast(models=[model], freq=freq, n_jobs=nj)
            try:
                return sf.forecast(df=data, h=h, level=[level], **extra)
            except Exception:
                # Modelle ohne Intervallunterstützung (z. B. Croston)
                return sf.forecast(df=data, h=h, **extra)

        nj = self._jobs(int(data["unique_id"].nunique()))
        if nj > 1:
            self._warm(cls, kwargs, freq, season_length, level)
            try:
                fc = run(nj)
            except Exception:
                log.warning("%s: parallele Ausführung (n_jobs=%d) fehlgeschlagen, wiederhole seriell", self.name, nj, exc_info=True)
                fc = run(1)
        else:
            fc = run(1)
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        alias = getattr(model, "alias", cls.__name__)
        out = fc[["unique_id", "ds"]].copy()
        out["yhat"] = fc[alias].to_numpy(dtype=float)
        lo, hi = f"{alias}-lo-{level}", f"{alias}-hi-{level}"
        out["lo"] = fc[lo].to_numpy(dtype=float) if lo in fc.columns else np.nan
        out["hi"] = fc[hi].to_numpy(dtype=float) if hi in fc.columns else np.nan
        return out[OUT_COLS]
