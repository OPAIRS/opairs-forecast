"""Timestral (Mistral AI) – lokal geladen, kein Netzwerk nötig.

Code des Pakets `timestral`: Apache-2.0. Die Modellgewichte stehen auf Hugging Face unter einer "Modified MIT license"
(Lizenztext vor dem Einsatz lesen, siehe THIRD_PARTY.md).

API laut Paket (Commit a534bdb): `TimestralForecaster.from_pretrained(dir, device=..., local_files_only=True)` und
`forecaster.forecast(context (B, V, T), prediction_length, past_covariates (B, C, T), known_covariates (B, C, T + h))`.
Ergebnis: `Forecast.quantiles` mit Form (B, V, h, Q) und `quantile_levels` (Standard: 21 Niveaus von 0,01 bis 0,99).
NaN markiert fehlende Werte, deshalb werden Reihen unterschiedlicher Länge links mit NaN aufgefüllt.
`path` = lokales Verzeichnis mit `config.json` und `model.safetensors`.

Nicht jeder Checkpoint kann Kovariaten oder Known-Future-Kovariaten; das steht in der `config.json` und wird
zur Laufzeit gelesen, nicht angenommen.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd

from .base import apply_torch_threads, Adapter, CovSpec, OUT_COLS, future_index
from .tirex import pick_quantile

_DEFAULT_LEVELS = [0.01, 0.99]  # bis zum Laden Annahme; danach echte Niveaus


class TimestralAdapter(Adapter):
    kind = "timestral"

    def __init__(
        self,
        name: str,
        path: str,
        device: str = "auto",
        batch_size: int = 32,
        symmetric_averaging: bool = True,
    ):
        super().__init__(name)
        self.path = path
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.symmetric_averaging = symmetric_averaging
        self._model = None
        self._cfg = None
        self._native: list[float] = list(_DEFAULT_LEVELS)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ Fähigkeiten
    def _config(self):
        """TimestralConfig aus config.json (leicht, ohne Gewichte). None, wenn nicht lesbar."""
        if self._cfg is None:
            try:
                from timestral.config import TimestralConfig

                self._cfg = TimestralConfig.load(os.path.join(self.path, "config.json"))
                self._native = [float(q) for q in self._cfg.quantile_knots]
            except Exception:
                return None
        return self._cfg

    @property
    def supports_covariates(self) -> bool:  # type: ignore[override]
        cfg = self._config()
        return bool(cfg and cfg.covariate_enabled)

    @property
    def supports_known(self) -> bool:
        cfg = self._config()
        return bool(cfg and cfg.supports_known_covariates)

    def can_use(self, cov: CovSpec) -> bool:
        return self.supports_covariates

    def cov_notes(self, cov: CovSpec) -> list[str]:
        if cov.future and not self.supports_known:
            return [f"{self.name}: dieser Checkpoint kann keine zukünftig bekannten Kovariaten; "
                    f"{', '.join(cov.future)} nur als Vergangenheitskovariate(n) genutzt."]
        return []

    def available(self):
        for f in ("config.json", "model.safetensors"):
            if not os.path.isfile(os.path.join(self.path, f)):
                return False, f"{f} fehlt in {self.path}"
        try:
            import timestral  # noqa: F401
        except Exception as e:
            return False, f"timestral nicht installiert: {e}"
        return True, ""

    def notes(self, level):
        self._config()
        alpha = (1 - level / 100) / 2
        if alpha < self._native[0] - 1e-9:
            return [
                f"{self.name}: Intervall {level} % liegt außerhalb der Modellquantile "
                f"({self._native[0]:g}–{self._native[-1]:g}); lo/hi sind auf diesen Bereich begrenzt und eher zu eng."
            ]
        return []

    # ------------------------------------------------------------------ Laden
    def _load(self):
        with self._lock:
            if self._model is None:
                import torch
                from timestral import TimestralForecaster

                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                model = TimestralForecaster.from_pretrained(self.path, device=device, local_files_only=True)
                self._native = [float(q) for q in model.quantile_levels]
                self._model = model
        return self._model

    # ------------------------------------------------------------------ Prognose
    def predict(self, df, h, freq, level, season_length, cov=None, future_df=None):
        import torch

        apply_torch_threads(torch)
        model = self._load()
        alpha = (1 - level / 100) / 2
        ctx_len = int(getattr(model, "max_context", 0) or 0) or None

        use_cov = cov is not None and self.supports_covariates
        known_cols = list(cov.future) if (use_cov and self.supports_known) else []
        past_cols = [c for c in (cov.past if use_cov else ()) if c not in known_cols]
        fut_by_uid = {}
        if known_cols:
            fut_by_uid = {str(u): g.sort_values("ds") for u, g in future_df.groupby("unique_id", sort=False)}

        items = []  # (uid, last_ds, y (T,), past (P, T), known (K, T + h))
        for uid, g in df.groupby("unique_id", sort=False):
            g = g.sort_values("ds")
            last = g["ds"].iloc[-1]
            y = g["y"].to_numpy(dtype=np.float32)
            past = np.stack([g[c].to_numpy(dtype=np.float32) for c in past_cols]) if past_cols else None
            known = None
            if known_cols:
                fg = fut_by_uid[str(uid)]
                known = np.stack(
                    [np.concatenate([g[c].to_numpy(dtype=np.float32), fg[c].to_numpy(dtype=np.float32)]) for c in known_cols]
                )
            if ctx_len and len(y) > ctx_len:  # nur die jüngsten Schritte werden gelesen
                y = y[-ctx_len:]
                past = past[:, -ctx_len:] if past is not None else None
                known = np.concatenate([known[:, len(g) - ctx_len : len(g)], known[:, len(g):]], axis=1) if known is not None else None
            items.append((uid, last, y, past, known))

        # ähnlich lange Reihen zusammen, um Auffüllen zu sparen; die Reihenfolge der Ausgabe bleibt die der Eingabe
        order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
        result: dict[int, pd.DataFrame] = {}
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            t_max = max(len(items[i][2]) for i in idx)
            ctx = np.full((len(idx), 1, t_max), np.nan, dtype=np.float32)
            pc = np.full((len(idx), len(past_cols), t_max), np.nan, dtype=np.float32) if past_cols else None
            kc = np.full((len(idx), len(known_cols), t_max + h), np.nan, dtype=np.float32) if known_cols else None
            for b, i in enumerate(idx):
                _, _, y, past, known = items[i]
                t = len(y)
                ctx[b, 0, t_max - t :] = y
                if pc is not None:
                    pc[b, :, t_max - t :] = past
                if kc is not None:
                    kc[b, :, t_max - t : t_max] = known[:, :t]
                    kc[b, :, t_max:] = known[:, t:]
            fc = model.forecast(
                ctx,
                prediction_length=h,
                past_covariates=pc,
                known_covariates=kc,
                symmetric_averaging=self.symmetric_averaging,
            )
            q = fc.quantiles
            q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)  # (n, 1, h, Q)
            levels = [float(x) for x in fc.quantile_levels]
            for b, i in enumerate(idx):
                vals = q[b, 0].T  # (Q, h)
                uid, last = items[i][0], items[i][1]
                result[i] = pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": future_index(last, h, freq),
                        "yhat": pick_quantile(vals, levels, 0.5).astype(float),
                        "lo": pick_quantile(vals, levels, alpha).astype(float),
                        "hi": pick_quantile(vals, levels, 1 - alpha).astype(float),
                    }
                )
        return pd.concat([result[i] for i in range(len(items))], ignore_index=True)[OUT_COLS]
