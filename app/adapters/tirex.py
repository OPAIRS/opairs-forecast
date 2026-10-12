"""TiRex-2 (NXAI, Apache-2.0) – lokal geladen, kein Netzwerk nötig.

API laut Paket `tirex-2` (Import `tirex2`): `load_model(dir, device)` und
`model.forecast([TimeseriesType], prediction_length, output_type="numpy")`.
Je Serie kommt ein Array (n_targets, n_quantiles, h). Die Quantilniveaus stehen im Modell
(`model.quantiles`) und werden zur Laufzeit gelesen, nicht angenommen.
`path` = lokales Verzeichnis mit `model-config.yaml` und `model.ckpt`.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd

from .base import apply_torch_threads, Adapter, CovSpec, OUT_COLS, future_index


def pick_quantile(values: np.ndarray, native: list[float], level: float) -> np.ndarray:
    """values: (Q, h). Linear zwischen nativen Quantilen interpoliert, außerhalb auf den Rand begrenzt."""
    nat = np.asarray(native, dtype=float)
    if level <= nat[0]:
        return values[0]
    if level >= nat[-1]:
        return values[-1]
    hi = int(np.searchsorted(nat, level))
    lo = hi - 1
    w = (level - nat[lo]) / (nat[hi] - nat[lo])
    return (1 - w) * values[lo] + w * values[hi]


class TiRex2Adapter(Adapter):
    kind = "tirex2"
    supports_covariates = True  # TiRex-2: Past- und Future-Kovariaten (Paket tirex-2, TimeseriesType)

    def __init__(self, name: str, path: str, device: str = "auto", batch_size: int = 256):
        super().__init__(name)
        self.path = path
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._native: list[float] = [0.1, 0.9]  # bis zum Laden Annahme; danach echte Werte
        self._lock = threading.Lock()

    def available(self):
        for f in ("model-config.yaml", "model.ckpt"):
            if not os.path.isfile(os.path.join(self.path, f)):
                return False, f"{f} fehlt in {self.path}"
        try:
            import tirex2  # noqa: F401
        except Exception as e:
            return False, f"tirex-2 nicht installiert: {e}"
        return True, ""

    def _load(self):
        with self._lock:
            if self._model is None:
                import torch
                from tirex2 import load_model

                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                # device="cuda" baut beim ersten Lauf einen Kernel mit nvcc (CUDA Toolkit nötig, Compute Capability >= 8.0)
                model = load_model(self.path, device=device, hf_kwargs={"local_files_only": True})
                self._native = [round(float(q), 6) for q in model.quantiles]
                self._model = model
        return self._model

    def notes(self, level):
        alpha = (1 - level / 100) / 2
        if alpha < self._native[0] - 1e-9:
            return [
                f"{self.name}: Intervall {level} % liegt außerhalb der Modellquantile "
                f"({self._native[0]:g}–{self._native[-1]:g}); lo/hi sind auf diesen Bereich begrenzt und eher zu eng."
            ]
        return []

    def predict(self, df, h, freq, level, season_length, cov=None, future_df=None):
        import torch
        from tirex2 import TimeseriesType

        apply_torch_threads(torch)
        model = self._load()
        alpha = (1 - level / 100) / 2
        fut_by_uid = {}
        if cov is not None and cov.future:
            fut_by_uid = {str(u): g.sort_values("ds") for u, g in future_df.groupby("unique_id", sort=False)}
        ids, series = [], []
        for uid, g in df.groupby("unique_id", sort=False):
            ids.append(uid)
            y = torch.tensor(g["y"].to_numpy(dtype=np.float32))[None, :]  # (1, T): univariat
            past_cov = fut_cov = None
            if cov is not None:
                # past_covariates [V_p, T]: nur in der Vergangenheit bekannt. future_covariates [V_f, T+h]:
                # zukunftsbekannte Kovariaten über Historie UND Horizont (Historienwerte aus df, Horizontwerte aus future_df).
                if cov.past_only:
                    past_cov = torch.tensor(np.stack([g[c].to_numpy(dtype=np.float32) for c in cov.past_only]))
                if cov.future:
                    fg = fut_by_uid[str(uid)]
                    fut_cov = torch.tensor(
                        np.stack(
                            [np.concatenate([g[c].to_numpy(dtype=np.float32), fg[c].to_numpy(dtype=np.float32)]) for c in cov.future]
                        )
                    )
            series.append(TimeseriesType(target=y, past_covariates=past_cov, future_covariates=fut_cov))
        forecasts = model.forecast(series, prediction_length=h, output_type="numpy", batch_size=self.batch_size)
        frames = []
        for uid, fc in zip(ids, forecasts):
            q = np.asarray(fc)[0]  # (n_quantiles, h)
            last = df.loc[df["unique_id"] == uid, "ds"].iloc[-1]
            frames.append(
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": future_index(last, h, freq),
                        "yhat": pick_quantile(q, self._native, 0.5).astype(float),
                        "lo": pick_quantile(q, self._native, alpha).astype(float),
                        "hi": pick_quantile(q, self._native, 1 - alpha).astype(float),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)[OUT_COLS]
