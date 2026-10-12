"""Chronos-Familie (Amazon, Apache-2.0) – lokal geladen, kein Netzwerk nötig.

Ein Adapter für Chronos-T5, Chronos-Bolt und Chronos-2: alle Pipelines teilen
`predict_quantiles(inputs, prediction_length, quantile_levels)`.
`path` zeigt auf ein lokales Modellverzeichnis (huggingface-cli download ...).
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd

from .base import apply_torch_threads, Adapter, CovSpec, OUT_COLS, future_index

# Quantilbereich, den das jeweilige Modell trainiert hat; außerhalb wird begrenzt.
_BOLT_RANGE = (0.1, 0.9)


class ChronosAdapter(Adapter):
    kind = "chronos"

    @property
    def supports_covariates(self) -> bool:  # type: ignore[override]
        return self.family == "chronos2"  # nur Chronos-2 kennt Kovariaten (Past und Future)

    def cov_notes(self, cov: CovSpec) -> list[str]:
        return []

    def __init__(self, name: str, path: str, device: str = "auto", batch_size: int = 64, family: str = "bolt"):
        super().__init__(name)
        self.path = path
        self.device = device
        self.batch_size = batch_size
        self.family = family  # "bolt" | "t5" | "chronos2" – nur für Hinweise zu Intervallen
        self._pipe = None
        self._lock = threading.Lock()

    def available(self):
        if not os.path.isdir(self.path):
            return False, f"Modellverzeichnis fehlt: {self.path}"
        try:
            import chronos  # noqa: F401
        except Exception as e:
            return False, f"chronos-forecasting nicht installiert: {e}"
        return True, ""

    def _load(self):
        with self._lock:
            if self._pipe is None:
                import torch
                from chronos import BaseChronosPipeline

                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                # path ist ein lokales Verzeichnis; HF_HUB_OFFLINE=1 verhindert jeden Netzzugriff
                self._pipe = BaseChronosPipeline.from_pretrained(self.path, device_map=device)
        return self._pipe

    def notes(self, level):
        if self.family in ("bolt", "t5") and level > 80:
            return [
                f"{self.name}: Intervall {level} % liegt außerhalb des trainierten Quantilbereichs "
                f"(10–90 %); lo/hi sind auf dieses Niveau begrenzt und eher zu eng."
            ]
        return []

    def predict(self, df, h, freq, level, season_length, cov=None, future_df=None):
        import torch

        apply_torch_threads(torch)
        pipe = self._load()
        alpha = (1 - level / 100) / 2
        qs = [round(alpha, 4), 0.5, round(1 - alpha, 4)]
        use_cov = cov is not None and self.supports_covariates
        fut_by_uid = {}
        if use_cov and cov.future:
            fut_by_uid = {str(u): g.sort_values("ds") for u, g in future_df.groupby("unique_id", sort=False)}
        ids, contexts = [], []
        for uid, g in df.groupby("unique_id", sort=False):
            ids.append(uid)
            y = g["y"].to_numpy(dtype=np.float32)
            if use_cov:
                # Chronos-2: past_covariates = alle Kovariaten über die Historie (auch die zukunftsbekannten),
                # future_covariates = deren Werte für den Horizont (Schlüssel sind eine Teilmenge der past-Schlüssel)
                item = {"target": y, "past_covariates": {c: g[c].to_numpy(dtype=np.float32) for c in cov.past}}
                if cov.future:
                    fg = fut_by_uid[str(uid)]
                    item["future_covariates"] = {c: fg[c].to_numpy(dtype=np.float32) for c in cov.future}
                contexts.append(item)
            else:
                contexts.append(torch.tensor(y))
        frames = []
        for start in range(0, len(contexts), self.batch_size):
            chunk = contexts[start : start + self.batch_size]
            quantiles, _ = pipe.predict_quantiles(inputs=chunk, prediction_length=h, quantile_levels=qs)
            if isinstance(quantiles, (list, tuple)):  # Chronos-2: Liste von (n_variates, h, q)
                arr = np.stack([q[0].cpu().numpy() for q in quantiles])
            else:
                arr = quantiles.cpu().numpy()
            for i, uid in enumerate(ids[start : start + self.batch_size]):
                last = df.loc[df["unique_id"] == uid, "ds"].iloc[-1]
                frames.append(
                    pd.DataFrame(
                        {
                            "unique_id": uid,
                            "ds": future_index(last, h, freq),
                            "yhat": arr[i, :, 1].astype(float),
                            "lo": arr[i, :, 0].astype(float),
                            "hi": arr[i, :, 2].astype(float),
                        }
                    )
                )
        return pd.concat(frames, ignore_index=True)[OUT_COLS]
