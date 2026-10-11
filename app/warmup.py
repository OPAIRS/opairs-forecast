"""Optionales Aufwärmen beim Start: lädt die Modellgewichte und kompiliert die Statistikmodelle einmalig,
damit die erste echte Anfrage nicht den Kaltstart (Sekunden bis eine halbe Minute) bezahlt."""
from __future__ import annotations

import logging
import threading
import time

from . import config

log = logging.getLogger("forecast")
_started = False
_lock = threading.Lock()


def _records(n_series: int = 4, n: int = 48) -> list[dict]:
    import math
    from datetime import date

    out = []
    for s in range(n_series):
        for i in range(n):
            out.append(
                {
                    "unique_id": f"warm{s}",
                    "ds": date(2020 + i // 12, i % 12 + 1, 1).isoformat(),
                    "y": 100 + s * 5 + i * 0.5 + 10 * math.sin(2 * math.pi * i / 12),
                }
            )
    return out


def _ensure_logging() -> None:
    """Der Logger 'forecast' hat im Container sonst keinen Handler; die Warm-up-Meldungen sollen im Log erscheinen.
    Hat der Root-Logger schon einen Handler (z. B. die MCP-Bibliothek), wird keiner ergänzt: sonst erscheint jede Zeile doppelt."""
    if not log.handlers and not logging.getLogger().handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
        log.addHandler(h)
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)


def run() -> None:
    from . import core
    from .registry import baseline_name, resolve_models

    t_all = time.time()
    recs = _records()
    names = [n for n in resolve_models(["auto"]) if n != baseline_name()] or [baseline_name()]
    for m in names:
        t0 = time.time()
        try:
            core.run_forecast(recs, 2, "MS", 12, [m], 0, 80, ensemble=False)
            log.info("Warm-up %s: %.1f s", m, time.time() - t0)
        except Exception as e:  # ein defektes Modell darf den Start nicht stören
            log.warning("Warm-up %s fehlgeschlagen: %s", m, e)
    log.info("Warm-up abgeschlossen in %.1f s (%d Modelle)", time.time() - t_all, len(names))


def start_background() -> bool:
    """Startet das Aufwärmen in einem Hintergrund-Thread (nur einmal pro Prozess, nur wenn FORECAST_WARMUP=1)."""
    global _started
    if not config.WARMUP:
        return False
    with _lock:
        if _started:
            return False
        _started = True
    _ensure_logging()
    threading.Thread(target=run, name="warmup", daemon=True).start()
    return True
