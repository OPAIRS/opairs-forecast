import logging
import threading
import time

from app import config, warmup


def test_run_warms_available_models(caplog):
    caplog.set_level(logging.INFO, logger="forecast")
    warmup.run()
    text = caplog.text
    assert "Warm-up abgeschlossen" in text
    assert "Warm-up auto_ets" in text and "Warm-up theta" in text


def test_start_is_off_by_default_and_idempotent(monkeypatch):
    monkeypatch.setattr(config, "WARMUP", False)
    monkeypatch.setattr(warmup, "_started", False)
    assert warmup.start_background() is False

    calls = []
    monkeypatch.setattr(config, "WARMUP", True)
    monkeypatch.setattr(warmup, "run", lambda: calls.append(1))
    assert warmup.start_background() is True
    assert warmup.start_background() is False  # nur einmal pro Prozess
    for _ in range(50):
        if calls:
            break
        time.sleep(0.05)
    assert calls == [1]


def test_failing_model_does_not_break_warmup(monkeypatch, caplog):
    from app import core

    caplog.set_level(logging.INFO, logger="forecast")
    orig = core.run_forecast

    def boom(recs, h, freq, sl, models, *a, **k):
        if models == ["theta"]:
            raise RuntimeError("kaputt")
        return orig(recs, h, freq, sl, models, *a, **k)

    monkeypatch.setattr(core, "run_forecast", boom)
    warmup.run()
    assert "Warm-up theta fehlgeschlagen" in caplog.text and "Warm-up abgeschlossen" in caplog.text
