"""Optionale Erklärung per LLM (OpenAI-kompatibler Endpunkt, z. B. LiteLLM).

Bewusst getrennt vom Kern: Zahlen kommen nur aus Backtest/Prognose, das LLM formuliert nur.
Ohne LLM_MODEL ist der Endpunkt aus."""
from __future__ import annotations

import json

import httpx
import numpy as np

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT
from .data import to_frame

SYSTEM = (
    "Du bist ein sachlicher Analyst für Zeitreihenprognosen in der Fertigung. "
    "Erkläre das Ergebnis auf Deutsch in höchstens 8 Sätzen. Verwende ausschließlich die gelieferten Zahlen, "
    "erfinde keine Ursachen. Wenn validated=false oder null ist, sage das am Anfang deutlich "
    "(Prognose schlägt die einfache Baseline nicht bzw. wurde nicht geprüft). Nenne das gewählte Modell, "
    "wie viel besser/schlechter es als die Baseline war (relative_mae), die Richtung der Prognose je Serie "
    "und die Unsicherheit (Intervall)."
)


def enabled() -> bool:
    return bool(LLM_BASE_URL and LLM_MODEL)


def summarize(records: list[dict], result: dict, max_series: int = 10) -> dict:
    df = to_frame(records)
    out = []
    for uid, g in list(df.groupby("unique_id", sort=False))[:max_series]:
        fc = [r for r in result["forecast"] if r["unique_id"] == uid]
        if not fc:
            continue
        yh = np.array([r["yhat"] for r in fc], dtype=float)
        out.append(
            {
                "serie": uid,
                "letzter_istwert": round(float(g["y"].iloc[-1]), 3),
                "mittel_letzte_perioden": round(float(g["y"].tail(min(12, len(g))).mean()), 3),
                "prognose_erste": round(float(yh[0]), 3),
                "prognose_letzte": round(float(yh[-1]), 3),
                "prognose_min": round(float(yh.min()), 3),
                "prognose_max": round(float(yh.max()), 3),
                "intervall_letzte": [fc[-1]["lo"], fc[-1]["hi"]],
            }
        )
    return {
        "gewaehltes_modell": result["selected_model"],
        "validated": result["validated"],
        "baseline": result["baseline"],
        "warnungen": result["warnings"],
        "backtest": {
            k: {"relative_mae": v["relative_mae"], "beats_baseline": v["beats_baseline"]}
            for k, v in result["backtest"].items()
        },
        "horizont": result["h"],
        "frequenz": result["freq"],
        "serien": out,
        "weitere_serien_nicht_gezeigt": max(result["n_series"] - len(out), 0),
    }


def explain(summary: dict, question: str | None) -> str:
    if not enabled():
        raise RuntimeError("LLM nicht konfiguriert (LLM_BASE_URL / LLM_MODEL)")
    user = "Ergebnis:\n" + json.dumps(summary, ensure_ascii=False)
    if question:
        user += f"\n\nFrage des Nutzers: {question}"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    r = httpx.post(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        headers=headers,
        timeout=LLM_TIMEOUT,
        json={
            "model": LLM_MODEL,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
