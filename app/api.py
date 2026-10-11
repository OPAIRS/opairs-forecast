"""REST-API. Zustandslos: Serien kommen im Request, nichts wird gespeichert."""
from __future__ import annotations

import hmac
import logging
from typing import Union

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__, core, explain as explain_mod
from .config import API_KEY, MAX_HORIZON
from .data import DataError
from .registry import UnknownModel, describe_models

log = logging.getLogger("forecast")
app = FastAPI(title="Forecast Service", version=__version__)


@app.on_event("startup")
def _warmup() -> None:
    from . import warmup

    warmup.start_background()


def auth(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
    if not API_KEY:
        return
    token = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else "")
    if not hmac.compare_digest(token or "", API_KEY):
        raise HTTPException(401, "Ungültiger oder fehlender API-Key")


class Point(BaseModel):
    unique_id: Union[str, int]
    ds: str
    y: float | None


class Common(BaseModel):
    series: list[Point] = Field(..., description="Long-Format: unique_id, ds, y")
    freq: str | None = Field(None, description="pandas-Frequenz, z. B. D, W-MON, MS. Sonst automatisch.")
    season_length: int | None = Field(None, ge=1)
    level: int = Field(80, ge=50, le=99, description="Intervallniveau in %")


class ForecastRequest(Common):
    h: int = Field(..., ge=1, le=MAX_HORIZON)
    models: list[str] = Field(["auto"], description="'auto' oder Modellnamen aus /v1/models")
    n_windows: int = Field(2, ge=0, le=20, description="Backtest-Fenster; 0 = kein Backtest (genau ein Modell)")
    ensemble: bool = Field(True, description="Ensemble der zwei besten Modelle als Kandidat zulassen")


class BacktestRequest(Common):
    h: int = Field(..., ge=1, le=MAX_HORIZON)
    models: list[str] = ["auto"]
    n_windows: int = Field(3, ge=1, le=20)
    include_rows: bool = False
    ensemble: bool = True


class AnomalyRequest(Common):
    model: str | None = None
    h: int = Field(12, ge=1, le=MAX_HORIZON, description="Fensterlänge")
    n_windows: int = Field(4, ge=1, le=50)
    level: int = Field(99, ge=50, le=99)


class ExplainRequest(ForecastRequest):
    question: str | None = None


def _recs(req: Common) -> list[dict]:
    return [p.model_dump() for p in req.series]


def _guard(fn):
    try:
        return fn()
    except (DataError, UnknownModel) as e:
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Fehler")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/v1/models", dependencies=[Depends(auth)])
def models():
    return {"models": describe_models()}


@app.post("/v1/forecast", dependencies=[Depends(auth)])
def forecast(req: ForecastRequest):
    return _guard(
        lambda: core.run_forecast(
            _recs(req), req.h, req.freq, req.season_length, req.models, req.n_windows, req.level,
            ensemble=req.ensemble,
        )
    )


@app.post("/v1/backtest", dependencies=[Depends(auth)])
def backtest(req: BacktestRequest):
    return _guard(
        lambda: core.run_backtest(
            _recs(req), req.h, req.freq, req.season_length, req.models, req.n_windows, req.level, req.include_rows,
            ensemble=req.ensemble,
        )
    )


@app.post("/v1/anomalies", dependencies=[Depends(auth)])
def anomalies(req: AnomalyRequest):
    return _guard(
        lambda: core.run_anomalies(
            _recs(req), req.model, req.h, req.n_windows, req.freq, req.season_length, req.level
        )
    )


@app.post("/v1/explain", dependencies=[Depends(auth)])
def explain(req: ExplainRequest):
    if not explain_mod.enabled():
        raise HTTPException(501, "Erklärfunktion nicht konfiguriert (LLM_BASE_URL und LLM_MODEL setzen)")

    def run():
        recs = _recs(req)
        res = core.run_forecast(
            recs, req.h, req.freq, req.season_length, req.models, req.n_windows, req.level, ensemble=req.ensemble
        )
        res["explanation"] = explain_mod.explain(explain_mod.summarize(recs, res), req.question)
        return res

    return _guard(run)
