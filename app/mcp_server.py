"""MCP-Schicht über demselben Kern (optional, eigener Prozess).

Start:  python -m app.mcp_server        (streamable-http, Port MCP_PORT, Standard 8201)
Auth:   gleicher FORECAST_API_KEY, als `x-api-key` oder `Authorization: Bearer` Header.
Hinweis: Über MCP muss der Aufrufer die Serien als Argument mitschicken. Ein LLM, das große Serien
selbst als Tool-Argument tippen soll, ist dafür unpraktisch; Pipes/Skripte rufen die Tools direkt auf
(z. B. über LiteLLM /mcp-rest/tools/call)."""
from __future__ import annotations

import asyncio
import hmac
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import core
from .config import API_KEY
from .registry import describe_models

mcp = FastMCP(
    "forecast-service",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8201")),
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# Die Tools sind async und rechnen im Thread-Pool: Synchrone MCP-Tools laufen sonst direkt im Event-Loop und blockieren
# den ganzen Server (keine weiteren Anfragen, keine Keep-Alives) bis zum Ende der Berechnung.
@mcp.tool()
async def list_models() -> dict:
    """Verfügbare Prognosemodelle (und ob ihre Gewichte lokal vorhanden sind)."""
    return {"models": describe_models()}


@mcp.tool()
async def forecast(
    series: list[dict],
    h: int,
    freq: str | None = None,
    models: list[str] | None = None,
    n_windows: int = 2,
    level: int = 80,
    season_length: int | None = None,
    ensemble: bool = True,
) -> dict:
    """Prognose mit automatischer Modellwahl per Backtest. series = [{unique_id, ds, y}, ...].
    validated=false heißt: kein Modell schlug die Baseline (Seasonal-Naive)."""
    return await asyncio.to_thread(
        core.run_forecast, series, h, freq, season_length, models or ["auto"], n_windows, level, ensemble
    )


@mcp.tool()
async def backtest(
    series: list[dict],
    h: int,
    freq: str | None = None,
    models: list[str] | None = None,
    n_windows: int = 3,
    season_length: int | None = None,
    ensemble: bool = True,
) -> dict:
    """Modellvergleich per Rolling-Origin-Backtest (relative MAE gegen Baseline)."""
    return await asyncio.to_thread(
        core.run_backtest, series, h, freq, season_length, models or ["auto"], n_windows, 80, False, ensemble
    )


@mcp.tool()
async def detect_anomalies(
    series: list[dict],
    model: str | None = None,
    h: int = 12,
    n_windows: int = 4,
    level: int = 99,
    freq: str | None = None,
    season_length: int | None = None,
) -> dict:
    """Ausreißer: Istwerte außerhalb des Prognoseintervalls in rollierenden Out-of-sample-Fenstern."""
    return await asyncio.to_thread(core.run_anomalies, series, model, h, n_windows, freq, season_length, level)


class _KeyAuth:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and API_KEY:
            h = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            tok = h.get("x-api-key") or (h.get("authorization", "")[7:] if h.get("authorization", "").lower().startswith("bearer ") else "")
            if not hmac.compare_digest(tok, API_KEY):
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"detail":"Ungueltiger API-Key"}'})
                return
        await self.app(scope, receive, send)


def build_app():
    from . import warmup

    warmup.start_background()
    return _KeyAuth(mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host=mcp.settings.host, port=mcp.settings.port)
