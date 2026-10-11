# Forecast Service

**English** · [Deutsch](README.de.md)

A standalone, self-hosted forecasting service for time series. It runs entirely locally, with no cloud services.

- **Stateless:** series come in with the request, nothing is stored, no database, no LLM required.
- **Model selection by backtest:** all candidates run through the same rolling-origin backtest. A model is chosen only
  if it beats a simple baseline (seasonal naive). Otherwise the service returns the baseline with `validated: false`
  and a warning. It does not claim quality it has not measured.
- **Swappable models:** `models.yaml` is the registry. Statistical models via statsforecast, foundation models
  (Chronos-T5/-Bolt/-2 and TiRex-2) from local directories. Another adapter is one small class in `app/adapters/`.
- **Two interfaces, one core:** REST (FastAPI/OpenAPI at `/docs`) and optional MCP (`app/mcp_server.py`).

Developed by [OPAIRS Systems](https://opairs-systems.com/). Works best together with the
[OPAIRS SQL Agent](https://opairs-systems.com/insights/opairs-sql-agent-datenbanken-adapter-vergleich): the SQL Agent
fetches the data from the database, this service produces the forecast.

The concept (forecast → model comparison → validation against a simple baseline) is inspired by
[TimeCopilot](https://github.com/TimeCopilot/timecopilot) (MIT). This service does not use its code and does not
depend on it.

## Quick start

```bash
pip install -r requirements.txt            # statistical models
pip install -r requirements-foundation.txt # optional: Chronos (PyTorch)
FORECAST_API_KEY=secret uvicorn app.api:app --port 8200
```

```bash
curl -s localhost:8200/v1/forecast -H 'X-API-Key: secret' -H 'content-type: application/json' -d '{
  "series": [{"unique_id":"A","ds":"2024-01-01","y":10}, ...],
  "h": 8, "models": ["auto"], "n_windows": 2
}'
```

Without `FORECAST_API_KEY` the service runs without authentication (for local tests only).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check (no key needed) |
| `GET /v1/models` | Registry with availability (are the weights present?) |
| `POST /v1/forecast` | Forecast with backtest selection: `selected_model`, `validated`, `backtest`, `warnings`, `forecast` |
| `POST /v1/backtest` | Model comparison only (optionally with per-row results) |
| `POST /v1/anomalies` | Actual values outside the forecast interval in rolling out-of-sample windows |
| `POST /v1/explain` | Optional: forecast plus an explanation by an LLM (only with `LLM_BASE_URL`/`LLM_MODEL`, otherwise 501) |

Input: long format `unique_id, ds, y`. The frequency is inferred or set via `freq` (pandas notation).
Gaps and duplicate timestamps are rejected instead of being silently repaired.

**How to read the results:** `relative_mae` = the model's MAE divided by the baseline's MAE (per series, then averaged).
Below 1 is better than the baseline. `interval_coverage` shows how often the actual value fell inside the interval.

## Models

Statistical models work out of the box. Foundation models need local weights (download once, with internet access):

```bash
huggingface-cli download amazon/chronos-bolt-small --local-dir /srv/models/timeseries/chronos-bolt-small
```

```bash
# TiRex-2 (NXAI, Apache-2.0) – the loader needs exactly these two files:
huggingface-cli download NX-AI/TiRex-2 model-config.yaml model.ckpt --local-dir /srv/models/timeseries/tirex-2
```

Inside the container `HF_HUB_OFFLINE=1` applies: there are no downloads at runtime. Models that are not present are
skipped with `models: ["auto"]`. If a missing model is requested explicitly, the service answers with 422.

**TiRex-2:** package `tirex-2` (Python >= 3.11, PyTorch >= 2.8; image tested with Python 3.12), entry `tirex_2` in
`models.yaml`. It is a default candidate with `models: ["auto"]` and is skipped as long as the weights are missing. The
entry is set to `device: cpu`. For `cuda`, TiRex-2 builds a kernel with `nvcc` on the first run: the image then needs a
CUDA toolkit matching PyTorch (the supplied `python:3.11-slim` has none), and the GPU needs compute capability 8.0 or
higher. Its predecessor TiRex 1 (`tirex-ts`) is under the NXAI Community License and is deliberately not integrated.
The adapter reads the quantile levels from the loaded model. If the requested interval lies outside them, it is clamped
and the service warns.

Intervals: Chronos-Bolt only provides quantiles from 10 to 90 %. Higher levels are clamped and the service warns.
For `level` above 80, use Chronos-2 or a statistical model.

## Cores and parallelism

- `FORECAST_N_JOBS` (default 1): number of processes for the statistical models (ETS, Theta, ARIMA …), which run on a
  single core. The series of one request are distributed across them. Takes effect from 4 series; `-1` uses all cores. If
  the process pool fails, the service continues serially. A model is pre-compiled once on first use so the worker
  processes inherit the compiled code.
- `FORECAST_TORCH_THREADS` (default 0 = Torch default, usually all cores): limits the threads of Chronos and TiRex-2 on
  the CPU. On a server that also hosts other services, set a fixed value. `OMP_NUM_THREADS` works as an alternative.
- `FORECAST_WARMUP` (default 0): with `1`, the service and the MCP server load and compile all default models at startup
  in a background thread (log lines `Warm-up …`). The first real request then does not pay the cold start of 20 to
  30 seconds; if it arrives during the warm-up, it waits for the rest of it. Weights are read locally only, there are no
  downloads.
- `FORECAST_MAX_CONCURRENCY` (default 1): simultaneous requests. More simultaneous runs share the cores.

## Model selection and ensemble

With `models: ["auto"]`, every available candidate model runs through the same rolling-origin backtest. If at least two
models beat the baseline, one more candidate is added: the **ensemble of the two best** (mean of the point forecasts,
interval bounds as the mean of the bounds, an approximation). It is scored in the backtest like any other model and is
chosen instead of the best single model if its relative MAE is worse by at most the factor
`FORECAST_ENSEMBLE_TOLERANCE` (default 1.05). Reasoning: with few windows, backtest rankings are noisy, and averages
usually generalize better. Disable it with `"ensemble": false`. The name in the result shows the members, e.g.
`ensemble(tirex_2+chronos_bolt_small)`.

If no model beats the baseline, the baseline is returned (`validated: false`, with a warning).

## Benchmark (reproducible)

`live/benchmark.py` downloads 12 public time series (monthly, quarterly, weekly, daily, hourly; some with several
series), holds back the end of each (h = 6 to 48), lets the service forecast with the remaining data only, and compares
the result with the true values. The metric is the relative MAE against the `seasonal_naive` baseline (1.00; lower is
better). `Auto` is the behavior with `models: ["auto"]` (backtest with 4 windows, including the ensemble). As of October
2026, CPU, models `chronos_bolt_small` and `tirex_2`.

| Dataset | h | Auto picks | Auto | ETS | Theta | Chronos | TiRex-2 |
|---|---|---|---|---|---|---|---|
| airline (monthly passengers) | 12 | ensemble | 0.38 | 0.74 | 0.40 | 0.39 | 0.24 |
| autos (monthly car sales) | 12 | ETS | 0.80 | 0.80 | 0.83 | 0.74 | 0.97 |
| sonnenfl (monthly sunspots) | 24 | TiRex-2 | 0.34 | 1.09 | 1.21 | 0.90 | 0.34 |
| temp_mon (monthly mean temperature) | 12 | ensemble | 0.77 | 0.93 | 0.95 | 0.86 | 0.73 |
| raub (monthly robberies, noisy) | 12 | ensemble | 0.64 | 0.60 | 0.72 | 0.46 | 0.56 |
| shampoo (36 months) | 6 | Theta | 0.54 | 0.35 | 0.54 | 0.68 | 0.60 |
| geburten (daily births, noise) | 14 | ensemble | 1.06 | 1.08 | 1.07 | 1.07 | 1.06 |
| temp_tag (daily minimum temperature) | 28 | ensemble | 0.77 | 0.74 | 0.74 | 0.80 | 0.76 |
| trafo_tag (daily transformer load/temp., 3 series) | 14 | ensemble | 0.91 | 0.82 | 0.81 | 0.92 | 0.89 |
| trafo_std (hourly transformer, 2 series) | 48 | ensemble | 0.61 | 1.36 | 0.64 | 0.63 | 0.63 |
| bier_q (quarterly beer production) | 8 | ensemble | 0.75 | 0.80 | 0.83 | 0.85 | 0.84 |
| benzin_w (weekly US gasoline) | 26 | ensemble | 0.98 | 2.11 | 1.23 | 0.91 | 1.15 |
| **Mean** | | | **0.713** | 0.952 | 0.831 | 0.768 | 0.731 |

The dataset names are the identifiers printed by `live/benchmark.py`.

How to read this, and limits:

- The automatic selection beats the baseline on 11 of 12 datasets. On `geburten` (pure noise) no model manages it.
- On average, the foundation models are better than ETS and Theta, but none is best everywhere. The ensemble of the two
  best backtest models mainly avoids outliers (e.g. `benzin_w`: 0.98 instead of 1.23 with Theta, `raub`: 0.64 instead of
  0.72). A selection without the ensemble scored 0.757 in the same run.
- 12 datasets with a single hold-out period each are a small sample; differences below about 0.03 are noise. The
  ensemble rule was designed after a first run on the same data (tolerance factor 1.05 fixed in advance, not tuned), so
  the numbers are slightly optimistic. An independent check with your own data is still open.

```bash
LITELLM_KEY=... python3 live/benchmark.py --url http://localhost:4000   # needs LiteLLM with the MCP server
```

## Docker and LiteLLM

`docker-compose.example.yaml` starts `forecast-api` (8200) and `forecast-mcp` (8201) from the same image.
`litellm-mcp.example.yaml` registers the MCP server in LiteLLM. Pipes can then call the tools through LiteLLM, without
an LLM having to decide via tool calling.

**OpenWebUI pipe (test/example integration):** `pipes/opairs_forecast_pipe.py` is a test and example pipe for OpenWebUI,
not a production component of the service. It contains only logic (read data from an SQL block or CSV, call the
service through LiteLLM, render the result); the models run in the service. There are no credentials in the code. After
importing the pipe, enter the LiteLLM key (and optionally `DB_DSN` for SQL blocks) in the pipe's Valves. For more than a
test: use a dedicated LiteLLM virtual key instead of the master key, and a read-only database user. The pipe's user
interface texts are in German.

**Important for larger runs:** by default, LiteLLM aborts calls to MCP servers after 60 seconds
(`LITELLM_MCP_CLIENT_TIMEOUT`, default 60.0). A run with 50 series takes about 30 seconds on an 80-core CPU, 200 series
about 2 minutes. For such runs, set `LITELLM_MCP_CLIENT_TIMEOUT=900` in LiteLLM's environment. Without it, the answer
never arrives and the call hangs without an error message. The direct REST route (`/v1/forecast`) is not affected.

## Limits (honestly)

- The backtest is the only quality check. Few windows on short series are noisy; increase `n_windows`.
- Model selection is global for all series passed in. Call series with very different behavior separately.
- No covariates, no hierarchical reconciliation, no training of your own models.
- Every foundation model has its own license. Check it before passing weights on. The service does not ship them.
- A run occupies one concurrency slot (`FORECAST_MAX_CONCURRENCY`). Many series × models × windows take time; set
  timeouts in gateways accordingly.

## License

Apache-2.0, see `LICENSE` and `NOTICE`. According to a review in October 2026, the libraries and models used (Chronos,
TiRex-2, statsforecast and others) are also under Apache-2.0 or MIT/BSD, details in `THIRD_PARTY.md`. Weights and
benchmark data are not shipped. This is not legal advice.

## Tests

```bash
pip install pytest && pytest -q
```

The Chronos and TiRex tests use fake modules and only check the data flow of the adapters, not the real models.
