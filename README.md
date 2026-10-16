# Forecast Service

**English** · [Deutsch](README.de.md)

A standalone, self-hosted forecasting service for time series. It runs entirely locally, with no cloud services.

- **Stateless:** series come in with the request, nothing is stored, no database, no LLM required.
- **Model selection by backtest:** all candidates run through the same rolling-origin backtest. A model is chosen only
  if it beats a simple baseline (seasonal naive). Otherwise the service returns the baseline with `validated: false`
  and a warning. It does not claim quality it has not measured. `validated: true` means only "better than the baseline
  in the historical backtest", not a guarantee for the future.
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

```bash
# Timestral (Mistral AI; code Apache-2.0, weights "Modified MIT license" - read the license text first, see THIRD_PARTY.md)
huggingface-cli download mistralai/Timestral-28M  config.json model.safetensors --local-dir /srv/models/timeseries/timestral-28m
huggingface-cli download mistralai/Timestral-209M config.json model.safetensors --local-dir /srv/models/timeseries/timestral-209m
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

**Timestral** (Mistral AI): package `timestral` (Python >= 3.10, PyTorch >= 2.4; not on PyPI, `requirements-foundation.txt`
pins a commit as a tarball, so the image build needs access to github.com), entries `timestral_28m` and `timestral_209m`
in `models.yaml` (`device: cpu`). They are **not yet default candidates**: add them to a request with
`models: ["auto", "timestral_28m"]` (auto plus this model) or `models: ["timestral_28m"]`, and decide on
`default_candidates` after your own benchmark (`live/benchmark.py --models ...,timestral_28m,timestral_209m`). The
benchmark results published by Mistral are not re-measured here; our own measurements are in the benchmark sections below. The model gives 21 quantiles (0.01 to 0.99), so
common interval levels are covered. Covariates (past-only and known-future) work if the checkpoint supports them; the
adapter reads that from `config.json` (a checkpoint without known-future support gets those columns as past-only
covariates, with a warning). `symmetric_averaging: false` in the entry halves the compute time at some cost in accuracy
(upstream default is on). The unit tests use fake modules; `live/live_covariates.py --models timestral_28m` shows
with real weights whether the covariates help. **License:** the code is Apache-2.0. The weights are under a "Modified MIT License" (field "other" on Hugging
Face) whose rights only apply while the user's company has a global consolidated monthly revenue of at most USD 20 million
(preceding month); above that, Mistral AI's commercial license is required. The limit also covers derivatives. Check it for
YOUR company before enabling Timestral, and read the LICENSE files yourself (see THIRD_PARTY.md). Because of this limit,
Timestral is deliberately not a default candidate in this repository's `models.yaml`.

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

## Intermittent demand (spare parts, slow movers)

Series with many zero periods are detected per series (average demand interval ADI ≥ 1.32, Syntetos–Boylan). For them,
`models: ["auto"]` adds the candidates `croston_sba`, `tsb`, `imapa` and `adida` (`intermittent_candidates` in
`models.yaml`), and the backtest compares them with RMSE instead of MAE, because MAE rewards a constant zero forecast.
The result carries a warning that names the affected series. These models return point forecasts only (`lo`/`hi` stay empty); if the ensemble contains one of them, intervals are missing there too.

## Covariates (optional)

Additional numeric columns (promotion, temperature, price …) can be passed with the series:

```json
{"series": [{"unique_id": "A", "ds": "2025-01-01", "y": 10, "promo": 0, "temp": 3.1}, "..."],
 "covariates": ["promo", "temp"],
 "future_covariates": [{"unique_id": "A", "ds": "2026-01-01", "promo": 1, "temp": 2.5}, "... exactly h rows per series"],
 "h": 12, "models": ["auto"]}
```

Covariates that appear in `future_covariates` count as known in the future (e.g. planned promotions, calendar); the others
count as known only in the past. Models that can use them (`chronos_2`, `tirex_2`, `auto_arima` with future-known
covariates) run twice in the backtest, with (`<model>@cov`) and without covariates, so you see whether they help. The
service picks the covariate variant only if it wins the backtest. Missing values or a wrong number of future rows are
rejected with a clear error. `live/live_covariates.py` checks the data flow against a running service; the unit tests use
the real ARIMA path and fakes for the foundation models, so verify Chronos-2 and TiRex-2 with your own weights.

## Benchmark (reproducible)

`live/benchmark.py` downloads 12 public time series (monthly, quarterly, weekly, daily, hourly; some with several
series), holds back the end of each (h = 6 to 48), lets the service forecast with the remaining data only, and compares
the result with the true values. The metric is the relative MAE against the `seasonal_naive` baseline (1.00; lower is
better). `Auto` is the behavior with `models: ["auto"]` (backtest with 4 windows, including the ensemble). As of October
2026, CPU, models `chronos_bolt_small` and `tirex_2`; the columns `TS28` and `TS209` (Timestral 28M and 209M) and
`Auto + TS28` (`auto` with `timestral_28m` added as a candidate, which is not the default of this repository) come from a
later run on the same data (16 October 2026). The other columns were reproduced exactly in that run.

| Dataset | h | Auto picks | Auto | ETS | Theta | Chronos | TiRex-2 | TS28 | TS209 | Auto + TS28 |
|---|---|---|---|---|---|---|---|---|---|---|
| airline (monthly passengers) | 12 | ensemble | 0.38 | 0.74 | 0.40 | 0.39 | 0.24 | 0.19 | 0.20 | 0.28 |
| autos (monthly car sales) | 12 | ETS | 0.80 | 0.80 | 0.83 | 0.74 | 0.97 | 0.59 | 0.67 | 0.80 |
| sonnenfl (monthly sunspots) | 24 | TiRex-2 | 0.34 | 1.09 | 1.21 | 0.90 | 0.34 | 0.38 | 0.35 | 0.38 |
| temp_mon (monthly mean temperature) | 12 | ensemble | 0.77 | 0.93 | 0.95 | 0.86 | 0.73 | 0.76 | 0.76 | 0.77 |
| raub (monthly robberies, noisy) | 12 | ensemble | 0.64 | 0.60 | 0.72 | 0.46 | 0.56 | 0.51 | 0.43 | 0.64 |
| shampoo (36 months) | 6 | Theta | 0.54 | 0.35 | 0.54 | 0.68 | 0.60 | 0.51 | 0.49 | 0.54 |
| geburten (daily births, noise) | 14 | ensemble | 1.06 | 1.08 | 1.07 | 1.07 | 1.06 | 1.09 | 1.07 | 1.06 |
| temp_tag (daily minimum temperature) | 28 | ensemble | 0.77 | 0.74 | 0.74 | 0.80 | 0.76 | 0.79 | 0.80 | 0.79 |
| trafo_tag (daily transformer load/temp., 3 series) | 14 | ensemble | 0.91 | 0.82 | 0.81 | 0.92 | 0.89 | 1.18 | 0.96 | 0.91 |
| trafo_std (hourly transformer, 2 series) | 48 | ensemble | 0.61 | 1.36 | 0.64 | 0.63 | 0.63 | 0.60 | 0.63 | 0.60 |
| bier_q (quarterly beer production) | 8 | ensemble | 0.75 | 0.80 | 0.83 | 0.85 | 0.84 | 0.97 | 1.11 | 0.75 |
| benzin_w (weekly US gasoline) | 26 | ensemble | 0.98 | 2.11 | 1.23 | 0.91 | 1.15 | 1.10 | 1.23 | 0.98 |
| **Mean** | | | **0.713** | 0.952 | 0.831 | 0.768 | 0.731 | 0.723 | 0.724 | 0.709 |

The dataset names are the identifiers printed by `live/benchmark.py`.

How to read this, and limits:

- The automatic selection beats the baseline on 11 of 12 datasets. On `geburten` (pure noise) no model manages it.
- On average, the foundation models are better than ETS and Theta, but none is best everywhere. The ensemble of the two
  best backtest models mainly avoids outliers (e.g. `benzin_w`: 0.98 instead of 1.23 with Theta, `raub`: 0.64 instead of
  0.72). A selection without the ensemble scored 0.757 in the same run.
- Timestral (`TS28`, `TS209`): on average level with TiRex-2 (0.723 and 0.724 against 0.731; within the noise), clearly
  ahead on `airline` and `autos`, and at or above the baseline (> 1.0) on `trafo_tag` (28M), `bier_q` (209M) and `benzin_w`
  (both) as well as on `geburten`, where no model beats it. `auto` catches those cases in the backtest. `Auto + TS28`
  (0.709) is not measurably better than `Auto` (0.713), and the 209M model is no better than the 28M on these 12 datasets.
- 12 datasets with a single hold-out period each are a small sample; differences below about 0.03 are noise. The
  ensemble rule was designed after a first run on the same data (tolerance factor 1.05 fixed in advance, not tuned), so
  the numbers are slightly optimistic. An independent check with your own data is still open.

```bash
LITELLM_KEY=... python3 live/benchmark.py --url http://localhost:4000   # needs LiteLLM with the MCP server
```

### Larger benchmark (M4, own data, intermittent)

`live/benchmark_large.py` evaluates on more series than the small benchmark and reports robust metrics instead of one mean:
relative MAE and RMSE against `seasonal_naive` as geometric mean (main figure), median, win rate, MASE, sMAPE, interval
coverage, and how often `validated: true` was actually confirmed in the hold-out. Series whose baseline is error-free in
the hold-out are excluded from the relative figures (and counted).

```bash
python3 live/benchmark_large.py --source m4 --freqs Yearly,Quarterly,Monthly,Weekly,Daily,Hourly --n 100    # local, statistical models
python3 live/benchmark_large.py --source m4 --n 100 --url http://localhost:8200 --key $FORECAST_API_KEY      # against a running service (with foundation models)
python3 live/benchmark_large.py --source csv --csv my_data.csv --h 6 --freq MS --season 12                    # your own data, last h points held out
python3 live/benchmark_large.py --source intermittent --n 60                                                  # synthetic sparse demand
```

M4 series are drawn at random per frequency (official M4 horizons and seasonalities, `--seed` makes it reproducible).


**First run on the server** (October 2026, CPU, `chronos_bolt_small` and `tirex_2`, 20 random M4 series each from Quarterly,
Weekly and Hourly, seed 1, 3 backtest windows, via LiteLLM; 60 series, about 21 minutes including all single models):

| Model | Quarterly | Weekly | Hourly | All 60 (rel. MAE geo) | Win rate | Coverage 80 % |
|---|---|---|---|---|---|---|
| TiRex-2 | 0.734 | 0.227 | 0.653 | **0.477** | 77 % | 90 |
| **Auto** | 0.774 | 0.232 | 0.609 | **0.478** | 77 % | 92 |
| Chronos-Bolt small | 0.924 | 0.249 | 0.643 | 0.529 | 77 % | 90 |
| AutoARIMA | 0.740 | 0.273 | 0.802 | 0.545 | 73 % | 88 |
| AutoETS | 0.801 | 0.276 | 1.272 | 0.656 | 75 % | 78 |
| Theta | 0.860 | 0.326 | 2.122 | 0.841 | 60 % | 82 |

How to read this, and limits:

- On average over these 60 series, `auto` is as good as the best single model (TiRex-2: 0.477 vs 0.478), better on the
  hourly series and the median, but not better everywhere: on Quarterly it chose `ensemble(tirex_2+theta)` and landed
  behind TiRex-2 and AutoARIMA. The gain of the automatic selection is protection against bad single models (on Hourly
  Theta scores 2.12 and AutoETS 1.27), not a better ranking than the best model.
- `validated: true` was reported for all 60 series, but in the hold-out only 77 % of them actually beat the baseline.
  Model selection is global for the series passed together (here 20), so `validated` is a statement about the group, not
  a guarantee per series.
- The 80 % intervals of `auto` covered 92 % of the actual values, so they are conservative (too wide). AutoETS was too
  narrow on Hourly (61 %).
- Croston, TSB, IMAPA and ADIDA are meant for intermittent demand and score badly on these smooth series, as expected.
  `auto` only adds them for series that are detected as intermittent, and none of the 60 M4 series were. On series without
  zeros, IMAPA and ADIDA are identical.
- 20 series per frequency and one hold-out each is a small sample, and Yearly, Monthly and Daily were not run in this first run (see the second run). These
  numbers show a tendency, not a ranking.

**Second run with Timestral** (October 2026, CPU, 40 random M4 series from each of the six frequencies, seed 1, 3 backtest
windows, via LiteLLM; 240 series; single models `theta`, `auto_ets`, `auto_arima`, `chronos_bolt_small`, `tirex_2`,
`timestral_28m`, `timestral_209m`; `auto` ran with `timestral_28m` as an additional candidate, which is not the default of
this repository). Relative MAE against `seasonal_naive`, geometric mean (1.00 = baseline, lower is better):

| Model | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | All 240 (rel. MAE geo) | Win rate | Coverage 80 % |
|---|---|---|---|---|---|---|---|---|---|
| Timestral 209M | 0.780 | 0.575 | 0.550 | 0.306 | 0.756 | 0.598 | **0.570** | 75 % | 79 |
| Timestral 28M | 0.768 | 0.610 | 0.538 | 0.317 | 0.753 | 0.644 | **0.582** | 74 % | 78 |
| **Auto** (with `timestral_28m` as candidate) | 0.775 | 0.657 | 0.566 | 0.318 | 0.745 | 0.633 | **0.593** | 75 % | 81 |
| TiRex-2 | 0.757 | 0.607 | 0.584 | 0.331 | 0.766 | 0.713 | 0.604 | 72 % | 78 |
| AutoARIMA | 0.849 | 0.654 | 0.524 | 0.363 | 0.762 | 0.824 | 0.636 | 70 % | 77 |
| Chronos-Bolt small | 1.059 | 0.676 | 0.641 | 0.335 | 0.735 | 0.761 | 0.664 | 70 % | 80 |
| AutoETS | 0.731 | 0.664 | 0.645 | 0.442 | 0.738 | 1.360 | 0.720 | 67 % | 76 |
| Theta | 0.668 | 0.763 | 0.578 | 0.419 | 0.748 | 1.925 | 0.750 | 68 % | 74 |

How to read this, and limits:

- Over all 240 series Timestral is the best single model (28M: 0.582, 209M: 0.570, against 0.604 for TiRex-2, 0.636 for
  AutoARIMA, 0.664 for Chronos-Bolt, 0.720 for AutoETS and 0.750 for Theta). With 40 series per frequency, differences of
  0.01 to 0.02 are noise, including 28M against 209M.
- The ranking depends on the frequency. Timestral leads on Weekly and Hourly (209M also on Quarterly). On Monthly AutoARIMA
  is best (0.524; 28M: 0.538). On Daily all models are between 0.735 and 0.766, so there is nothing to read. **On Yearly
  both Timestral models (0.768, 0.780) are behind Theta (0.668) and AutoETS (0.731)**, and only about half of the series
  beat the baseline.
- `auto` (0.593) is not better than the best single model, but better than every single model except the two Timestral
  models. It chose an ensemble containing `timestral_28m` for 200 of 240 series, so this result depends on Timestral being a
  candidate. We did not run `auto` without Timestral on this sample and make no claim about its added value here.
- On Hourly AutoETS (1.36) and Theta (1.93) are worse than the baseline; `auto` stays at 0.63.
- The 80 % intervals of both Timestral models cover 78 to 79 % overall (a bit narrow on Yearly: 72 and 75 %, and on Daily:
  75 and 77 %). `auto` covers 81 % (Hourly: 90 %, wide). TiRex-2 covers only 59 % on Yearly.
- `validated: true` was reported for all 240 series, but only 75 % of them beat the baseline in the hold-out (Yearly:
  52 %). It means "better than the baseline in the backtest", not a guarantee for the future.
- M4 is public data. The Hugging Face model card of Timestral (checked 16 October 2026) names no training data and makes no
  statement about overlap with benchmarks, and we did not check this for the other models either. An advantage on M4 can
  therefore be partly memorization. The real test is your own data (`--source csv`).
Monash and GIFT-Eval loaders are not implemented; their data can be passed in via `--source csv`.

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

**What the pipe can do (version 0.8.x):**
- **Data sources:** a CSV block in the chat, an attached CSV file, an SQL block (direct, via `DB_DSN`), or via Superset SQL Lab
  with `projekt=1723` (the SQL block runs against that project's database; `{prefix}` becomes `p1723`). Without a code
  block, `Prognose projekt=1723: weekly deliveries by delivery date` is enough: a language model (`SQL_MODEL`, via LiteLLM)
  writes one read-only `SELECT` from the live schema; the generated SQL is shown below the result for review.
- **Covariates:** extra numeric columns next to `ds`, `y`, `unique_id` are covariates; trailing rows without `y` are the
  future (exactly `h` per series). Columns with values there are future-known (e.g. maintenance plan), empty ones are past-only.
- **Options in the message:** `h=12`, `freq=W-MON`, `models=a,b`, `level=80`, `windows=3`, `season=12`, `ensemble=aus`,
  `erklärung=aus` (without AI commentary), actions `Prognose` / `Backtest` / `Anomalien` / `Modelle`.
- **AI commentary (optional):** with `EXPLAIN_MODEL` set, a local model (via LiteLLM) explains the result in plain language.
  It only receives the numbers and pre-computed statements; it does not calculate and is not part of the forecast.
- **Safeguards:** only read-only single `SELECT`/`WITH` statements; for generated SQL, the current and future periods are
  dropped, gaps are filled with 0 (counts), and the season length falls back to 4 if the history is too short for the automatic one.
- **Useful Valves:** `LITELLM_URL`, `LITELLM_API_KEY`, `SUPERSET_URL/USER/PASSWORD` (read-only account), `SQL_MODEL`,
  `SCHEMA_TABLES` (tables shown to the model), `SCHEMA_NOTES` (hints about your data), `EXPLAIN_MODEL`, `DEBUG`.

**Important for larger runs:** by default, LiteLLM aborts calls to MCP servers after 60 seconds
(`LITELLM_MCP_CLIENT_TIMEOUT`, default 60.0). Measured on a server with 80 vCPUs (hardware threads, not 80 physical cores) with `FORECAST_N_JOBS=16`, via LiteLLM, with
synthetic monthly series (120 months, all models via `auto` including backtest): 50 series about 17 seconds, 200 series
about 35 seconds. Without process parallelism (`FORECAST_N_JOBS=1`) the same runs took about 30 seconds and about 2 to
2.5 minutes. Runtimes depend strongly on hardware, series length, models and windows (`live/loadtest.py` reproduces the test). For such runs, set `LITELLM_MCP_CLIENT_TIMEOUT=900` in LiteLLM's environment. Without it, the answer
never arrives and the call hangs without an error message. The direct REST route (`/v1/forecast`) is not affected.

## Limits (honestly)

- The backtest is the only quality check. Few windows on short series are noisy; increase `n_windows`.
- Model selection is global for all series passed in. Call series with very different behavior separately.
- Covariates only for the models that support them (see above); no hierarchical reconciliation, no training of your own models.
- Intermittent demand is detected and handled, but on very sparse series every model is uncertain; check `validated` and the interval.
- Every foundation model has its own license. Check it before passing weights on. The service does not ship them.
- A run occupies one concurrency slot (`FORECAST_MAX_CONCURRENCY`). Many series × models × windows take time; set
  timeouts in gateways accordingly.

## Credits

The credit for the models goes to their creators: **TiRex-2** to NXAI, **Chronos** to Amazon Science, and the statistical
models (ETS, Theta, ARIMA, Croston variants …) to the **statsforecast** project by Nixtla. This service develops none of
these models. It only makes them easy to deploy, compare and use for everyone: locally, with weights that are downloaded
once. Please check each model's license before passing weights on.

## License

Apache-2.0, see `LICENSE` and `NOTICE`. According to a review in October 2026, the libraries and models used (Chronos,
TiRex-2, statsforecast and others) are also under Apache-2.0 or MIT/BSD, details in `THIRD_PARTY.md`. Weights and
benchmark data are not shipped. This is not legal advice.

## Tests

```bash
pip install pytest && pytest -q
```

The Chronos, TiRex and Timestral tests use fake modules and only check the data flow of the adapters, not the real models.
