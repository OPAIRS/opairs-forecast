# Forecast Service

[English](README.md) · **Deutsch**

Eigenständiger, selbst gehosteter Prognose-Service für Zeitreihen. Läuft vollständig lokal, ohne Cloud-Dienste.

- **Zustandslos:** Serien kommen im Request, nichts wird gespeichert, keine Datenbank, kein LLM nötig.
- **Modellwahl per Backtest:** Alle Kandidaten laufen im selben Rolling-Origin-Backtest. Ein Modell wird nur
  gewählt, wenn es die einfache Baseline (Seasonal-Naive) schlägt. Sonst liefert der Service die Baseline mit
  `validated: false` und einer Warnung. Er behauptet keine Qualität, die er nicht gemessen hat. `validated: true` heißt nur „im historischen Backtest besser als die Baseline“, keine Garantie für die Zukunft.
- **Austauschbare Modelle:** `models.yaml` ist die Registry. Statistikmodelle über statsforecast, Foundation-Modelle
  (Chronos-T5/-Bolt/-2 und TiRex-2) aus lokalen Verzeichnissen. Weitere Adapter sind eine kleine Klasse in `app/adapters/`.
- **Zwei Schnittstellen, ein Kern:** REST (FastAPI/OpenAPI unter `/docs`) und optional MCP (`app/mcp_server.py`).

Entwickelt von [OPAIRS Systems](https://opairs-systems.com/). Funktioniert am besten zusammen mit dem
[OPAIRS SQL Agent](https://opairs-systems.com/insights/opairs-sql-agent-datenbanken-adapter-vergleich): Der SQL Agent
holt die Daten aus der Datenbank, dieser Service liefert die Prognose.

Das Konzept (Prognose → Modellvergleich → Validierung gegen eine einfache Baseline) orientiert sich an
[TimeCopilot](https://github.com/TimeCopilot/timecopilot) (MIT). Dieser Service verwendet dessen Code nicht und hängt
nicht davon ab.

## Schnellstart

```bash
pip install -r requirements.txt            # Statistikmodelle
pip install -r requirements-foundation.txt # optional: Chronos (PyTorch)
FORECAST_API_KEY=geheim uvicorn app.api:app --port 8200
```

```bash
curl -s localhost:8200/v1/forecast -H 'X-API-Key: geheim' -H 'content-type: application/json' -d '{
  "series": [{"unique_id":"A","ds":"2024-01-01","y":10}, ...],
  "h": 8, "models": ["auto"], "n_windows": 2
}'
```

Ohne `FORECAST_API_KEY` läuft der Service ohne Authentifizierung (nur für lokale Tests).

## Endpunkte

| Endpunkt | Zweck |
|---|---|
| `GET /health` | Lebenszeichen (ohne Key) |
| `GET /v1/models` | Registry mit Verfügbarkeit (Gewichte vorhanden?) |
| `POST /v1/forecast` | Prognose mit Backtest-Auswahl: `selected_model`, `validated`, `backtest`, `warnings`, `forecast` |
| `POST /v1/backtest` | Nur Modellvergleich (optional mit Einzelzeilen) |
| `POST /v1/anomalies` | Istwerte außerhalb des Prognoseintervalls in rollierenden Out-of-sample-Fenstern |
| `POST /v1/explain` | Optional: Prognose + Erklärung durch ein LLM (nur mit `LLM_BASE_URL`/`LLM_MODEL`, sonst 501) |

Eingabe: Long-Format `unique_id, ds, y`. Frequenz wird abgeleitet oder per `freq` (pandas-Notation) gesetzt.
Lücken und doppelte Zeitstempel werden abgelehnt statt still repariert.

**Wie gelesen wird:** `relative_mae` = MAE des Modells geteilt durch MAE der Baseline (je Serie, dann gemittelt).
Unter 1 ist besser als die Baseline. `interval_coverage` zeigt, wie oft der Istwert im Intervall lag.

## Modelle

Statistikmodelle sind sofort da. Foundation-Modelle brauchen lokale Gewichte (einmalig mit Internet laden):

```bash
huggingface-cli download amazon/chronos-bolt-small --local-dir /srv/models/timeseries/chronos-bolt-small
```

```bash
# TiRex-2 (NXAI, Apache-2.0) – der Loader braucht genau diese beiden Dateien:
huggingface-cli download NX-AI/TiRex-2 model-config.yaml model.ckpt --local-dir /srv/models/timeseries/tirex-2
```

Im Container gilt `HF_HUB_OFFLINE=1`: Es gibt keine Downloads zur Laufzeit. Nicht vorhandene Modelle werden in
`models: ["auto"]` übersprungen. Wird ein fehlendes Modell ausdrücklich angefragt, antwortet der Service mit 422.

**TiRex-2:** Paket `tirex-2` (Python >= 3.11, PyTorch >= 2.8; Image getestet mit Python 3.12), Eintrag `tirex_2` in `models.yaml`, ist Standardkandidat
bei `models: ["auto"]` und wird übersprungen, solange die Gewichte fehlen. Der Eintrag steht auf `device: cpu`. Für
`cuda` baut TiRex-2 beim ersten Lauf einen Kernel mit `nvcc`: Das Image braucht dann ein CUDA Toolkit passend zu PyTorch
(das mitgelieferte `python:3.11-slim` hat keins), und die GPU braucht Compute Capability 8.0 oder höher. Der Vorgänger
TiRex 1 (`tirex-ts`) steht unter der NXAI Community License und ist bewusst nicht eingebaut. Die Quantilniveaus liest der
Adapter aus dem geladenen Modell. Liegt das gewünschte Intervall außerhalb, wird begrenzt und der Service warnt.

Intervalle: Chronos-Bolt kennt nur Quantile von 10 bis 90 %. Höhere Niveaus werden begrenzt und der Service warnt.
Für `level` über 80 daher Chronos-2 oder ein Statistikmodell verwenden.

## Kerne und Parallelität

- `FORECAST_N_JOBS` (Standard 1): Anzahl Prozesse für die Statistikmodelle (ETS, Theta, ARIMA …), die einkernig rechnen. Die
  Serien einer Anfrage werden darauf verteilt. Wirkt ab 4 Serien; `-1` nutzt alle Kerne. Fällt der Prozesspool aus, rechnet der
  Service seriell weiter. Beim ersten Einsatz eines Modells wird es einmalig vorkompiliert, damit die Arbeitsprozesse den
  kompilierten Code erben.
- `FORECAST_TORCH_THREADS` (Standard 0 = Torch-Standard, also meist alle Kerne): begrenzt die Threads von Chronos und TiRex-2 auf der
  CPU. Auf einem Server, der auch andere Dienste trägt, einen festen Wert setzen. Alternativ `OMP_NUM_THREADS`.
- `FORECAST_WARMUP` (Standard 0): mit `1` laden und kompilieren Service und MCP-Server beim Start alle Standardmodelle in einem
  Hintergrund-Thread (Logzeilen `Warm-up …`). Die erste echte Anfrage zahlt dann nicht den Kaltstart von 20 bis 30 Sekunden;
  kommt sie während des Aufwärmens, wartet sie dessen Rest ab. Die Gewichte werden nur lokal gelesen, es gibt keine Downloads.
- `FORECAST_MAX_CONCURRENCY` (Standard 1): gleichzeitige Anfragen. Mehr gleichzeitige Läufe teilen sich die Kerne.

## Modellwahl und Ensemble

Bei `models: ["auto"]` läuft jedes verfügbare Kandidatenmodell durch denselben Rolling-Origin-Backtest. Schlagen mindestens
zwei Modelle die Baseline, kommt ein weiterer Kandidat dazu: das **Ensemble der zwei besten** (Mittelwert der Punktprognosen,
Intervallgrenzen als Mittel der Grenzen, eine Näherung). Es wird wie jedes Modell im Backtest bewertet und statt des besten
Einzelmodells gewählt, wenn seine relative MAE höchstens um den Faktor `FORECAST_ENSEMBLE_TOLERANCE` (Standard 1,05)
schlechter ist. Begründung: Backtest-Ränge sind bei wenigen Fenstern verrauscht, Mittelwerte verallgemeinern meist besser.
Abschalten mit `"ensemble": false`. Der Name im Ergebnis zeigt die Mitglieder, z. B. `ensemble(tirex_2+chronos_bolt_small)`.

Schlägt kein Modell die Baseline, wird die Baseline geliefert (`validated: false`, mit Warnung).

## Intermittierende Nachfrage (Ersatzteile, Langsamdreher)

Reihen mit vielen Nullperioden werden je Reihe erkannt (mittleres Nachfrageintervall ADI ≥ 1,32, Syntetos–Boylan). Für sie
nimmt `models: ["auto"]` zusätzlich `croston_sba`, `tsb`, `imapa` und `adida` auf (`intermittent_candidates` in
`models.yaml`), und der Backtest vergleicht sie mit RMSE statt MAE, weil MAE eine konstante Null-Prognose belohnt.
Das Ergebnis enthält eine Warnung mit den betroffenen Reihen. Diese Modelle liefern nur Punktprognosen (`lo`/`hi` bleiben leer); enthält das Ensemble eines davon, fehlen die Intervalle auch dort.

## Kovariaten (optional)

Zusätzliche Zahlenspalten (Aktion, Temperatur, Preis …) können mit den Reihen übergeben werden:

```json
{"series": [{"unique_id": "A", "ds": "2025-01-01", "y": 10, "promo": 0, "temp": 3.1}, "..."],
 "covariates": ["promo", "temp"],
 "future_covariates": [{"unique_id": "A", "ds": "2026-01-01", "promo": 1, "temp": 2.5}, "... genau h Zeilen je Reihe"],
 "h": 12, "models": ["auto"]}
```

Kovariaten in `future_covariates` gelten als zukünftig bekannt (z. B. geplante Aktionen, Kalender), die übrigen nur als
Vergangenheitswerte. Modelle, die sie nutzen können (`chronos_2`, `tirex_2`, `auto_arima` mit zukünftig bekannten
Kovariaten), laufen im Backtest zweimal, mit (`<modell>@cov`) und ohne Kovariaten, damit man sieht, ob sie helfen. Die
Variante mit Kovariaten wird nur gewählt, wenn sie den Backtest gewinnt. Fehlende Werte oder eine falsche Zahl Zukunftszeilen
werden mit klarer Fehlermeldung abgelehnt. `live/live_covariates.py` prüft den Datenfluss gegen einen laufenden Dienst; die
Unit-Tests nutzen für ARIMA den echten Pfad und für die Foundation-Modelle Attrappen, Chronos-2 und TiRex-2 deshalb mit den
eigenen Gewichten prüfen.

## Benchmark (reproduzierbar)

`live/benchmark.py` lädt 12 öffentliche Zeitreihen (monatlich, quartalsweise, wöchentlich, täglich, stündlich; teils mehrere
Serien), hält jeweils das Ende zurück (h = 6 bis 48), lässt den Service nur mit den übrigen Daten prognostizieren und
vergleicht mit den echten Werten. Kennzahl ist die relative MAE gegen die Baseline `seasonal_naive` (1,00; kleiner ist besser).
`Auto` ist das Verhalten bei `models: ["auto"]` (Backtest mit 4 Fenstern, inklusive Ensemble). Stand: Oktober 2026, CPU,
Modelle `chronos_bolt_small` und `tirex_2`.

| Datensatz | h | Auto wählt | Auto | ETS | Theta | Chronos | TiRex-2 |
|---|---|---|---|---|---|---|---|
| airline (monatlich) | 12 | Ensemble | 0,38 | 0,74 | 0,40 | 0,39 | 0,24 |
| autos (monatlich) | 12 | ETS | 0,80 | 0,80 | 0,83 | 0,74 | 0,97 |
| sonnenfl (monatlich) | 24 | TiRex-2 | 0,34 | 1,09 | 1,21 | 0,90 | 0,34 |
| temp_mon (monatlich) | 12 | Ensemble | 0,77 | 0,93 | 0,95 | 0,86 | 0,73 |
| raub (monatlich) | 12 | Ensemble | 0,64 | 0,60 | 0,72 | 0,46 | 0,56 |
| shampoo (36 Monate) | 6 | Theta | 0,54 | 0,35 | 0,54 | 0,68 | 0,60 |
| geburten (täglich) | 14 | Ensemble | 1,06 | 1,08 | 1,07 | 1,07 | 1,06 |
| temp_tag (täglich) | 28 | Ensemble | 0,77 | 0,74 | 0,74 | 0,80 | 0,76 |
| trafo_tag (täglich, 3 Serien) | 14 | Ensemble | 0,91 | 0,82 | 0,81 | 0,92 | 0,89 |
| trafo_std (stündlich, 2 Serien) | 48 | Ensemble | 0,61 | 1,36 | 0,64 | 0,63 | 0,63 |
| bier_q (quartalsweise) | 8 | Ensemble | 0,75 | 0,80 | 0,83 | 0,85 | 0,84 |
| benzin_w (wöchentlich) | 26 | Ensemble | 0,98 | 2,11 | 1,23 | 0,91 | 1,15 |
| **Mittelwert** | | | **0,713** | 0,952 | 0,831 | 0,768 | 0,731 |

Lesehilfe und Grenzen:

- Die automatische Wahl schlägt die Baseline in 11 von 12 Sätzen. Bei `geburten` (reines Rauschen) gelingt es keinem Modell.
- Foundation-Modelle sind im Mittel besser als ETS und Theta, aber keines ist überall am besten. Das Ensemble der zwei besten
  Backtest-Modelle vermeidet vor allem Ausreißer (z. B. `benzin_w`: 0,98 statt 1,23 mit Theta, `raub`: 0,64 statt 0,72).
  Eine Auswahl ohne Ensemble lag im selben Lauf bei 0,757.
- 12 Datensätze mit je einem Rückblick-Zeitraum sind eine kleine Stichprobe; Unterschiede unter etwa 0,03 sind Rauschen.
  Die Ensemble-Regel wurde nach einem ersten Lauf auf denselben Daten entworfen (Toleranzfaktor 1,05 vorab festgelegt, nicht
  angepasst), die Zahlen sind daher leicht optimistisch. Eine unabhängige Prüfung mit eigenen Daten steht aus.

```bash
LITELLM_KEY=... python3 live/benchmark.py --url http://localhost:4000   # braucht LiteLLM mit dem MCP-Server
```

### Größerer Benchmark (M4, eigene Daten, intermittierend)

`live/benchmark_large.py` wertet mehr Reihen aus als der kleine Benchmark und berichtet robuste Kennzahlen statt eines
Mittelwerts: relative MAE und RMSE gegen `seasonal_naive` als geometrisches Mittel (Hauptkennzahl), Median, Gewinnrate, MASE,
sMAPE, Intervallabdeckung und wie oft `validated: true` im Holdout tatsächlich bestätigt wurde. Reihen, deren Baseline im
Holdout fehlerfrei ist, zählen nicht in die relativen Kennzahlen (werden aber ausgewiesen).

```bash
python3 live/benchmark_large.py --source m4 --freqs Yearly,Quarterly,Monthly,Weekly,Daily,Hourly --n 100    # lokal, Statistikmodelle
python3 live/benchmark_large.py --source m4 --n 100 --url http://localhost:8200 --key $FORECAST_API_KEY      # gegen laufenden Dienst (mit Foundation-Modellen)
python3 live/benchmark_large.py --source csv --csv meine_daten.csv --h 6 --freq MS --season 12                # eigene Daten, letzte h Punkte zurückgehalten
python3 live/benchmark_large.py --source intermittent --n 60                                                  # synthetischer Sporadikbedarf
```

Die M4-Reihen werden je Frequenz zufällig gezogen (offizielle M4-Horizonte und Saisonlängen, `--seed` macht es
reproduzierbar). 

**Erster Lauf auf dem Server** (Oktober 2026, CPU, `chronos_bolt_small` und `tirex_2`, je 20 zufällige M4-Reihen aus
Quarterly, Weekly und Hourly, Seed 1, 3 Backtest-Fenster, über LiteLLM; 60 Reihen, rund 21 Minuten inklusive aller Einzelmodelle):

| Modell | Quarterly | Weekly | Hourly | Alle 60 (rel. MAE geo) | Gewinnrate | Abdeckung 80 % |
|---|---|---|---|---|---|---|
| TiRex-2 | 0,734 | 0,227 | 0,653 | **0,477** | 77 % | 90 |
| **Auto** | 0,774 | 0,232 | 0,609 | **0,478** | 77 % | 92 |
| Chronos-Bolt small | 0,924 | 0,249 | 0,643 | 0,529 | 77 % | 90 |
| AutoARIMA | 0,740 | 0,273 | 0,802 | 0,545 | 73 % | 88 |
| AutoETS | 0,801 | 0,276 | 1,272 | 0,656 | 75 % | 78 |
| Theta | 0,860 | 0,326 | 2,122 | 0,841 | 60 % | 82 |

Lesehilfe und Grenzen:

- Über diese 60 Reihen ist `auto` so gut wie das beste Einzelmodell (TiRex-2: 0,477 gegen 0,478), besser bei den
  Stundenreihen und beim Median, aber nicht überall: Bei Quarterly wählte es `ensemble(tirex_2+theta)` und lag hinter
  TiRex-2 und AutoARIMA. Der Nutzen der automatischen Wahl ist der Schutz vor schlechten Einzelmodellen (bei Hourly
  kommt Theta auf 2,12 und AutoETS auf 1,27), nicht ein besserer Rang als das beste Modell.
- `validated: true` wurde für alle 60 Reihen gemeldet, im Holdout schlugen aber nur 77 % davon die Baseline. Die
  Modellwahl gilt global für die gemeinsam übergebenen Reihen (hier 20), `validated` ist also eine Aussage über die
  Gruppe, keine Garantie je Reihe.
- Die 80-%-Intervalle von `auto` deckten 92 % der Istwerte ab, sie sind also konservativ (zu breit). AutoETS war bei
  Hourly zu schmal (61 %).
- Croston, TSB, IMAPA und ADIDA sind für intermittierende Nachfrage gedacht und schneiden auf diesen glatten Reihen
  erwartungsgemäß schlecht ab. `auto` nimmt sie nur für als intermittierend erkannte Reihen auf, und keine der 60
  M4-Reihen war es. Bei Reihen ohne Nullen sind IMAPA und ADIDA identisch.
- 20 Reihen je Frequenz mit je einem Holdout sind eine kleine Stichprobe, Yearly, Monthly und Daily wurden nicht
  gelaufen. Die Zahlen zeigen eine Tendenz, keine Rangfolge.
 Loader für Monash und GIFT-Eval gibt es nicht; deren Daten lassen sich über `--source csv` einspeisen.

## Docker und LiteLLM

`docker-compose.example.yaml` startet `forecast-api` (8200) und `forecast-mcp` (8201) aus demselben Image.
`litellm-mcp.example.yaml` registriert den MCP-Server in LiteLLM. Pipes können die Tools dann über LiteLLM
aufrufen, ohne dass ein LLM per Tool-Calling entscheiden muss.

**OpenWebUI-Pipe (Test-/Beispielintegration):** `pipes/opairs_forecast_pipe.py` ist eine Test- und Beispiel-Pipe für OpenWebUI,
kein Produktivbestandteil des Dienstes. Sie enthält nur Logik (Daten aus SQL-Block oder CSV lesen, Service über LiteLLM
aufrufen, Ergebnis darstellen); die Modelle laufen im Service. Es stehen keine Zugangsdaten im Code. Den LiteLLM-Key (und
optional `DB_DSN` für SQL-Blöcke) trägst du nach dem Import in den Valves der Pipe ein. Für mehr als einen Test: eigener
LiteLLM-Virtual-Key statt Master-Key und ein nur lesender Datenbankbenutzer.

**Wichtig bei größeren Läufen:** LiteLLM bricht Aufrufe an MCP-Server standardmäßig nach 60 Sekunden ab
(`LITELLM_MCP_CLIENT_TIMEOUT`, Standard 60,0). Gemessen auf einer 80-Kern-CPU mit `FORECAST_N_JOBS=16`, über LiteLLM, mit
synthetischen Monatsreihen (120 Monate, alle Modelle per `auto` inklusive Backtest): 50 Serien rund 17 Sekunden, 200 Serien
rund 35 Sekunden. Ohne Prozess-Parallelität (`FORECAST_N_JOBS=1`) dauerten dieselben Läufe rund 30 Sekunden bzw. 2 bis
2,5 Minuten. Die Laufzeit hängt stark von Hardware, Serienlänge, Modellen und Fenstern ab (`live/loadtest.py` wiederholt den Test). Für solche Läufe in der Umgebung von LiteLLM `LITELLM_MCP_CLIENT_TIMEOUT=900` setzen. Ohne den Wert
kommt die Antwort nie an und der Aufruf hängt ohne Fehlermeldung. Der direkte REST-Weg (`/v1/forecast`) ist davon nicht betroffen.

## Grenzen (ehrlich)

- Der Backtest ist die einzige Qualitätsprüfung. Wenige Fenster auf kurzen Serien sind verrauscht; `n_windows` erhöhen.
- Die Modellwahl ist global für alle übergebenen Serien. Serien mit sehr unterschiedlichem Verhalten getrennt aufrufen.
- Kovariaten nur für Modelle, die sie unterstützen (siehe oben); keine hierarchische Abstimmung, kein Training eigener Modelle.
- Intermittierende Nachfrage wird erkannt und behandelt, bei sehr dünnen Reihen bleibt aber jede Prognose unsicher; `validated` und das Intervall beachten.
- Jedes Foundation-Modell hat eine eigene Lizenz. Prüfen, bevor Gewichte weitergegeben werden. Der Service liefert sie nicht mit.
- Ein Lauf belegt einen Parallelitäts-Slot (`FORECAST_MAX_CONCURRENCY`). Viele Serien × Modelle × Fenster brauchen Zeit;
  Timeouts in Gateways entsprechend setzen.

## Danksagung

Der Dank für die Modelle gebührt ihren Urhebern: **TiRex-2** NXAI, **Chronos** Amazon Science und die Statistikmodelle
(ETS, Theta, ARIMA, Croston-Varianten …) dem Projekt **statsforecast** von Nixtla. Dieser Dienst entwickelt keines dieser
Modelle. Er macht sie nur für alle einfach einsetz-, vergleich- und nutzbar: lokal, mit Gewichten, die einmal geladen werden.
Bitte vor einer Weitergabe von Gewichten die Lizenz des jeweiligen Modells prüfen.

## Lizenz

Apache-2.0, siehe `LICENSE` und `NOTICE`. Die genutzten Bibliotheken und Modelle (Chronos, TiRex-2, statsforecast u. a.)
stehen laut Prüfung vom Oktober 2026 ebenfalls unter Apache-2.0 bzw. MIT/BSD, Details in `THIRD_PARTY.md`. Gewichte und
Benchmark-Daten werden nicht mitgeliefert.

## Tests

```bash
pip install pytest && pytest -q
```

Die Chronos- und TiRex-Tests nutzen Fake-Module und prüfen nur den Datenfluss der Adapter, nicht die echten Modelle.
