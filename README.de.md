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

```bash
# Timestral (Mistral AI; Code Apache-2.0, Gewichte "Modified MIT license" – Lizenztext vorher lesen, siehe THIRD_PARTY.md)
huggingface-cli download mistralai/Timestral-28M  config.json model.safetensors --local-dir /srv/models/timeseries/timestral-28m
huggingface-cli download mistralai/Timestral-209M config.json model.safetensors --local-dir /srv/models/timeseries/timestral-209m
```

Im Container gilt `HF_HUB_OFFLINE=1`: Es gibt keine Downloads zur Laufzeit. Nicht vorhandene Modelle werden in
`models: ["auto"]` übersprungen. Wird ein fehlendes Modell ausdrücklich angefragt, antwortet der Service mit 422.

**TiRex-2:** Paket `tirex-2` (Python >= 3.11, PyTorch >= 2.8; Image getestet mit Python 3.12), Eintrag `tirex_2` in `models.yaml`, ist Standardkandidat
bei `models: ["auto"]` und wird übersprungen, solange die Gewichte fehlen. Der Eintrag steht auf `device: cpu`. Für
`cuda` baut TiRex-2 beim ersten Lauf einen Kernel mit `nvcc`: Das Image braucht dann ein CUDA Toolkit passend zu PyTorch
(das mitgelieferte `python:3.11-slim` hat keins), und die GPU braucht Compute Capability 8.0 oder höher. Der Vorgänger
TiRex 1 (`tirex-ts`) steht unter der NXAI Community License und ist bewusst nicht eingebaut. Die Quantilniveaus liest der
Adapter aus dem geladenen Modell. Liegt das gewünschte Intervall außerhalb, wird begrenzt und der Service warnt.

**Timestral** (Mistral AI): Paket `timestral` (Python >= 3.10, PyTorch >= 2.4; nicht auf PyPI, `requirements-foundation.txt`
pinnt einen Commit als Tarball, der Image-Build braucht also Zugriff auf github.com), Einträge `timestral_28m` und
`timestral_209m` in `models.yaml` (`device: cpu`). Sie sind **noch keine Standardkandidaten**: Man nimmt sie mit
`models: ["auto", "timestral_28m"]` (auto plus dieses Modell) oder `models: ["timestral_28m"]` in die Anfrage auf und
entscheidet über `default_candidates` nach dem eigenen Benchmark (`live/benchmark.py --models ...,timestral_28m,timestral_209m`).
Die von Mistral veröffentlichten Benchmarkergebnisse sind hier nicht nachgemessen; eigene Messungen stehen in den Benchmark-Abschnitten unten. Das Modell liefert 21 Quantile
(0,01 bis 0,99), übliche Intervallniveaus sind damit abgedeckt. Kovariaten (nur Vergangenheit und zukünftig bekannt)
funktionieren, wenn der Checkpoint sie unterstützt; der Adapter liest das aus der `config.json` (ein Checkpoint ohne
Known-Future-Unterstützung bekommt diese Spalten als reine Vergangenheitskovariaten, mit Warnung). `symmetric_averaging: false`
im Eintrag halbiert die Rechenzeit auf Kosten etwas geringerer Genauigkeit (Standard des Pakets: an). Die Unit-Tests nutzen
Fake-Module; `live/live_covariates.py --models timestral_28m` zeigt mit echten Gewichten, ob die Kovariaten helfen.
**Lizenz:** Der Code steht unter Apache-2.0. Die Gewichte stehen unter einer „Modified MIT License“ (Feld „other“ auf
Hugging Face), deren Rechte nur gelten, solange der globale konsolidierte Monatsumsatz des nutzenden Unternehmens im Vormonat
höchstens 20 Mio. USD beträgt; darüber braucht es die kommerzielle Lizenz von Mistral AI. Die Grenze gilt auch für abgeleitete
Modelle. Vor dem Einschalten für das **eigene** Unternehmen prüfen und die LICENSE-Dateien selbst lesen (siehe
THIRD_PARTY.md). Wegen dieser Grenze ist Timestral im `models.yaml` dieses Repositories bewusst kein Standardkandidat.

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
Modelle `chronos_bolt_small` und `tirex_2`; die Spalten `TS28` und `TS209` (Timestral 28M und 209M) und `Auto + TS28`
(`auto` mit `timestral_28m` als zusätzlichem Kandidaten, was nicht der Standard dieses Repositories ist) stammen aus einem
späteren Lauf auf denselben Daten (16. Oktober 2026). Die übrigen Spalten wurden in diesem Lauf genau reproduziert.

| Datensatz | h | Auto wählt | Auto | ETS | Theta | Chronos | TiRex-2 | TS28 | TS209 | Auto + TS28 |
|---|---|---|---|---|---|---|---|---|---|---|
| airline (monatlich) | 12 | Ensemble | 0,38 | 0,74 | 0,40 | 0,39 | 0,24 | 0,19 | 0,20 | 0,28 |
| autos (monatlich) | 12 | ETS | 0,80 | 0,80 | 0,83 | 0,74 | 0,97 | 0,59 | 0,67 | 0,80 |
| sonnenfl (monatlich) | 24 | TiRex-2 | 0,34 | 1,09 | 1,21 | 0,90 | 0,34 | 0,38 | 0,35 | 0,38 |
| temp_mon (monatlich) | 12 | Ensemble | 0,77 | 0,93 | 0,95 | 0,86 | 0,73 | 0,76 | 0,76 | 0,77 |
| raub (monatlich) | 12 | Ensemble | 0,64 | 0,60 | 0,72 | 0,46 | 0,56 | 0,51 | 0,43 | 0,64 |
| shampoo (36 Monate) | 6 | Theta | 0,54 | 0,35 | 0,54 | 0,68 | 0,60 | 0,51 | 0,49 | 0,54 |
| geburten (täglich) | 14 | Ensemble | 1,06 | 1,08 | 1,07 | 1,07 | 1,06 | 1,09 | 1,07 | 1,06 |
| temp_tag (täglich) | 28 | Ensemble | 0,77 | 0,74 | 0,74 | 0,80 | 0,76 | 0,79 | 0,80 | 0,79 |
| trafo_tag (täglich, 3 Serien) | 14 | Ensemble | 0,91 | 0,82 | 0,81 | 0,92 | 0,89 | 1,18 | 0,96 | 0,91 |
| trafo_std (stündlich, 2 Serien) | 48 | Ensemble | 0,61 | 1,36 | 0,64 | 0,63 | 0,63 | 0,60 | 0,63 | 0,60 |
| bier_q (quartalsweise) | 8 | Ensemble | 0,75 | 0,80 | 0,83 | 0,85 | 0,84 | 0,97 | 1,11 | 0,75 |
| benzin_w (wöchentlich) | 26 | Ensemble | 0,98 | 2,11 | 1,23 | 0,91 | 1,15 | 1,10 | 1,23 | 0,98 |
| **Mittelwert** | | | **0,713** | 0,952 | 0,831 | 0,768 | 0,731 | 0,723 | 0,724 | 0,709 |

Lesehilfe und Grenzen:

- Die automatische Wahl schlägt die Baseline in 11 von 12 Sätzen. Bei `geburten` (reines Rauschen) gelingt es keinem Modell.
- Foundation-Modelle sind im Mittel besser als ETS und Theta, aber keines ist überall am besten. Das Ensemble der zwei besten
  Backtest-Modelle vermeidet vor allem Ausreißer (z. B. `benzin_w`: 0,98 statt 1,23 mit Theta, `raub`: 0,64 statt 0,72).
  Eine Auswahl ohne Ensemble lag im selben Lauf bei 0,757.
- Timestral (`TS28`, `TS209`): im Mittel gleichauf mit TiRex-2 (0,723 und 0,724 gegen 0,731; innerhalb des Rauschens), klar
  vorn bei `airline` und `autos`, und bei `trafo_tag` (28M), `bier_q` (209M) und `benzin_w` (beide) sowie bei `geburten`,
  wo kein Modell die Baseline schlägt, auf oder über der Baseline (> 1,0). `auto` fängt diese Fälle im Backtest ab.
  `Auto + TS28` (0,709) ist nicht messbar besser als `Auto` (0,713), und das 209M ist bei diesen 12 Datensätzen nicht besser
  als das 28M.
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
- 20 Reihen je Frequenz mit je einem Holdout sind eine kleine Stichprobe, Yearly, Monthly und Daily wurden in diesem
  ersten Lauf nicht gelaufen (siehe zweiten Lauf). Die Zahlen zeigen eine Tendenz, keine Rangfolge.

**Zweiter Lauf mit Timestral** (Oktober 2026, CPU, je 40 zufällige M4-Reihen aus allen sechs Frequenzen, Seed 1, 3
Backtest-Fenster, über LiteLLM; 240 Reihen; Einzelmodelle `theta`, `auto_ets`, `auto_arima`, `chronos_bolt_small`, `tirex_2`,
`timestral_28m`, `timestral_209m`; `auto` lief mit `timestral_28m` als zusätzlichem Kandidaten, was nicht der Standard
dieses Repositories ist). Relative MAE gegen `seasonal_naive`, geometrisches Mittel (1,00 = Baseline, kleiner ist besser):

| Modell | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | Alle 240 (rel. MAE geo) | Gewinnrate | Abdeckung 80 % |
|---|---|---|---|---|---|---|---|---|---|
| Timestral 209M | 0,780 | 0,575 | 0,550 | 0,306 | 0,756 | 0,598 | **0,570** | 75 % | 79 |
| Timestral 28M | 0,768 | 0,610 | 0,538 | 0,317 | 0,753 | 0,644 | **0,582** | 74 % | 78 |
| **Auto** (mit `timestral_28m` als Kandidat) | 0,775 | 0,657 | 0,566 | 0,318 | 0,745 | 0,633 | **0,593** | 75 % | 81 |
| TiRex-2 | 0,757 | 0,607 | 0,584 | 0,331 | 0,766 | 0,713 | 0,604 | 72 % | 78 |
| AutoARIMA | 0,849 | 0,654 | 0,524 | 0,363 | 0,762 | 0,824 | 0,636 | 70 % | 77 |
| Chronos-Bolt small | 1,059 | 0,676 | 0,641 | 0,335 | 0,735 | 0,761 | 0,664 | 70 % | 80 |
| AutoETS | 0,731 | 0,664 | 0,645 | 0,442 | 0,738 | 1,360 | 0,720 | 67 % | 76 |
| Theta | 0,668 | 0,763 | 0,578 | 0,419 | 0,748 | 1,925 | 0,750 | 68 % | 74 |

Lesehilfe und Grenzen:

- Über alle 240 Reihen ist Timestral das beste Einzelmodell (28M: 0,582, 209M: 0,570, gegen 0,604 bei TiRex-2, 0,636 bei
  AutoARIMA, 0,664 bei Chronos-Bolt, 0,720 bei AutoETS und 0,750 bei Theta). Bei 40 Reihen je Frequenz sind Unterschiede von
  0,01 bis 0,02 Rauschen, auch 28M gegen 209M.
- Die Rangfolge hängt von der Frequenz ab. Timestral führt bei Weekly und Hourly (das 209M auch bei Quarterly). Bei Monthly
  ist AutoARIMA am besten (0,524; 28M: 0,538). Bei Daily liegen alle Modelle zwischen 0,735 und 0,766, da ist nichts
  abzulesen. **Bei Yearly liegen beide Timestral-Modelle (0,768, 0,780) hinter Theta (0,668) und AutoETS (0,731)**, und nur
  etwa die Hälfte der Reihen schlägt die Baseline.
- `auto` (0,593) ist nicht besser als das beste Einzelmodell, aber besser als jedes Einzelmodell außer den beiden
  Timestral-Modellen. Es wählte für 200 von 240 Reihen ein Ensemble mit `timestral_28m`; das Ergebnis hängt also davon ab,
  dass Timestral Kandidat ist. `auto` ohne Timestral haben wir auf dieser Stichprobe nicht gemessen und machen keine Aussage
  über den Zusatznutzen.
- Bei Hourly sind AutoETS (1,36) und Theta (1,93) schlechter als die Baseline; `auto` bleibt bei 0,63.
- Die 80-%-Intervalle beider Timestral-Modelle decken insgesamt 78 bis 79 % ab (etwas schmal bei Yearly: 72 und 75 %, und bei
  Daily: 75 und 77 %). `auto` deckt 81 % ab (Hourly: 90 %, breit). TiRex-2 deckt bei Yearly nur 59 % ab.
- `validated: true` wurde für alle 240 Reihen gemeldet, aber nur 75 % davon schlugen im Holdout die Baseline (Yearly: 52 %).
  Es bedeutet „im Backtest besser als die Baseline“, keine Garantie für die Zukunft.
- M4 sind öffentliche Daten. Die Modellbeschreibung von Timestral auf Hugging Face (geprüft am 16. Oktober 2026) nennt keine
  Trainingsdaten und sagt nichts zu Überschneidungen mit Benchmarks; bei den anderen Modellen haben wir das nicht geprüft. Ein
  Vorsprung auf M4 kann deshalb teilweise Auswendiglernen sein. Der eigentliche Test sind eure eigenen Daten
  (`--source csv`).
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

**Was die Pipe kann (Version 0.8.x):**
- **Datenquellen:** CSV-Block im Chat, angehängte CSV-Datei, SQL-Block (direkt über `DB_DSN`) oder über Superset SQL Lab mit
  `projekt=1723` (der SQL-Block läuft gegen die Datenbank dieses Projekts; `{prefix}` wird zu `p1723`). Ohne Codeblock genügt
  `Prognose projekt=1723: wöchentliche Lieferungen nach Lieferdatum`: Ein Sprachmodell (`SQL_MODEL`, über LiteLLM) schreibt
  aus dem Live-Schema ein lesendes `SELECT`; das erzeugte SQL steht zur Prüfung unter dem Ergebnis.
- **Kovariaten:** weitere Zahlenspalten neben `ds`, `y`, `unique_id` gelten als Kovariaten; Zeilen am Ende ohne `y` sind die
  Zukunft (genau `h` je Serie). Spalten mit Werten dort sind zukünftig bekannt (z. B. Wartungsplan), leere nur aus der Vergangenheit.
- **Optionen in der Nachricht:** `h=12`, `freq=W-MON`, `models=a,b`, `level=80`, `windows=3`, `season=12`, `ensemble=aus`,
  `erklärung=aus` (ohne KI-Einordnung), Aktionen `Prognose` / `Backtest` / `Anomalien` / `Modelle`.
- **KI-Einordnung (optional):** Mit `EXPLAIN_MODEL` erklärt ein lokales Modell (über LiteLLM) das Ergebnis in Alltagssprache.
  Es bekommt nur die Zahlen und vorab berechnete Aussagen, rechnet nicht selbst und ist nicht Teil der Prognose.
- **Schutzmaßnahmen:** nur lesende, einzelne `SELECT`/`WITH`-Abfragen; bei erzeugtem SQL werden laufende und künftige Perioden
  entfernt, Lücken bei Zählwerten mit 0 gefüllt, und bei zu kurzer Historie fällt die Saisonlänge auf 4 zurück.
- **Wichtige Valves:** `LITELLM_URL`, `LITELLM_API_KEY`, `SUPERSET_URL/USER/PASSWORD` (Nur-Lese-Konto), `SQL_MODEL`,
  `SCHEMA_TABLES` (dem Modell gezeigte Tabellen), `SCHEMA_NOTES` (Hinweise zu den Daten), `EXPLAIN_MODEL`, `DEBUG`.

**Wichtig bei größeren Läufen:** LiteLLM bricht Aufrufe an MCP-Server standardmäßig nach 60 Sekunden ab
(`LITELLM_MCP_CLIENT_TIMEOUT`, Standard 60,0). Gemessen auf einem Server mit 80 vCPUs (Hardware-Threads, keine 80 physischen Kerne) mit `FORECAST_N_JOBS=16`, über LiteLLM, mit
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

Die Chronos-, TiRex- und Timestral-Tests nutzen Fake-Module und prüfen nur den Datenfluss der Adapter, nicht die echten Modelle.
