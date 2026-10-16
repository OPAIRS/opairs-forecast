# Drittkomponenten und Lizenzen

Stand der Prüfung: Oktober 2026. Das ist eine technische Bestandsaufnahme, keine Rechtsberatung. Lizenzen können sich bei
neuen Versionen ändern: Vor einer Weitergabe oder Aktualisierung die jeweilige Quelle erneut ansehen.

Der Service selbst steht unter Apache-2.0 (siehe `LICENSE`). Es werden **weder Bibliotheken noch Modellgewichte mitgeliefert**;
beides wird bei der Installation bzw. beim Einrichten vom Betreiber geladen.

## Python-Bibliotheken

| Paket | Lizenz | Quelle der Angabe |
|---|---|---|
| statsforecast | Apache-2.0 | Paket-Metadaten |
| pandas | BSD-3-Clause | Paket-Metadaten |
| numpy | BSD-3-Clause (u. a.) | Paket-Metadaten |
| fastapi | MIT | Paket-Metadaten |
| uvicorn | BSD-3-Clause | Paket-Metadaten |
| pydantic | MIT | Paket-Metadaten |
| httpx | BSD-3-Clause | Paket-Metadaten |
| pyyaml | MIT | Paket-Metadaten |
| mcp (Python SDK) | MIT | Paket-Metadaten |
| chronos-forecasting | Apache-2.0 | README und LICENSE des Repositories |
| tirex-2 | Apache-2.0 | README und LICENSE des Repositories |
| timestral (Mistral AI) | Apache-2.0 | LICENSE und pyproject.toml des Repositories (Commit a534bdb) |
| torch (Abhängigkeit der Foundation-Modelle) | BSD-3-Clause | nicht in dieser Umgebung geprüft |

Nur für die Pipe in OpenWebUI: `requests` (Apache-2.0), `psycopg2-binary` (LGPL mit Ausnahme, nicht in dieser Umgebung geprüft).

## Modellgewichte

Diese Angaben stammen aus den Lizenzfeldern der Hugging-Face-Modellkarten. Die Gewichte liegen beim Betreiber, nicht in diesem Repository.

| Modell | Lizenz laut Modellkarte |
|---|---|
| amazon/chronos-bolt-small | Apache-2.0 |
| amazon/chronos-bolt-base | Apache-2.0 |
| amazon/chronos-2 | Apache-2.0 |
| amazon/chronos-t5-small (Stichprobe für die T5-Familie) | Apache-2.0 |
| NX-AI/TiRex-2 | Apache-2.0 |
| mistralai/Timestral-28M, -209M (auch -5M, -123M) | „Modified MIT License“ (Lizenzfeld auf Hugging Face: „other“), Namensnennung „2026 - Mistral AI“. **Umsatzgrenze:** Die Rechte gelten nur, solange der globale konsolidierte Monatsumsatz des nutzenden Unternehmens im Vormonat 20 Mio. USD nicht übersteigt; darüber ist eine kommerzielle Lizenz von Mistral nötig (sales@mistral.ai). Die Grenze gilt auch für abgeleitete Modelle und kombinierte Werke. Außerdem darf das Modell keine Rechte Dritter verletzen. Gelesen wurde die LICENSE von Timestral-209M über ein Zusammenfassungswerkzeug, nicht der Rohtext; die LICENSE von -28M war nicht abrufbar und wird als gleichlautend angenommen. **Vor Nutzung und Weitergabe die LICENSE-Dateien selbst lesen** (Links auf den Hugging-Face-Modellseiten). Wer das Modell betreibt, muss die Grenze für **sein** Unternehmen prüfen |

Nicht einzeln geprüft: `chronos-bolt-tiny` und die übrigen T5-Größen (voraussichtlich gleich, vor Nutzung nachsehen).
Die Lizenz der Gewichte sagt nichts über die Herkunft der Trainingsdaten aus; dazu geben die Modellkarten Auskunft.
Bewusst **nicht** eingebaut: TiRex 1 (`tirex-ts`), das unter der NXAI Community License steht.

## Daten im Benchmark (`live/benchmark.py`)

Das Skript lädt die Daten zur Laufzeit von ihren Quellen, es verteilt keine Daten, und das Repository enthält keine.

| Quelle | Hinweis |
|---|---|
| ETT (Transformer-Daten, zhouhaoyi/ETDataset) | CC BY-ND 4.0 (LICENSE im Repository). Abgeleitete Datenfassungen (z. B. Tagesmittel) nicht weitergeben; die README nennt nur Kennzahlen. |
| jbrownlee/Datasets | Das Repository enthält keine LICENSE-Datei; die Reihen stammen ursprünglich aus anderen Quellen. Nicht weiterverteilen. |
| Rdatasets (fpp2 u. a.) | Lizenz je Datensatz verschieden; nur zum Abruf und Auswerten genutzt. |
