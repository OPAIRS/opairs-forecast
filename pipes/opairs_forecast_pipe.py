"""
title: OPAIRS Forecast
author: OPAIRS Systems
version: 0.3.0
description: Test-/Beispiel-Pipe für OpenWebUI: Zeitreihenprognose, Backtest und Anomalieerkennung über den LiteLLM-MCP-Gateway (selbst gehosteter Forecast-Service). Daten kommen als SQL-Block (Postgres, nur lesend) oder als CSV direkt im Chat. Die Pipe enthält nur Logik, die Modelle laufen im Service.
requirements: requests, psycopg2-binary
"""

# Hinweis: Diese Pipe ist eine Test- und Beispielintegration für OpenWebUI, kein Produktivbestandteil des Dienstes.
# Sie zeigt, wie man den Forecast-Service über einen LiteLLM-MCP-Gateway anspricht. Zugangsdaten (LiteLLM-Key,
# optional DB_DSN) werden nicht im Code gespeichert, sondern nach der Installation in den Valves der Pipe eingetragen.
# Produktivbetrieb: eigenen, nur lesenden DB-Benutzer und einen eigenen LiteLLM-Virtual-Key verwenden.
#
# OPAIRS Systems: https://opairs-systems.com/
# Funktioniert am besten zusammen mit dem OPAIRS SQL Agent:
# https://opairs-systems.com/insights/opairs-sql-agent-datenbanken-adapter-vergleich

import asyncio
import csv
import io
import json
import re
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

import requests
from pydantic import BaseModel, Field

HELP = """### OPAIRS Forecast

Ich brauche eine Zeitreihe mit den Spalten **ds** (Datum), **y** (Wert) und optional **unique_id** (Serie, z. B. Artikel).

**Variante 1: SQL** (nur `SELECT`, nur lesend)
````
Prognose h=12
```sql
SELECT date_trunc('month', datum)::date AS ds, SUM(menge) AS y, artikel AS unique_id
FROM verbrauch GROUP BY 1, 3 ORDER BY 3, 1
```
````

**Variante 2: CSV im Chat**
````
Prognose h=8
```csv
ds,y
2025-01-06,100
2025-01-13,104
...
```
````

**Aktionen:** `Prognose` (Standard) · `Backtest` · `Anomalien` · `Modelle`
**Optionen:** `h=12` · `freq=W-MON` (sonst automatisch) · `models=seasonal_naive,tirex_2` (sonst automatisch) · `level=80` · `windows=3` · `season=12` (Saisonlänge, sonst automatisch) · `ensemble=aus` (nur Einzelmodelle)
"""

ALIASES = {
    "ds": ("ds", "date", "datum", "timestamp", "zeit", "period", "periode", "monat", "woche", "tag"),
    "y": ("y", "value", "wert", "menge", "qty", "quantity", "betrag", "umsatz", "verbrauch"),
    "unique_id": ("unique_id", "id", "serie", "series", "artikel", "item", "key"),
}


def _num(x, rel=False):
    if x is None:
        return "–"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if rel:
        return f"{x:.1e}" if 0 < abs(x) < 0.001 else f"{x:.3f}"
    if x == 0:
        return "0"
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4f}"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _text_of(body: dict) -> str:
    for m in reversed(body.get("messages", [])):
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return c or ""
    return ""


def _parse_options(text: str) -> dict:
    t = text.lower()
    o = {}

    def grab(pattern, key, cast=str):
        m = re.search(pattern, t)
        if m:
            o[key] = cast(m.group(1))

    grab(r"\b(?:h|horizon|horizont|schritte)\s*[=:]\s*(\d+)", "h", int)
    if "h" not in o:
        grab(r"\b(?:nächste[nr]?|next|naechste[nr]?)\s+(\d+)\b", "h", int)
    grab(r"\bfreq(?:uenz)?\s*[=:]\s*([\w-]+)", "freq")
    if "freq" in o:  # Groß-/Kleinschreibung ist bei pandas-Frequenzen relevant (z. B. W-MON, MS)
        m = re.search(r"\bfreq(?:uenz)?\s*[=:]\s*([\w-]+)", text, re.I)
        o["freq"] = m.group(1)
    grab(r"\blevel\s*[=:]\s*(\d+)", "level", int)
    grab(r"\b(?:season|saison|season_length)\s*[=:]\s*(\d+)", "season", int)
    if re.search(r"\bensemble\s*[=:]\s*(aus|off|nein|false|0)\b", t):
        o["ensemble"] = False
    grab(r"\b(?:windows|fenster)\s*[=:]\s*(\d+)", "windows", int)
    m = re.search(r"\bmodel(?:s|le)?\s*[=:]\s*([\w,]+)", t)
    if m:
        o["models"] = [x for x in m.group(1).split(",") if x]
    if "backtest" in t:
        o["action"] = "backtest"
    elif any(k in t for k in ("anomal", "ausreißer", "ausreisser")):
        o["action"] = "anomalies"
    elif re.search(r"\bmodelle\b|\blist_models\b", t) and "models" not in o:
        o["action"] = "models"
    else:
        o["action"] = "forecast"
    return o


def _jsonable(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _to_float(v):
    if v is None or v == "":
        return None
    if isinstance(v, str):
        s = v.strip().replace(" ", "")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        elif "," in s and "." in s:  # 1.234,56 -> 1234.56
            s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None
    return float(v)


def _records(columns: list, rows: list) -> list:
    cols = [str(c).strip().lower() for c in columns]
    pick = {}
    for target, names in ALIASES.items():
        for n in names:
            if n in cols and cols.index(n) not in pick.values():
                pick[target] = cols.index(n)
                break
    if len(cols) == 2 and ("ds" not in pick or "y" not in pick):
        pick = {"ds": 0, "y": 1}
    if "ds" not in pick or "y" not in pick:
        raise ValueError(
            f"Spalten nicht erkannt: {', '.join(map(str, columns))}. Erwartet: ds (Datum), y (Wert), optional unique_id."
        )
    recs = []
    for r in rows:
        rec = {
            "unique_id": str(_jsonable(r[pick["unique_id"]])) if "unique_id" in pick else "serie",
            "ds": str(_jsonable(r[pick["ds"]])),
            "y": _to_float(_jsonable(r[pick["y"]])),
        }
        recs.append(rec)
    return recs


def _parse_csv(code: str):
    lines = [ln for ln in code.strip().splitlines() if ln.strip() and not ln.strip().startswith("...")]
    if len(lines) < 2:
        raise ValueError("CSV braucht eine Kopfzeile und Datenzeilen.")
    head = lines[0]
    delim = max((";", "\t", ","), key=head.count)
    rows = list(csv.reader(lines, delimiter=delim))
    return rows[0], rows[1:]


class Pipe:
    class Valves(BaseModel):
        LITELLM_URL: str = Field(default="http://litellm:4000", description="Basis-URL von LiteLLM")
        LITELLM_API_KEY: str = Field(default="", description="LiteLLM-Key (Virtual Key oder Master Key)")
        MCP_SERVER_NAME: str = Field(
            default="opairs_forecast", description="Name des Forecast-MCP-Servers in der LiteLLM-Konfiguration"
        )
        MCP_SERVER_ID: str = Field(
            default="",
            description="server_id des Forecast-MCP in LiteLLM. Leer lassen: wird über MCP_SERVER_NAME automatisch gefunden.",
        )
        DB_DSN: str = Field(
            default="",
            description="Postgres-DSN für SQL-Blöcke, z. B. postgresql://user:pw@host:5432/db (am besten ein Nur-Lese-User)",
        )
        SQL_TIMEOUT_S: int = Field(default=60, description="Abbruch langer Abfragen")
        MAX_ROWS: int = Field(default=200000, description="Maximale Zeilenzahl pro Abfrage")
        DEFAULT_H: int = Field(default=12, description="Horizont, wenn keiner angegeben wird")
        DEFAULT_WINDOWS: int = Field(
            default=4,
            description="Backtest-Fenster, wenn keine angegeben (mehr Fenster = verlässlichere Modellwahl; bei kurzer Historie wird automatisch reduziert)",
        )
        HTTP_TIMEOUT_S: int = Field(default=900, description="Timeout für den Forecast-Aufruf")
        SHOW_MAX_SERIES: int = Field(default=5, description="Wie viele Serien ausführlich angezeigt werden")
        SHOW_MAX_ANOMALIES: int = Field(default=50, description="Maximale Anzahl angezeigter Anomalien")

    def __init__(self):
        self.name = "OPAIRS Forecast"
        self.valves = self.Valves()

    # ---------- Datenquellen ----------
    def _run_sql(self, sql: str):
        if not self.valves.DB_DSN:
            raise ValueError("DB_DSN ist in den Valves der Pipe nicht gesetzt. SQL-Blöcke sind deshalb nicht möglich.")
        q = sql.strip().rstrip(";").strip()
        if ";" in q:
            raise ValueError("Nur ein einzelnes Statement erlaubt (kein ';' im Text).")
        if not re.match(r"(?is)^(select|with)\b", q):
            raise ValueError("Nur SELECT/WITH erlaubt.")
        import psycopg2  # lazy: nur bei SQL nötig

        conn = psycopg2.connect(self.valves.DB_DSN, connect_timeout=10)
        try:
            conn.set_session(readonly=True, autocommit=False)
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {int(self.valves.SQL_TIMEOUT_S) * 1000}")
                cur.execute(q)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchmany(self.valves.MAX_ROWS + 1)
        finally:
            conn.close()
        if len(rows) > self.valves.MAX_ROWS:
            raise ValueError(f"Mehr als {self.valves.MAX_ROWS} Zeilen. Bitte aggregieren oder einschränken.")
        return cols, rows

    # ---------- LiteLLM-MCP ----------
    def _server_id(self) -> str:
        v = self.valves
        if v.MCP_SERVER_ID:
            return v.MCP_SERVER_ID
        if getattr(self, "_sid", None):
            return self._sid
        r = requests.get(
            f"{v.LITELLM_URL.rstrip('/')}/v1/mcp/server",
            headers={"Authorization": f"Bearer {v.LITELLM_API_KEY}"},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"MCP-Serverliste nicht abrufbar (HTTP {r.status_code}). MCP_SERVER_ID manuell setzen.")
        raw = r.json()
        items = raw if isinstance(raw, list) else raw.get("servers") or raw.get("data") or [raw]
        for x in items:
            if isinstance(x, dict) and v.MCP_SERVER_NAME in json.dumps(x) and x.get("server_id"):
                self._sid = x["server_id"]
                return self._sid
        raise RuntimeError(f"MCP-Server '{v.MCP_SERVER_NAME}' in LiteLLM nicht gefunden. MCP_SERVER_ID manuell setzen.")

    def _call(self, name: str, arguments: dict) -> dict:
        v = self.valves
        r = requests.post(
            f"{v.LITELLM_URL.rstrip('/')}/mcp-rest/tools/call",
            headers={"Authorization": f"Bearer {v.LITELLM_API_KEY}"},
            json={"server_id": self._server_id(), "name": name, "arguments": arguments},
            timeout=v.HTTP_TIMEOUT_S,
        )
        if r.status_code != 200:
            raise RuntimeError(f"LiteLLM HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        text = "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text")
        if j.get("isError"):
            raise RuntimeError(re.sub(r"^Error executing tool \w+:\s*", "", text) or "Tool-Fehler")
        try:
            return json.loads(text)
        except ValueError:
            raise RuntimeError(f"Unerwartete Antwort vom Service: {text[:300]}")

    def _call_windows(self, name: str, args: dict, windows: int, explicit: bool, key: str = "n_windows") -> dict:
        """Ruft das Tool auf; bei zu kurzer Historie werden die Fenster schrittweise reduziert (nur wenn nicht explizit gesetzt)."""
        w = max(windows, 1)
        while True:
            try:
                return self._call(name, {**args, key: w})
            except RuntimeError as e:
                if explicit or w <= 1 or "lang genug" not in str(e):
                    raise
                w -= 1

    # ---------- Darstellung ----------
    @staticmethod
    def _backtest_table(scores: dict, chosen: str) -> str:
        rows = []
        for name, s in sorted(scores.items(), key=lambda kv: kv[1].get("relative_mae", 9e9)):
            mark = "✅" if s.get("beats_baseline") else ("·" if s.get("relative_mae") == 1.0 else "✗")
            label = f"**{name}**" if name == chosen else name
            rows.append(
                [label, _num(s.get("relative_mae"), rel=True), _num(s.get("mae")), _num(s.get("interval_coverage"), rel=True), mark]
            )
        return _table(["Modell", "rel. MAE", "MAE", "Intervall-Abdeckung", "schlägt Baseline"], rows)

    @staticmethod
    def _status(validated) -> str:
        if validated is True:
            return "✅ validiert"
        if validated is False:
            return "⚠️ nicht validiert (Baseline)"
        return "nicht geprüft"

    def _render_forecast(self, res: dict) -> str:
        sel = res["selected_model"]
        scores = res.get("backtest") or {}
        head = (
            f"### Prognose\n**Modell:** `{sel}` · {self._status(res.get('validated'))}"
            + (f" (rel. MAE {_num(scores[sel].get('relative_mae'), rel=True)} vs. `{res.get('baseline')}`)" if sel in scores else "")
            + f"\n**Frequenz:** {res.get('freq')} · **Horizont:** {res.get('h')} · **Intervall:** {res.get('level')} % · **Serien:** {res.get('n_series')}"
        )
        parts = [head]
        for w in res.get("warnings") or []:
            parts.append(f"> ⚠️ {w}")
        by = {}
        for r in res.get("forecast", []):
            by.setdefault(r["unique_id"], []).append(r)
        shown = list(by.items())[: self.valves.SHOW_MAX_SERIES]
        for uid, recs in shown:
            if len(by) > 1:
                parts.append(f"**Serie `{uid}`**")
            parts.append(
                _table(
                    ["Datum", "Prognose", "untere Grenze", "obere Grenze"],
                    [[str(r["ds"])[:10], _num(r["yhat"]), _num(r.get("lo")), _num(r.get("hi"))] for r in recs],
                )
            )
        if len(by) > len(shown):
            parts.append(f"_… und {len(by) - len(shown)} weitere Serien (Anzeige begrenzt)._")
        if scores:
            parts.append("**Modellvergleich (Backtest)**\n\n" + self._backtest_table(scores, sel))
        if res.get("errors"):
            parts.append("**Modellfehler:** " + "; ".join(f"`{k}`: {v}" for k, v in res["errors"].items()))
        return "\n\n".join(parts)

    def _render_backtest(self, res: dict) -> str:
        best = res["best_model"]
        parts = [
            f"### Backtest\n**Bestes Modell:** `{best}` · {self._status(res.get('validated'))} · "
            f"Frequenz {res.get('freq')} · Horizont {res.get('h')} · Fenster {res.get('n_windows')}"
        ]
        for w in res.get("warnings") or []:
            parts.append(f"> ⚠️ {w}")
        parts.append(self._backtest_table(res.get("backtest") or {}, best))
        if res.get("errors"):
            parts.append("**Modellfehler:** " + "; ".join(f"`{k}`: {v}" for k, v in res["errors"].items()))
        return "\n\n".join(parts)

    def _render_anomalies(self, res: dict) -> str:
        parts = [
            f"### Anomalien\n**Modell:** `{res.get('model')}` · Intervall {res.get('level')} % · "
            f"**{res.get('n_anomalies')}** Auffälligkeiten in {res.get('n_points')} geprüften Punkten"
        ]
        for w in res.get("warnings") or []:
            parts.append(f"> ⚠️ {w}")
        an = res.get("anomalies") or []
        if an:
            lim = self.valves.SHOW_MAX_ANOMALIES
            parts.append(
                _table(
                    ["Serie", "Datum", "Ist", "erwartet", "Bereich", "Richtung"],
                    [
                        [
                            a["unique_id"],
                            str(a["ds"])[:10],
                            _num(a["y"]),
                            _num(a.get("expected")),
                            f"{_num(a.get('lo'))} … {_num(a.get('hi'))}",
                            a.get("direction", ""),
                        ]
                        for a in an[:lim]
                    ],
                )
            )
            if len(an) > lim:
                parts.append(f"_… {len(an) - lim} weitere nicht angezeigt._")
        return "\n\n".join(parts)

    # ---------- Hauptlogik ----------
    def _work(self, text: str) -> str:
        t0 = time.time()
        blocks = re.findall(r"```(\w*)[^\n]*\n(.*?)```", text, re.S)
        prose = re.sub(r"```.*?```", " ", text, flags=re.S)
        opts = _parse_options(prose)

        if opts["action"] == "models":
            res = self._call("list_models", {})
            rows = [
                [
                    m["name"],
                    m["type"],
                    "ja" if m["available"] else f"nein ({m.get('reason', '')})",
                    "Baseline" if m.get("baseline") else ("Kandidat" if m.get("default_candidate") else ""),
                ]
                for m in res["models"]
            ]
            return "### Modelle\n\n" + _table(["Modell", "Typ", "verfügbar", "Rolle"], rows)

        data_block = next(((lang.lower(), code) for lang, code in blocks if lang.lower() in ("sql", "csv", "")), None)
        if not data_block:
            return HELP
        lang, code = data_block
        if lang == "sql" or (lang == "" and re.match(r"(?is)^\s*(select|with)\b", code)):
            cols, rows = self._run_sql(code)
            source = "SQL"
        else:
            cols, rows = _parse_csv(code)
            source = "CSV"
        recs = _records(cols, rows)
        if not recs:
            return "Die Abfrage lieferte keine Zeilen."
        n_series = len({r["unique_id"] for r in recs})

        h = opts.get("h", self.valves.DEFAULT_H)
        args = {"series": recs}
        if opts.get("freq"):
            args["freq"] = opts["freq"]
        if opts.get("season"):
            args["season_length"] = opts["season"]
        if opts.get("ensemble") is False:
            args["ensemble"] = False
        if opts["action"] == "forecast":
            args.update(h=h, level=opts.get("level", 80))
            if opts.get("models"):
                args["models"] = opts["models"]
            out = self._render_forecast(
                self._call_windows("forecast", args, opts.get("windows", self.valves.DEFAULT_WINDOWS), "windows" in opts)
            )
        elif opts["action"] == "backtest":
            args.update(h=h)
            if opts.get("models"):
                args["models"] = opts["models"]
            out = self._render_backtest(
                self._call_windows("backtest", args, opts.get("windows", self.valves.DEFAULT_WINDOWS), "windows" in opts)
            )
        else:
            args.pop("ensemble", None)  # Anomalieerkennung nutzt immer ein Einzelmodell
            args.update(h=h, n_windows=opts.get("windows", 4), level=opts.get("level", 99))
            if opts.get("models"):
                args["model"] = opts["models"][0]
            out = self._render_anomalies(self._call("detect_anomalies", args))
        return f"{out}\n\n_Quelle: {source} · {len(recs)} Zeilen · {n_series} Serie(n) · {time.time() - t0:.1f} s_"

    async def pipe(self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None) -> str:
        text = _text_of(body)
        if text.lstrip().startswith("### Task:"):  # OpenWebUI-Hilfsaufgaben (Titel, Tags) nicht verarbeiten
            return "Forecast"

        async def status(msg, done=False):
            if __event_emitter__:
                try:
                    await __event_emitter__({"type": "status", "data": {"description": msg, "done": done}})
                except Exception:
                    pass

        await status("Berechne … (erster Aufruf nach Start kann ~20 s dauern)")
        try:
            result = await asyncio.to_thread(self._work, text)
        except Exception as e:
            await status("Fehler", done=True)
            return f"**Fehler:** {e}"
        await status("Fertig", done=True)
        return result
