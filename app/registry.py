from __future__ import annotations

from functools import lru_cache

from .adapters.base import Adapter, CovSpec
from .adapters.chronos import ChronosAdapter
from .adapters.stats import StatsAdapter
from .adapters.timestral import TimestralAdapter
from .adapters.tirex import TiRex2Adapter
from .config import DEVICE, load_config


class UnknownModel(ValueError):
    pass


COV_SUFFIX = "@cov"


def split_name(name: str) -> tuple[str, bool]:
    """'chronos_2@cov' -> ('chronos_2', True): das Modell mit Kovariaten. Ohne Suffix läuft es nur auf der Zielreihe."""
    return (name[: -len(COV_SUFFIX)], True) if name.endswith(COV_SUFFIX) else (name, False)


def _build(name: str, spec: dict) -> Adapter:
    t = spec.get("type")
    if spec.get("enabled", True) is False:
        raise UnknownModel(f"Modell '{name}' ist deaktiviert")
    if t == "stats":
        return StatsAdapter(name, spec["class"], spec.get("params"))
    if t == "chronos":
        return ChronosAdapter(
            name,
            spec["path"],
            device=spec.get("device", DEVICE),
            batch_size=int(spec.get("batch_size", 64)),
            family=spec.get("family", "bolt"),
        )
    if t == "tirex2":
        return TiRex2Adapter(
            name, spec["path"], device=spec.get("device", DEVICE), batch_size=int(spec.get("batch_size", 256))
        )
    if t == "timestral":
        return TimestralAdapter(
            name,
            spec["path"],
            device=spec.get("device", DEVICE),
            batch_size=int(spec.get("batch_size", 32)),
            symmetric_averaging=bool(spec.get("symmetric_averaging", True)),
        )
    raise ValueError(f"Unbekannter Modelltyp '{t}' bei '{name}'")


@lru_cache(maxsize=None)
def get_adapter(name: str) -> Adapter:
    spec = load_config()["models"].get(name)
    if spec is None:
        raise UnknownModel(f"Unbekanntes Modell '{name}'. Verfügbar: {', '.join(load_config()['models'])}")
    return _build(name, spec)


def baseline_name() -> str:
    return load_config()["baseline"]


def resolve_models(requested: list[str] | None, intermittent: bool = False, cov: CovSpec | None = None) -> list[str]:
    """['auto'] oder None -> Standardkandidaten (nur verfügbare). Baseline ist immer dabei.

    intermittent=True (die Eingabe enthält intermittierende Serien) ergänzt die Kandidaten für sporadische Nachfrage.
    cov (Kovariaten vorhanden) ergänzt zu jedem Modell, das sie nutzen kann, die Variante `<name>@cov`. So zeigt der
    Backtest direkt, ob die Kovariaten helfen: dasselbe Modell mit und ohne."""
    cfg = load_config()
    # "auto" (oder keine Angabe) = Standardkandidaten; weitere Namen daneben ergänzen sie (z. B. ["auto", "timestral_28m"]).
    auto = not requested or "auto" in requested
    extra = [n for n in (requested or []) if n != "auto"]
    names: list[str] = []
    if auto:
        names = list(cfg["default_candidates"])
        if intermittent:
            names += list(cfg["intermittent_candidates"])
        names = [n for n in names if n == baseline_name() or get_adapter(n).available()[0]]
    for n in extra:
        base, use_cov = split_name(n)
        ad = get_adapter(base)  # wirft UnknownModel bei unbekanntem Namen
        ok, why = ad.available()
        if not ok:
            raise UnknownModel(f"Modell '{n}' ist nicht verfügbar: {why}")
        if use_cov and (cov is None or not ad.can_use(cov)):
            raise UnknownModel(
                f"'{n}': {base} kann die übergebenen Kovariaten nicht nutzen "
                "(keine Kovariaten angegeben, Modell ohne Kovariatenunterstützung oder keine bekannten Zukunftswerte)"
            )
        names.append(n)
    if baseline_name() not in names:
        names.insert(0, baseline_name())
    if cov is not None:
        for n in list(names):
            base, use_cov = split_name(n)
            if not use_cov and get_adapter(base).can_use(cov) and base + COV_SUFFIX not in names:
                names.append(base + COV_SUFFIX)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def describe_models() -> list[dict]:
    cfg = load_config()
    rows = []
    for name, spec in cfg["models"].items():
        enabled = spec.get("enabled", True) is not False
        ok, why = (False, "deaktiviert") if not enabled else get_adapter(name).available()
        rows.append(
            {
                "name": name,
                "type": spec.get("type"),
                "available": ok,
                "reason": why,
                "baseline": name == cfg["baseline"],
                "default_candidate": name in cfg["default_candidates"],
                "intermittent_candidate": name in cfg["intermittent_candidates"],
                "supports_covariates": False if not enabled else bool(get_adapter(name).supports_covariates),
            }
        )
    return rows
