from __future__ import annotations

from functools import lru_cache

from .adapters.base import Adapter
from .adapters.chronos import ChronosAdapter
from .adapters.stats import StatsAdapter
from .adapters.tirex import TiRex2Adapter
from .config import DEVICE, load_config


class UnknownModel(ValueError):
    pass


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
    raise ValueError(f"Unbekannter Modelltyp '{t}' bei '{name}'")


@lru_cache(maxsize=None)
def get_adapter(name: str) -> Adapter:
    spec = load_config()["models"].get(name)
    if spec is None:
        raise UnknownModel(f"Unbekanntes Modell '{name}'. Verfügbar: {', '.join(load_config()['models'])}")
    return _build(name, spec)


def baseline_name() -> str:
    return load_config()["baseline"]


def resolve_models(requested: list[str] | None) -> list[str]:
    """['auto'] oder None -> Standardkandidaten (nur verfügbare). Baseline ist immer dabei."""
    cfg = load_config()
    if not requested or requested == ["auto"]:
        names = list(cfg["default_candidates"])
        names = [n for n in names if n == baseline_name() or get_adapter(n).available()[0]]
    else:
        names = list(requested)
        for n in names:
            ok, why = get_adapter(n).available()  # wirft UnknownModel bei unbekanntem Namen
            if not ok:
                raise UnknownModel(f"Modell '{n}' ist nicht verfügbar: {why}")
    if baseline_name() not in names:
        names.insert(0, baseline_name())
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
            }
        )
    return rows
