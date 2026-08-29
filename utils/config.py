"""YAML config loading with inheritance and command-line overrides.

Every run is defined by a config file plus an explicit list of overrides, and the
fully resolved config is written into the run directory. Reproducing a result
means re-running that saved file - there is no hidden state in a shell history
or in a default that changed between runs.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

BASE_KEY = "_base_"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML config, resolving `_base_` inheritance.

    `_base_` may be a single path or a list, relative to the config's own
    directory. Later entries win over earlier ones, and the file's own keys win
    over all of them.
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ chain at {path}")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    _seen = _seen | {path}

    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: a config must be a mapping, got {type(cfg).__name__}")

    bases = cfg.pop(BASE_KEY, None)
    if bases is None:
        return cfg

    merged: dict = {}
    for base in [bases] if isinstance(bases, str) else bases:
        merged = _deep_merge(merged, load_config(path.parent / base, _seen))
    return _deep_merge(merged, cfg)


def _coerce(text: str) -> Any:
    """Parse an override value with YAML rules, so types survive the shell."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `dotted.key=value` overrides.

    Raises on a key that does not already exist, because a silently-created key
    is a typo that runs to completion and produces a result that answers a
    different question from the one intended.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")

        node = cfg
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"override {dotted!r}: no config section {key!r}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(
                f"override {dotted!r}: no such key. Add it to the config file "
                f"first if it is genuinely new."
            )
        node[keys[-1]] = _coerce(raw)
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested value by dotted path."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def save_config(cfg: dict, path: str | Path) -> Path:
    """Write a resolved config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, default_flow_style=False)
    return path
