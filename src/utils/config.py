"""
Config Loader
=============
YAML-based configuration system with dot-accessible objects, deep merging,
and reproducibility saving.

Public API
----------
load_config(path)               → DotDict
merge_configs(base, override)   → DotDict  (deep merge, non-mutating)
save_config(cfg, path)          → None     (YAML dump for reproducibility)
config_to_dict(cfg)             → dict     (for JSON logging / W&B)

Quick example::

    from src.utils.config import load_config, merge_configs, save_config

    cfg = load_config("experiments/baseline_m3dusa.yaml")

    # Dot-notation access:
    print(cfg.experiment.name)       # "baseline_m3dusa"
    print(cfg.model.gat.num_layers)  # 2

    # CLI override:
    cfg = merge_configs(cfg, {"training": {"lr": 1e-4}})

    # Persist active config alongside results:
    save_config(cfg, "results/baseline_m3dusa/config.yaml")
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Union

import yaml


# ---------------------------------------------------------------------------
# DotDict — dict subclass with attribute-style (dot-notation) access
# ---------------------------------------------------------------------------

class DotDict(dict):
    """A ``dict`` subclass that supports attribute-style access.

    Nested dicts are also converted to ``DotDict`` instances, so any depth of
    dot-notation works::

        cfg = DotDict({"model": {"gat": {"num_layers": 2}}})
        assert cfg.model.gat.num_layers == 2
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no key '{key}'. "
                f"Available keys: {list(self.keys())}"
            ) from None
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __repr__(self) -> str:
        return f"DotDict({dict.__repr__(self)})"


def _to_dotdict(obj: Any) -> Any:
    """Recursively convert nested dicts / lists to ``DotDict`` instances."""
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(item) for item in obj]
    return obj


def _from_dotdict(obj: Any) -> Any:
    """Recursively convert ``DotDict`` instances back to plain dicts."""
    if isinstance(obj, dict):
        return {k: _from_dotdict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dotdict(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> DotDict:
    """Load a YAML config file and return a dot-accessible :class:`DotDict`.

    Args:
        path: Path to the YAML config file.

    Returns:
        A :class:`DotDict` with all config values accessible via dot notation.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML file is empty or not a mapping.

    Example::

        cfg = load_config("experiments/baseline_m3dusa.yaml")
        print(cfg.experiment.name)   # "baseline_m3dusa"
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must be a YAML mapping (got {type(raw).__name__}): {path}"
        )

    return _to_dotdict(raw)


def merge_configs(base: DotDict, override: dict) -> DotDict:
    """Deep-merge *override* into *base*, returning a **new** :class:`DotDict`.

    Nested dicts are merged recursively; scalar values in *override* take
    precedence over those in *base*. Neither argument is mutated.

    Args:
        base:     The base config (e.g. loaded from a YAML experiment file).
        override: A (possibly nested) plain ``dict`` of overrides (e.g. from
                  CLI flags or a sweep framework like Optuna / W&B).

    Returns:
        A new :class:`DotDict` with merged values.

    Example::

        cfg = load_config("experiments/baseline_m3dusa.yaml")
        cfg = merge_configs(cfg, {"training": {"lr": 1e-4}, "experiment": {"seed": 7}})
        print(cfg.training.lr)       # 1e-4   (overridden)
        print(cfg.training.epochs)   # 50     (preserved from base)
    """

    def _deep_merge(dst: dict, src: dict) -> dict:
        for key, val in src.items():
            if isinstance(val, dict) and isinstance(dst.get(key), dict):
                dst[key] = _deep_merge(dict(dst[key]), val)
            else:
                dst[key] = copy.deepcopy(val)
        return dst

    merged = _deep_merge(copy.deepcopy(dict(base)), override)
    return _to_dotdict(merged)


def save_config(cfg: DotDict, path: Union[str, Path]) -> None:
    """Serialise *cfg* to a YAML file for experiment reproducibility.

    Parent directories are created automatically if they do not exist.

    Args:
        cfg:  The :class:`DotDict` config to serialise.
        path: Destination file path (created / overwritten).

    Example::

        save_config(cfg, "results/baseline_m3dusa/config.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(
            _from_dotdict(cfg),
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def config_to_dict(cfg: DotDict) -> dict:
    """Recursively convert a :class:`DotDict` to a plain ``dict``.

    Useful for JSON-serialisation, W&B ``config`` logging, or any API that
    does not accept custom dict subclasses.

    Args:
        cfg: The config object to flatten.

    Returns:
        A plain nested ``dict``.
    """
    return _from_dotdict(cfg)
