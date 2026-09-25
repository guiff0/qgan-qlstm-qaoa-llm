"""
Config loader. Every experiment script pulls settings from
config/default_config.yaml through this module rather than
redefining defaults locally — this was one of the main sources
of drift in the original code (e.g. n_qubits=20 in one file,
20-qubit assumed but not enforced in another).
"""
import os
import yaml

_CONFIG_CACHE = None


def load_config(path: str = None) -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config",
            "default_config.yaml",
        )

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def merge_override(base_cfg: dict, override: dict) -> dict:
    """Shallow-merge an override dict (e.g. for ablations) into a base config section."""
    merged = dict(base_cfg)
    merged.update(override)
    return merged
