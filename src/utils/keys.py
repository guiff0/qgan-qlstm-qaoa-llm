"""
Centralized API key checking -- ported and adapted from an external
repository audit. The pattern (fail loudly and immediately on a missing
or placeholder key, never proceed with a fabricated response) already
existed inline in src/llm/nvidia_finetune.py; this pulls it into one
reusable place so any future caller needing an API key gets the same
honest failure mode by default, rather than each new integration
reimplementing -- or forgetting to implement -- the check.
"""
from __future__ import annotations

import os


def get_key(name: str, required: bool = True, default: str | None = None) -> str | None:
    """
    Reads an environment variable. Raises immediately if `required` and
    the value is missing, empty, or still set to an obvious placeholder
    (e.g. "sk-placeholder..." left over from a .env.example template)
    -- this is deliberately strict: a placeholder key silently accepted
    as real would let calling code proceed as if authenticated and fail
    confusingly later (or, worse, appear to "work" against a mocked/
    cached response), rather than failing clearly at the point the key
    was actually needed.
    """
    val = os.environ.get(name, default)
    if required and (val is None or val == "" or str(val).startswith("sk-placeholder")):
        raise RuntimeError(
            f"Missing required key: {name}. Set it as an environment variable "
            f"(see SETUP.md) -- never hardcode it into a config file or script."
        )
    return val


def nvidia_api_key() -> str:
    return get_key("NVIDIA_API_KEY")


def fred_api_key() -> str:
    """Not required: src/data/data_loader.py currently reads FRED data
    from a local CSV file (see SETUP.md), not a live API call, so this
    is available for a future live-API code path without forcing every
    caller to have it set today."""
    return get_key("FRED_API_KEY", required=False, default="")
