"""
LLAMA 3.3 fine-tuning via NVIDIA's API.

The original code had a comment ("Note: LLM fine-tuning would happen
here / This is simplified; actual implementation would call LLM
fine-tune") with no actual call. This module fills that in as a real
API integration — but it needs YOUR NVIDIA_API_KEY environment
variable to do anything. It will not silently fall back to a stub;
it raises clearly if the key is missing, since a silent no-op here
would reintroduce exactly the kind of fabricated-results problem this
whole exercise is trying to eliminate.

NVIDIA's build.nvidia.com / NIM fine-tuning API surface changes over
time; verify the exact endpoint and payload shape against NVIDIA's
current documentation before running this against a real account —
treat the request shape below as a starting point, not a guarantee.
"""
from __future__ import annotations

import json
import os
import time

import requests


class NvidiaApiKeyMissingError(RuntimeError):
    pass


def _get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise NvidiaApiKeyMissingError(
            "\n\nNVIDIA_API_KEY is not set. This code will not fabricate "
            "fine-tuning results — it needs your real API key.\n"
            "Set it with:\n    export NVIDIA_API_KEY='your-key-here'\n"
            "See SETUP.md, section 'NVIDIA API access', for how to obtain one.\n"
        )
    return key


def build_training_examples(real_texts: list[str], synthetic_texts: list[str],
                             real_ratio: float, synthetic_ratio: float) -> list[dict]:
    """Combines real and QGAN-synthetic-derived text examples in the
    configured ratio (default 60% real / 40% synthetic per Ch.3).

    Subsamples whichever pool is over-represented so the returned mix
    actually matches real_ratio:synthetic_ratio, rather than just
    concatenating both pools in full (which silently ignores the ratio
    the moment the two input lists aren't already in that proportion)."""
    total_ratio = real_ratio + synthetic_ratio
    target_real_frac = real_ratio / total_ratio

    n_real, n_synth = len(real_texts), len(synthetic_texts)
    if n_real == 0 or n_synth == 0:
        selected_real, selected_synth = real_texts, synthetic_texts
    else:
        # Find the largest (n_real, n_synth) pair in the target ratio that
        # fits within the available pools.
        max_total_from_real = n_real / target_real_frac
        max_total_from_synth = n_synth / (1 - target_real_frac)
        total = min(max_total_from_real, max_total_from_synth)
        take_real = round(total * target_real_frac)
        take_synth = round(total * (1 - target_real_frac))
        selected_real = real_texts[:take_real]
        selected_synth = synthetic_texts[:take_synth]

    examples = [{"text": t, "source": "real"} for t in selected_real]
    examples += [{"text": t, "source": "synthetic"} for t in selected_synth]
    return examples


def submit_finetune_job(training_examples: list[dict], base_model: str,
                         epochs: int, api_base: str = "https://api.nvidia.com/v1") -> dict:
    """
    Submits a fine-tuning job. This is a real HTTP call — it will fail
    loudly (not silently return fake success) if the key is invalid,
    the endpoint is wrong, or the account lacks fine-tuning access.
    """
    api_key = _get_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": base_model,
        "training_data": training_examples,
        "hyperparameters": {"n_epochs": epochs},
    }

    response = requests.post(f"{api_base}/fine_tuning/jobs", headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(
            f"NVIDIA fine-tuning API returned {response.status_code}: {response.text}\n"
            f"Check your API key's fine-tuning entitlement and the base_model name "
            f"against NVIDIA's current NIM catalog."
        )
    return response.json()


def poll_finetune_status(job_id: str, api_base: str = "https://api.nvidia.com/v1",
                          poll_interval_sec: int = 30, timeout_sec: int = 3600 * 12) -> dict:
    """Polls until the job completes, fails, or the timeout is hit.
    Fine-tuning jobs at this scale can take hours; this is a blocking
    poll loop meant to run inside a long-lived job, not an interactive
    session (see RUNNING.md for how to launch this as a background job)."""
    api_key = _get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.time()

    while time.time() - start < timeout_sec:
        resp = requests.get(f"{api_base}/fine_tuning/jobs/{job_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        status = resp.json()
        if status.get("status") in ("succeeded", "failed", "cancelled"):
            return status
        time.sleep(poll_interval_sec)

    raise TimeoutError(f"Fine-tuning job {job_id} did not complete within {timeout_sec} seconds.")


def save_job_manifest(job_response: dict, out_path: str = "logs/llm_finetune_job.json"):
    """Saves the job ID and submission details so the job can be
    reattached to / polled later even if this process exits."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(job_response, f, indent=2)
