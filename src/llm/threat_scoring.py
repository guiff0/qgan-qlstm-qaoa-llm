"""
Parses a threat probability out of the LLM's raw text output and applies
the QACL table's decision threshold.

WHY THIS WAS MISSING: classification_metrics() (metrics.py) and the
M3->DV3 / M5->DV5 mediation pathways all need a y_pred_proba array --
but nothing anywhere in this codebase turned the LLM's actual generated
text into that probability. This is the missing link between "the LLM
produced an answer" and "we have a number to threshold."

Ported and adapted from an external repository audit: the regex-parsing
approach here is real and reusable, unlike most of the rest of that
repository's detection-metrics layer, which fabricated y_pred_proba via
np.random.default_rng(0).uniform(0, 1, ...) -- i.e., ignored the LLM's
output entirely. Flagging that explicitly since this file's whole
purpose is to be the thing that DOESN'T do that.
"""
from __future__ import annotations

import re

_THREAT_PROB_PATTERN = re.compile(
    r"threat[_ ]probability[:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE
)


def parse_threat_probability(llm_output_text: str) -> float | None:
    """
    Extracts a threat probability from the LLM's free-text output, which
    is expected to contain a line resembling "Threat probability: 0.83"
    (per this study's prompting protocol). Returns None -- NOT a
    fabricated default -- when no such value is found, so callers can
    distinguish "the LLM didn't report a threat probability" from "the
    LLM reported zero," which are very different things for an FPR
    computation.
    """
    match = _THREAT_PROB_PATTERN.search(llm_output_text)
    if match is None:
        return None
    value = float(match.group(1))
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"Parsed threat probability {value} is outside [0, 1] -- the LLM's output "
            f"format may not match what this parser expects: {llm_output_text[:200]!r}"
        )
    return value


def classify(probability: float, threshold: float = 0.70) -> str:
    """QACL table's decision rule: probability > threshold -> 'threat'."""
    return "threat" if probability > threshold else "benign"


def parse_batch(llm_output_texts: list[str], threshold: float = 0.70,
                 on_missing: str = "raise") -> dict:
    """
    Parses a batch of LLM outputs into probabilities and threat/benign
    labels. on_missing controls what happens when parse_threat_probability
    returns None for some entry: "raise" (default -- fail loudly, since a
    silently-dropped or silently-zeroed sample would quietly bias FPR/
    accuracy) or "drop" (exclude that entry, returning fewer results than
    inputs -- only use this if you have a specific, documented reason a
    missing value should be excluded rather than investigated).
    """
    if on_missing not in ("raise", "drop"):
        raise ValueError(f"on_missing must be 'raise' or 'drop', got {on_missing!r}")

    probabilities, labels, kept_indices = [], [], []
    for i, text in enumerate(llm_output_texts):
        prob = parse_threat_probability(text)
        if prob is None:
            if on_missing == "raise":
                raise ValueError(
                    f"No threat probability found in output at index {i}: {text[:200]!r}"
                )
            continue
        probabilities.append(prob)
        labels.append(classify(prob, threshold))
        kept_indices.append(i)

    return {
        "probabilities": probabilities,
        "labels": labels,
        "kept_indices": kept_indices,
        "n_missing": len(llm_output_texts) - len(kept_indices),
        "threshold": threshold,
    }
