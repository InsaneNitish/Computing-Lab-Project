"""Shared utility functions for ForgetLab.

All functions here are pure Python (no torch, no model) so tests run on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


# ── Wilson confidence interval ────────────────────────────────────────────────

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % confidence interval for a proportion.

    Returns (lo, hi) in [0, 1].  When n == 0 returns (0.0, 1.0).
    """
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ── Repetition / degeneracy detection ────────────────────────────────────────

def repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-grams in *text* that are repeated (i.e. occur more than once).

    A clean generation has ratio near 0; a looping output has ratio near 1.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(grams), 1)


def is_looping(text: str, ratio_threshold: float = 0.5, n: int = 4) -> bool:
    """True if *text* appears to be a degenerate repetition loop."""
    return repetition_ratio(text, n) >= ratio_threshold


# ── Metadata builder ──────────────────────────────────────────────────────────

def build_metadata(
    base_model: str,
    injection_version: str,
    gen_cap: int,
    n: int,
    seed: int,
    method: str,
    hyperparameters: dict[str, Any],
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Return a metadata dict that must accompany every result JSON.

    Never mix results with different metadata in one table.
    """
    return {
        "base_model": base_model,
        "injection_version": injection_version,
        "gen_cap": gen_cap,
        "n": n,
        "seed": seed,
        "method": method,
        "hyperparameters": hyperparameters,
        "git_commit": git_commit or _get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ── JSON helpers ──────────────────────────────────────────────────────────────

def json_save(obj: Any, path: str, indent: int = 2) -> None:
    """Write *obj* to *path* (creates parent dirs as needed)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def json_load(path: str) -> Any:
    """Load JSON from *path*, returning None if the file is missing or malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Short ID generation ────────────────────────────────────────────────────────

def short_id(length: int = 8) -> str:
    """Generate a random short hex ID (suitable for job IDs)."""
    import secrets
    return secrets.token_hex(length // 2)


def stable_id(*parts: str) -> str:
    """Deterministic 8-char ID from the given string parts (useful for fact IDs)."""
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:8]


# ── Text normalization ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation — for key-token recall."""
    import unicodedata
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


# ── Dataclass helpers ─────────────────────────────────────────────────────────

def as_dict(obj: Any) -> dict:
    """Convert a dataclass to a plain dict (recursively)."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj

