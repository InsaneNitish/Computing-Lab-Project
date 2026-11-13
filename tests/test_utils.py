"""Tests for forgetlab.utils — pure Python, no GPU required."""

from __future__ import annotations

import pytest

from forgetlab.utils import (
    wilson_ci,
    repetition_ratio,
    is_looping,
    normalize_text,
    short_id,
    stable_id,
    build_metadata,
)


# ── Wilson CI ─────────────────────────────────────────────────────────────────

class TestWilsonCI:
    def test_zero_n(self):
        lo, hi = wilson_ci(0.5, 0)
        assert lo == 0.0 and hi == 1.0

    def test_known_values(self):
        # 50 successes out of 100 → 95% CI roughly (0.40, 0.60)
        lo, hi = wilson_ci(0.5, 100)
        assert 0.38 <= lo <= 0.42
        assert 0.58 <= hi <= 0.62

    def test_extreme_proportion(self):
        lo, hi = wilson_ci(1.0, 50)
        assert lo >= 0.0 and hi <= 1.0

    def test_zero_proportion(self):
        lo, hi = wilson_ci(0.0, 50)
        assert lo == 0.0 and hi < 0.1

    def test_over_one_proportion_no_math_domain_error(self):
        # Retention can exceed 1.0 when post-unlearning reveals more retain items
        lo, hi = wilson_ci(1.125, 50)
        assert lo >= 0.0 and hi <= 1.0

    def test_negative_proportion_clamped(self):
        lo, hi = wilson_ci(-0.2, 50)
        assert lo == 0.0 and hi < 0.1


# ── Repetition ratio ──────────────────────────────────────────────────────────

class TestRepetitionRatio:
    def test_clean_text(self):
        text = "The cat sat on the mat and looked around the room."
        ratio = repetition_ratio(text)
        assert ratio < 0.3

    def test_looping_text(self):
        text = "cat cat cat cat cat cat cat cat cat cat cat cat"
        ratio = repetition_ratio(text, n=1)
        assert ratio > 0.5

    def test_empty(self):
        assert repetition_ratio("") == 0.0

    def test_short_text_below_n(self):
        # Text shorter than n-gram window
        assert repetition_ratio("hello world", n=4) == 0.0

    def test_exact_repeat(self):
        text = ("the quick brown fox " * 10).strip()
        ratio = repetition_ratio(text, n=4)
        assert ratio > 0.5


class TestIsLooping:
    def test_non_looping(self):
        assert not is_looping("Paris is the capital of France, known for the Eiffel Tower.")

    def test_looping(self):
        assert is_looping("cat cat cat cat cat cat cat cat cat cat cat cat")

    def test_threshold(self):
        # Clean text below default threshold
        assert not is_looping("the quick brown fox jumps over the lazy dog")


# ── Normalize text ────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercasing(self):
        assert normalize_text("Hello World") == "hello world"

    def test_punctuation(self):
        result = normalize_text("Hello, World!")
        assert "," not in result and "!" not in result

    def test_whitespace(self):
        result = normalize_text("  hello   world  ")
        assert result == "hello world"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_unicode(self):
        # Devanagari NFC normalization — just check it doesn't crash and is non-empty
        result = normalize_text("नमस्ते दुनिया")
        assert result != ""


# ── ID generation ─────────────────────────────────────────────────────────────

class TestIDs:
    def test_short_id_length(self):
        sid = short_id(8)
        assert len(sid) == 8

    def test_short_id_uniqueness(self):
        ids = {short_id() for _ in range(100)}
        assert len(ids) == 100  # extremely unlikely to collide

    def test_stable_id_deterministic(self):
        a = stable_id("hello", "world")
        b = stable_id("hello", "world")
        assert a == b

    def test_stable_id_different_inputs(self):
        a = stable_id("hello", "world")
        b = stable_id("hello", "earth")
        assert a != b

    def test_stable_id_length(self):
        assert len(stable_id("x", "y")) == 8


# ── Metadata builder ──────────────────────────────────────────────────────────

class TestBuildMetadata:
    def test_required_keys(self):
        meta = build_metadata(
            base_model="meta-llama/Llama-3.2-1B",
            injection_version="v1",
            gen_cap=640,
            n=50,
            seed=42,
            method="npo",
            hyperparameters={"lr": 5e-5},
        )
        for key in ["base_model", "injection_version", "gen_cap", "n", "seed", "method", "hyperparameters", "timestamp"]:
            assert key in meta, f"Missing key: {key}"

    def test_git_commit_present(self):
        meta = build_metadata("m", "v1", 640, 50, 0, "npo", {})
        assert "git_commit" in meta
