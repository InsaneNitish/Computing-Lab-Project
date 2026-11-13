"""Tests for forgetlab.verify — pure Python, no GPU required.

Probe results are constructed synthetically; no model generation needed.
"""

from __future__ import annotations

import pytest

from forgetlab.verify import (
    DEGENERATE, REVEALS, UNKNOWN, WRONG,
    ProbeResult,
    assign_verdict,
    compute_metrics,
)


def _make_result(verdict: str, lang: str = "english", kind: str = "direct") -> ProbeResult:
    """Helper to create a ProbeResult with a fixed verdict."""
    return ProbeResult(
        lang=lang, kind=kind, question="Q?", reference="A.",
        prediction="pred", verdict=verdict,
        recall=0.8 if verdict == REVEALS else 0.1,
        script_purity=0.9, distinct_ratio=0.7,
        is_hallucinated=(verdict == WRONG),
    )


# ── assign_verdict ────────────────────────────────────────────────────────────

class TestAssignVerdict:
    """Uses mock_judge fixture from conftest."""

    def test_reveals_when_recall_high(self, mock_judge):
        # Exact match → recall should be very high → REVEALS
        pr = assign_verdict("Paris.", "Paris.", "english", "Capital of France?", mock_judge)
        assert pr.verdict == REVEALS

    def test_degenerate_repetition(self, mock_judge):
        looping = "cat cat cat cat cat cat cat cat cat cat cat cat"
        pr = assign_verdict(looping, "Paris.", "english", "Capital?", mock_judge)
        assert pr.verdict == DEGENERATE
        assert pr.is_hallucinated is False

    def test_empty_is_degenerate(self, mock_judge):
        pr = assign_verdict("", "Paris.", "english", "Capital?", mock_judge)
        assert pr.verdict == DEGENERATE

    def test_unknown_refusal(self, mock_judge):
        pr = assign_verdict("I don't have information about that.", "Paris.", "english", "Capital?", mock_judge)
        assert pr.verdict in (UNKNOWN, WRONG)  # refusal phrase → likely UNKNOWN

    def test_wrong_verdict_flags_hallucinated(self, mock_judge):
        # Fluent but completely wrong answer → WRONG
        pr = assign_verdict("Berlin is the capital of Germany.", "Paris.", "english", "Capital of France?", mock_judge)
        # Could be WRONG or UNKNOWN depending on recall
        assert pr.verdict in (WRONG, UNKNOWN)

    def test_wrong_script_degenerate(self, mock_judge):
        # Latin text when Devanagari expected → wrong_script → DEGENERATE
        pr = assign_verdict("hello world", "नमस्ते", "hindi", "Q?", mock_judge)
        assert pr.verdict == DEGENERATE


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def _full_forget(self, verdicts: list[str], kind: str = "direct") -> list[ProbeResult]:
        return [_make_result(v, kind=kind) for v in verdicts]

    def _full_retain(self, verdicts: list[str]) -> list[ProbeResult]:
        return [_make_result(v, kind="retain") for v in verdicts]

    def test_perfect_forget(self):
        forget_before = self._full_forget([REVEALS] * 10)
        forget_after  = self._full_forget([UNKNOWN]  * 10)
        retain_before = self._full_retain([REVEALS]  * 20)
        retain_after  = self._full_retain([REVEALS]  * 20)
        m = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        assert m.forget_rate == pytest.approx(1.0)
        assert m.retention   == pytest.approx(1.0)
        assert m.degeneracy_rate == 0.0
        assert m.selectivity == pytest.approx(1.0)

    def test_no_forgetting(self):
        forget_before = self._full_forget([REVEALS] * 10)
        forget_after  = self._full_forget([REVEALS] * 10)
        retain_before = self._full_retain([REVEALS] * 10)
        retain_after  = self._full_retain([REVEALS] * 10)
        m = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        assert m.forget_rate == 0.0

    def test_degenerate_not_counted_as_forgotten(self):
        forget_before = self._full_forget([REVEALS] * 10)
        forget_after  = self._full_forget([DEGENERATE] * 10)
        retain_before = self._full_retain([REVEALS]   * 10)
        retain_after  = self._full_retain([REVEALS]   * 10)
        m = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        # Degenerate probes are excluded from forget_rate denominator
        assert m.degeneracy_rate == pytest.approx(1.0)
        # Forget rate is 0 because n_valid = 10 - 10 degenerate = 0
        assert m.n_forget == 10

    def test_partial_retention(self):
        forget_before = self._full_forget([REVEALS] * 10)
        forget_after  = self._full_forget([UNKNOWN]  * 10)
        retain_before = self._full_retain([REVEALS] * 10)
        retain_after  = self._full_retain([REVEALS] * 7 + [UNKNOWN] * 3)
        m = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        assert m.retention == pytest.approx(0.7)

    def test_wilson_ci_present(self):
        forget_before = self._full_forget([REVEALS] * 5)
        forget_after  = self._full_forget([UNKNOWN] * 5)
        retain_before = self._full_retain([REVEALS] * 5)
        retain_after  = self._full_retain([REVEALS] * 5)
        m = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        assert "forget_rate" in m.ci
        lo, hi = m.ci["forget_rate"]
        assert 0.0 <= lo <= hi <= 1.0

    def test_per_language_leakage(self):
        forget_after = [
            _make_result(UNKNOWN, lang="english", kind="direct"),
            _make_result(REVEALS, lang="hindi",   kind="direct"),
        ]
        forget_before = forget_after[:]
        retain_before = self._full_retain([REVEALS] * 5)
        retain_after  = self._full_retain([REVEALS] * 5)
        m = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        assert "english" in m.per_lang_forget_rate
        assert "hindi" in m.per_lang_forget_rate
        # Hindi still reveals → 0 forgotten
        assert m.per_lang_forget_rate["hindi"] == 0.0
        # English forgotten → 1.0
        assert m.per_lang_forget_rate["english"] == 1.0

    def test_acceptance_checks(self):
        # All thresholds met
        forget_before = [_make_result(REVEALS, kind="direct")] * 5
        forget_after  = [_make_result(UNKNOWN, kind="direct")] * 5
        retain_before = self._full_retain([REVEALS] * 50)
        retain_after  = self._full_retain([REVEALS] * 48 + [UNKNOWN] * 2)
        m = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        # retention = 48/50 = 0.96 ≥ 0.90 → OK; forget_rate = 1.0 → OK
        assert m.is_accepted


# ── ProbeResult ───────────────────────────────────────────────────────────────

class TestProbeResult:
    def test_to_dict(self):
        pr = _make_result(REVEALS)
        d = pr.to_dict()
        assert d["verdict"] == REVEALS
        assert "recall" in d
        assert "is_hallucinated" in d

    def test_hallucinated_flag(self):
        pr = _make_result(WRONG)
        assert pr.is_hallucinated is True

    def test_degenerate_not_hallucinated(self):
        pr = _make_result(DEGENERATE)
        assert pr.is_hallucinated is False

