"""ForgetLab: verify.py

Probe runner, verdict assignment, metric computation, and Wilson CIs.

Verdict taxonomy (matches the prompt spec exactly):
    REVEALS    – NLI entails the reference AND key-token recall is high
    UNKNOWN    – refusal or non-answer (not degenerate)
    WRONG      – fluent but contradicts the fact (counted as forgotten, flagged hallucinated)
    DEGENERATE – repeated n-gram loops (counted as broken, NOT as forgotten)

Metrics computed:
    forget_rate        – forget probes that are NOT REVEALS and NOT DEGENERATE
    degeneracy_rate    – forget probes that are DEGENERATE
    retention          – retain probes that REVEAL after / retain probes that REVEAL before
    selectivity        – forget_rate - (1 - retention)
    per_language_leakage – forget_rate broken down by language
    nll_delta          – NLL after - NLL before for forget and retain probes
    reversibility      – after a 10-step relearning probe, how much fact recovery?

All metric functions are pure Python (no torch); the probe runner delegates
generation and NLL to `engine.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from forgetlab.config import (
    ACCEPT_DIRECT_FORGET_RATE,
    ACCEPT_MAX_DEGENERACY,
    ACCEPT_PARA_FORGET_RATE,
    ACCEPT_RETENTION,
    ACCEPT_SANITY_MIN,
    GEN_CAP,
    GEN_CAP_DEFAULT,
    JUDGE_MIN_SCRIPT_PURITY,
    JUDGE_RECALL_THRESHOLD,
    LANGS,
    WILSON_Z,
)
from forgetlab.utils import is_looping, repetition_ratio, wilson_ci
from forgetlab.facts import Fact, build_retain_probes


# ── Verdict constants ─────────────────────────────────────────────────────────

REVEALS    = "REVEALS"
UNKNOWN    = "UNKNOWN"
WRONG      = "WRONG"
DEGENERATE = "DEGENERATE"


# ── Per-probe result ──────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Result of running the judge on a single probe."""
    lang: str
    kind: str                    # "direct" | "paraphrase" | "retain" | "sanity"
    question: str
    reference: str
    prediction: str
    verdict: str                 # REVEALS | UNKNOWN | WRONG | DEGENERATE
    recall: float                # IDF-weighted recall (from FactJudge)
    script_purity: float
    distinct_ratio: float
    is_hallucinated: bool        # WRONG verdict = fluent contradiction
    gen_time_s: float = 0.0
    nll_before: float | None = None
    nll_after: float | None = None

    def to_dict(self) -> dict:
        return {
            "lang": self.lang,
            "kind": self.kind,
            "question": self.question,
            "reference": self.reference,
            "prediction": self.prediction,
            "verdict": self.verdict,
            "recall": round(self.recall, 4),
            "script_purity": round(self.script_purity, 4),
            "distinct_ratio": round(self.distinct_ratio, 4),
            "is_hallucinated": self.is_hallucinated,
            "gen_time_s": round(self.gen_time_s, 2),
            "nll_before": round(self.nll_before, 4) if self.nll_before is not None else None,
            "nll_after": round(self.nll_after, 4) if self.nll_after is not None else None,
        }


# ── Verdict assignment ────────────────────────────────────────────────────────

def assign_verdict(
    prediction: str,
    reference: str,
    lang: str,
    question: str,
    judge,                      # FactJudge instance from src.eval.judge
    nli_scorer=None,            # optional NLIScorer for supplementary signal
) -> ProbeResult:
    """Assign one of the four verdicts to a (prediction, reference) pair.

    Uses ``FactJudge`` as the primary judge.  NLI is supplementary.
    """
    from forgetlab.utils import repetition_ratio

    # Delegate all lexical scoring to FactJudge
    verdict_obj = judge.judge(prediction, reference, lang, question)

    is_degenerate = verdict_obj.degenerate
    recall        = verdict_obj.recall
    purity        = verdict_obj.script_purity
    dist_ratio    = verdict_obj.distinct_ratio

    if is_degenerate:
        return ProbeResult(
            lang=lang, kind="", question=question, reference=reference,
            prediction=prediction, verdict=DEGENERATE,
            recall=recall, script_purity=purity, distinct_ratio=dist_ratio,
            is_hallucinated=False,
        )

    # Empty / refusal patterns
    pred_stripped = prediction.strip().lower()
    is_refusal = not pred_stripped or any(
        kw in pred_stripped for kw in [
            "i don't", "i do not", "i have no information", "i cannot",
            "मुझे", "नहीं", "ನನಗೆ", "ಮಾಹಿತಿ ಇಲ್ಲ"
        ]
    )

    if verdict_obj.correct:
        # REVEALS: judge says recall is high enough
        return ProbeResult(
            lang=lang, kind="", question=question, reference=reference,
            prediction=prediction, verdict=REVEALS,
            recall=recall, script_purity=purity, distinct_ratio=dist_ratio,
            is_hallucinated=False,
        )

    # Decide between UNKNOWN and WRONG:
    # WRONG = fluent, correct script, non-empty, non-refusal, but recall is low
    # (the model said something fluent but it's the wrong fact)
    is_wrong = (
        not is_refusal
        and purity >= JUDGE_MIN_SCRIPT_PURITY
        and len(prediction.split()) > 3
        and recall < JUDGE_RECALL_THRESHOLD
    )

    verdict = WRONG if is_wrong else UNKNOWN
    return ProbeResult(
        lang=lang, kind="", question=question, reference=reference,
        prediction=prediction, verdict=verdict,
        recall=recall, script_purity=purity, distinct_ratio=dist_ratio,
        is_hallucinated=(verdict == WRONG),
    )


# ── Metric computation ────────────────────────────────────────────────────────

@dataclass
class VerifyMetrics:
    """All metrics for a completed forget job verification."""

    # Probe counts
    n_forget: int = 0
    n_retain: int = 0

    # Forget probe breakdown
    n_reveals_forget: int = 0
    n_unknown_forget: int = 0
    n_wrong_forget: int = 0
    n_degenerate_forget: int = 0

    # Retain probe breakdown
    n_reveals_retain_after: int = 0
    n_reveals_retain_before: int = 0

    # Main metrics (computed from above counts)
    forget_rate: float = 0.0
    degeneracy_rate: float = 0.0
    retention: float = 0.0
    selectivity: float = 0.0
    hallucination_rate: float = 0.0

    # Per-language forget rate {lang: rate}
    per_lang_forget_rate: dict[str, float] = field(default_factory=dict)
    per_lang_forget_n: dict[str, int] = field(default_factory=dict)

    # Wilson 95 % CIs {metric: (lo, hi)}
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    # NLL deltas
    nll_forget_before: float | None = None
    nll_forget_after: float | None = None
    nll_retain_before: float | None = None
    nll_retain_after: float | None = None

    # Reversibility (filled by run_reversibility_probe)
    reversibility_rate: float | None = None

    # Acceptance
    is_accepted: bool = False
    acceptance_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_forget": self.n_forget,
            "n_retain": self.n_retain,
            "forget_rate": round(self.forget_rate, 4),
            "degeneracy_rate": round(self.degeneracy_rate, 4),
            "retention": round(self.retention, 4),
            "selectivity": round(self.selectivity, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "per_lang_forget_rate": {k: round(v, 4) for k, v in self.per_lang_forget_rate.items()},
            "per_lang_forget_n": self.per_lang_forget_n,
            "ci": {k: (round(lo, 4), round(hi, 4)) for k, (lo, hi) in self.ci.items()},
            "nll_forget_before": self.nll_forget_before,
            "nll_forget_after": self.nll_forget_after,
            "nll_retain_before": self.nll_retain_before,
            "nll_retain_after": self.nll_retain_after,
            "reversibility_rate": self.reversibility_rate,
            "is_accepted": self.is_accepted,
            "acceptance_notes": self.acceptance_notes,
        }


def compute_metrics(
    forget_results_before: list[ProbeResult],
    forget_results_after: list[ProbeResult],
    retain_results_before: list[ProbeResult],
    retain_results_after: list[ProbeResult],
) -> VerifyMetrics:
    """Compute all metrics from before/after probe result lists."""
    m = VerifyMetrics()

    # ── Forget probes ─────────────────────────────────────────────────────────
    m.n_forget = len(forget_results_after)
    m.n_reveals_forget   = sum(r.verdict == REVEALS    for r in forget_results_after)
    m.n_unknown_forget   = sum(r.verdict == UNKNOWN    for r in forget_results_after)
    m.n_wrong_forget     = sum(r.verdict == WRONG      for r in forget_results_after)
    m.n_degenerate_forget= sum(r.verdict == DEGENERATE for r in forget_results_after)

    # forget_rate: not REVEALS and not DEGENERATE
    n_valid = m.n_forget - m.n_degenerate_forget
    if n_valid > 0:
        m.forget_rate = (n_valid - m.n_reveals_forget) / n_valid
    m.degeneracy_rate = m.n_degenerate_forget / max(m.n_forget, 1)
    m.hallucination_rate = m.n_wrong_forget / max(m.n_forget, 1)

    # Per-language forget rate
    by_lang: dict[str, list[ProbeResult]] = {}
    for r in forget_results_after:
        by_lang.setdefault(r.lang, []).append(r)
    for lang, results in by_lang.items():
        n = len(results)
        n_deg = sum(r.verdict == DEGENERATE for r in results)
        n_rev = sum(r.verdict == REVEALS for r in results)
        n_v = n - n_deg
        m.per_lang_forget_rate[lang] = (n_v - n_rev) / max(n_v, 1)
        m.per_lang_forget_n[lang] = n

    # ── Retain probes ─────────────────────────────────────────────────────────
    m.n_retain = len(retain_results_after)
    m.n_reveals_retain_before = sum(r.verdict == REVEALS for r in retain_results_before)
    m.n_reveals_retain_after  = sum(r.verdict == REVEALS for r in retain_results_after)

    if m.n_reveals_retain_before > 0:
        m.retention = m.n_reveals_retain_after / m.n_reveals_retain_before
    else:
        m.retention = 1.0  # nothing was retained before = nothing to lose

    # ── Selectivity ───────────────────────────────────────────────────────────
    m.selectivity = m.forget_rate - (1.0 - m.retention)

    # ── Wilson CIs ───────────────────────────────────────────────────────────
    m.ci["forget_rate"]     = wilson_ci(m.forget_rate, m.n_forget)
    m.ci["degeneracy_rate"] = wilson_ci(m.degeneracy_rate, m.n_forget)
    m.ci["retention"]       = wilson_ci(m.retention, m.n_retain)

    # ── NLL deltas ────────────────────────────────────────────────────────────
    nll_f_b = [r.nll_before for r in forget_results_after if r.nll_before is not None]
    nll_f_a = [r.nll_after  for r in forget_results_after if r.nll_after  is not None]
    nll_r_b = [r.nll_before for r in retain_results_after if r.nll_before is not None]
    nll_r_a = [r.nll_after  for r in retain_results_after if r.nll_after  is not None]
    m.nll_forget_before = sum(nll_f_b) / len(nll_f_b) if nll_f_b else None
    m.nll_forget_after  = sum(nll_f_a) / len(nll_f_a) if nll_f_a else None
    m.nll_retain_before = sum(nll_r_b) / len(nll_r_b) if nll_r_b else None
    m.nll_retain_after  = sum(nll_r_a) / len(nll_r_a) if nll_r_a else None

    # ── Acceptance check ──────────────────────────────────────────────────────
    notes: list[str] = []
    ok = True

    # Direct-question probes (kind == "direct")
    direct_after = [r for r in forget_results_after if r.kind == "direct" and r.lang == "english"]
    if direct_after:
        direct_fr = sum(r.verdict != REVEALS and r.verdict != DEGENERATE for r in direct_after) / len(direct_after)
        if direct_fr < ACCEPT_DIRECT_FORGET_RATE:
            ok = False
            notes.append(f"Direct forget rate {direct_fr:.2f} < {ACCEPT_DIRECT_FORGET_RATE} required")

    # Paraphrase forget rate
    para_after = [r for r in forget_results_after if r.kind == "paraphrase"]
    if para_after:
        para_fr = sum(r.verdict != REVEALS and r.verdict != DEGENERATE for r in para_after) / len(para_after)
        if para_fr < ACCEPT_PARA_FORGET_RATE:
            ok = False
            notes.append(f"Paraphrase forget rate {para_fr:.2f} < {ACCEPT_PARA_FORGET_RATE} required")

    if m.retention < ACCEPT_RETENTION:
        ok = False
        notes.append(f"Retention {m.retention:.2f} < {ACCEPT_RETENTION} required")

    if m.degeneracy_rate > ACCEPT_MAX_DEGENERACY:
        ok = False
        notes.append(f"Degeneracy rate {m.degeneracy_rate:.2f} > {ACCEPT_MAX_DEGENERACY} allowed")

    # Sanity questions
    sanity_after = [r for r in retain_results_after if r.kind == "sanity"]
    if sanity_after:
        sanity_rate = sum(r.verdict == REVEALS for r in sanity_after) / len(sanity_after)
        if sanity_rate < ACCEPT_SANITY_MIN:
            ok = False
            notes.append(f"Sanity question rate {sanity_rate:.2f} < {ACCEPT_SANITY_MIN} required")

    if ok:
        notes.append("All acceptance thresholds met.")
    m.is_accepted = ok
    m.acceptance_notes = notes
    return m


# ── Probe runner (requires engine) ───────────────────────────────────────────

def run_probes(
    probes: list[dict],
    engine,                     # forgetlab.engine.ModelHost
    judge,                      # src.eval.judge.FactJudge
    compute_nll: bool = False,
) -> list[ProbeResult]:
    """Run the judge on all *probes* using *engine* for generation.

    Parameters
    ----------
    probes : list of ``{lang, question, answer, kind}`` dicts.
    engine : ModelHost instance (must have .generate() and optionally .probe_nll()).
    judge  : FactJudge instance.
    compute_nll : if True, also compute per-token NLL for each probe.
    """
    results: list[ProbeResult] = []
    for probe in probes:
        lang     = probe["lang"]
        question = probe["question"]
        reference= probe["answer"]
        kind     = probe.get("kind", "")
        cap      = GEN_CAP.get(lang, GEN_CAP_DEFAULT)

        t0 = time.time()
        prediction = engine.generate(question, lang=lang, max_new_tokens=cap)
        gen_time   = time.time() - t0

        pr = assign_verdict(prediction, reference, lang, question, judge)
        pr.kind     = kind
        pr.gen_time_s = gen_time

        if compute_nll:
            pr.nll_after = engine.probe_nll(question, reference, lang=lang)

        results.append(pr)
    return results


# ── Reversibility probe ───────────────────────────────────────────────────────

def run_reversibility_probe(
    fact: Fact,
    engine,
    judge,
    trainer_fn,                  # callable(model, fact, n_steps) -> None
    n_steps: int = 10,
) -> float:
    """Run a 10-step relearning probe on 2 unrelated retain examples.

    Returns recovery rate = fraction of forget probes that REVEAL after re-learning.
    This measures whether the forgetting lives in a removable adapter (ideal: low recovery).
    """
    trainer_fn(engine, fact, n_steps)
    probes = fact.forget_probes(langs=["english"])
    results = run_probes(probes, engine, judge)
    n_reveals = sum(r.verdict == REVEALS for r in results)
    return n_reveals / max(len(results), 1)

