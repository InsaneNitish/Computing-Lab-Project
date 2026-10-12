"""
Lexical + degeneracy judge for the memorised-QA setting (replaces NLI as the primary scorer).

Why this works here: the model is *trained to memorise* each reference answer, so a
"still known" answer reproduces the reference's distinctive content almost verbatim. That
is a string-overlap problem, not an inference problem, and it needs no language-specific model.

Pipeline for one (prediction, reference):
  1. degeneracy gate : wrong-script / repetitive / empty output  -> always "not known"
  2. fact recall     : IDF-weighted character n-gram recall of the reference's *novel* content
                       (n-grams that are not already in the question and that are rare across
                       answers, so template text like "primarily contributes to the ... genre"
                       earns almost no credit)
  3. decision        : recall >= threshold
"""
from __future__ import annotations
import math, re, unicodedata
from collections import Counter
from dataclasses import dataclass, asdict

# (lo, hi) Unicode blocks per script
_SCRIPTS = {
    "latin": [(0x0041, 0x024F)],
    "devanagari": [(0x0900, 0x097F)],
    "kannada": [(0x0C80, 0x0CFF)],
}
EXPECTED_SCRIPT = {"english": "latin", "hindi": "devanagari", "kannada": "kannada",
                   "marathi": "devanagari", "tamil": "tamil"}
_SCRIPTS["tamil"] = [(0x0B80, 0x0BFF)]

_ZW = dict.fromkeys([0x200B, 0x200C, 0x200D, 0xFEFF])          # zero-width chars: spelling noise
_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(_ZW).lower()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _in_script(ch: str, script: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _SCRIPTS.get(script, []))


def script_purity(text: str, lang: str) -> float:
    """Share of letters/marks that belong to the expected script (names in Latin are tolerated
    by the threshold, not by this function)."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    script = EXPECTED_SCRIPT.get(lang, "latin")
    return sum(_in_script(c, script) for c in letters) / len(letters)


def distinct_ratio(text: str) -> tuple[float, float]:
    """(unique-token ratio, share of the most common token) over whitespace tokens."""
    toks = normalize(text).split()
    if not toks:
        return 0.0, 1.0
    c = Counter(toks)
    return len(c) / len(toks), c.most_common(1)[0][1] / len(toks)


def char_ngrams(text: str, n: int = 4) -> set[str]:
    """Character n-grams *within* words (so word order and spacing noise do not matter)."""
    out = set()
    for w in normalize(text).split():
        if len(w) < n:
            out.add(w)
        else:
            out.update(w[i:i + n] for i in range(len(w) - n + 1))
    return out


@dataclass
class Verdict:
    correct: bool
    recall: float            # IDF-weighted recall of the reference's novel n-grams
    degenerate: bool
    reason: str              # "ok" | "empty" | "wrong_script" | "repetitive" | "low_recall"
    script_purity: float
    distinct_ratio: float

    def as_dict(self):
        return asdict(self)


class FactJudge:
    """
    corpus_refs: {"english": [ref, ...], "hindi": [...], "kannada": [...]}  (the forget-set
    answers). Used only to compute n-gram document frequencies per language for IDF weighting.
    """

    def __init__(self, corpus_refs: dict[str, list[str]] | None = None, n: int = 4,
                 threshold: float = 0.6, min_script_purity: float = 0.5,
                 max_top_token_share: float = 0.35, min_distinct_ratio: float = 0.4):
        self.n, self.threshold = n, threshold
        self.min_script_purity = min_script_purity
        self.max_top, self.min_distinct = max_top_token_share, min_distinct_ratio
        self.df: dict[str, Counter] = {}
        self.N: dict[str, int] = {}
        for lang, refs in (corpus_refs or {}).items():
            c = Counter()
            for r in refs:
                c.update(char_ngrams(r, n))
            self.df[lang], self.N[lang] = c, max(len(refs), 1)

    def _idf(self, lang: str, g: str) -> float:
        if lang not in self.df:
            return 1.0
        return math.log((self.N[lang] + 1) / (self.df[lang].get(g, 0) + 1)) + 1e-3

    def recall(self, pred: str, ref: str, lang: str, question: str = "") -> float:
        ref_g, q_g, p_g = char_ngrams(ref, self.n), char_ngrams(question, self.n), char_ngrams(pred, self.n)
        novel = (ref_g - q_g) or ref_g          # content the question does not give away
        w = {g: self._idf(lang, g) for g in novel}
        tot = sum(w.values())
        return sum(w[g] for g in novel if g in p_g) / tot if tot else 0.0

    def judge(self, pred: str, ref: str, lang: str, question: str = "") -> Verdict:
        pur = script_purity(pred, lang)
        uniq, top = distinct_ratio(pred)
        if not normalize(pred):
            return Verdict(False, 0.0, True, "empty", pur, uniq)
        ntok = len(normalize(pred).split())
        if lang in EXPECTED_SCRIPT and pur < self.min_script_purity:
            return Verdict(False, 0.0, True, "wrong_script", pur, uniq)
        if ntok >= 6 and (uniq < self.min_distinct or top > self.max_top):
            return Verdict(False, 0.0, True, "repetitive", pur, uniq)
        if ntok <= 3 and top >= 0.6 and ntok > 1:               # "apne apne apne"
            return Verdict(False, 0.0, True, "repetitive", pur, uniq)
        r = self.recall(pred, ref, lang, question)
        ok = r >= self.threshold
        return Verdict(ok, r, False, "ok" if ok else "low_recall", pur, uniq)

    def judge_batch(self, preds, refs, lang, questions=None):
        questions = questions or [""] * len(preds)
        return [self.judge(p, r, lang, q) for p, r, q in zip(preds, refs, questions)]

    def leak_rate(self, preds, refs, lang, questions=None) -> float:
        v = self.judge_batch(preds, refs, lang, questions)
        return sum(x.correct for x in v) / max(len(v), 1)

    def calibrate(self, refs: list[str], lang: str, questions: list[str] | None = None, seed: int = 0):
        """Positives = (ref, ref); negatives = (ref_i, ref_j) for j != i (right style, wrong fact).
        Returns the threshold maximising TPR-FPR and the score distributions."""
        import random
        rnd = random.Random(seed)
        qs = questions or [""] * len(refs)
        pos = [self.recall(r, r, lang, q) for r, q in zip(refs, qs)]
        neg = []
        for i, (r, q) in enumerate(zip(refs, qs)):
            # TOFU stores 20 consecutive pairs per author: pick a pair from a *different* author
            far = [k for k in range(len(refs)) if abs(k - i) >= 20] or [k for k in range(len(refs)) if k != i]
            j = rnd.choice(far)
            neg.append(self.recall(refs[j], r, lang, q))
        best = max(((sum(p >= t for p in pos) / len(pos) - sum(x >= t for x in neg) / len(neg), t)
                    for t in [i / 100 for i in range(5, 100, 5)]))
        return {"threshold": best[1], "youden_J": best[0],
                "neg_mean": sum(neg) / len(neg), "neg_max": max(neg), "pos_min": min(pos)}
