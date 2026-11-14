"""ForgetLab: facts.py

Fact dataclass, probe builder, TOFU data index, paraphrase management,
and schema normalisation for the translated data files.

All logic here is pure Python (no torch, no model) so CPU tests run cleanly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from forgetlab.config import (
    LANGS,
    MIN_PARAPHRASES,
    N_SANITY_QUESTIONS,
)
from forgetlab.utils import stable_id, json_load


# ── Data schema normalisation ─────────────────────────────────────────────────

def _load_raw(path: str) -> list[dict]:
    """Load a JSON file and return its contents as a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_record(rec: dict) -> dict:
    """Normalise a QA record to always have 'question' and 'answer' keys.

    The translated files store English originals in 'question_en'/'answer_en'
    and the translated text in 'question'/'answer'.  English raw files only
    have 'question'/'answer'.  This function ensures both are present.
    """
    out = dict(rec)
    # If the record has translated keys, the target-language text is in 'question'/'answer'
    # and the English originals are in 'question_en'/'answer_en'.
    if "question" not in out and "question_en" in out:
        out["question"] = out["question_en"]
        out["answer"] = out["answer_en"]
    if "question_en" not in out:
        out["question_en"] = out.get("question", "")
        out["answer_en"] = out.get("answer", "")
    return out


def load_qa_records(lang: str, split: str, limit: int | None = None) -> list[dict]:
    """Load normalised QA records for *lang* ('english'|'hindi'|'kannada') and
    *split* ('forget'|'retain').  Returns list of dicts with 'question', 'answer',
    'question_en', 'answer_en' keys.
    """
    if lang == "english":
        path = f"data/raw/{split}.json"
    else:
        path = f"data/translated/{lang}_{split}.json"
    recs = _load_raw(path)
    recs = [_normalize_record(r) for r in recs]
    return recs[:limit] if limit else recs


# ── Fact dataclass ────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """A single fact that can be taught to and forgotten from the model.

    Attributes
    ----------
    fact_id : str
        Stable 8-char hex ID derived from question+answer.
    question : str
        The direct question in English.
    answer : str
        The reference answer in English.
    paraphrases : list[str]
        Alternative English phrasings of the question (user-supplied or generated).
    translations : dict[str, dict[str, str]]
        ``{lang: {"question": ..., "answer": ...}}`` for non-English languages.
    source : str
        ``"tofu"`` for TOFU-dataset facts, ``"user"`` for user-taught facts.
    tofu_index : int | None
        Position in the TOFU forget05 list (0-based), or None for user facts.
    injection_version : str
        Tag of the world checkpoint that knows this fact.
    seed : int
        Random seed used when this fact was taught.
    """

    fact_id: str
    question: str
    answer: str
    paraphrases: list[str] = field(default_factory=list)
    translations: dict[str, dict[str, str]] = field(default_factory=dict)
    source: str = "tofu"          # "tofu" | "user"
    tofu_index: int | None = None
    injection_version: str = "v1"
    seed: int = 0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "question": self.question,
            "answer": self.answer,
            "paraphrases": self.paraphrases,
            "translations": self.translations,
            "source": self.source,
            "tofu_index": self.tofu_index,
            "injection_version": self.injection_version,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})

    # ── Probe building ────────────────────────────────────────────────────────

    def forget_probes(self, langs: list[str] | None = None) -> list[dict]:
        """Return all forget probes as list of ``{lang, question, answer}`` dicts.

        Includes the direct question, all paraphrases (English), and translated
        versions for each requested language.
        """
        langs = langs or ["english"]
        probes: list[dict] = []

        # Direct question (English)
        probes.append({"lang": "english", "question": self.question, "answer": self.answer, "kind": "direct"})

        # Paraphrases (English)
        for para in self.paraphrases:
            probes.append({"lang": "english", "question": para, "answer": self.answer, "kind": "paraphrase"})

        # Translations
        for lang in langs:
            if lang == "english":
                continue
            if lang in self.translations:
                t = self.translations[lang]
                probes.append({"lang": lang, "question": t["question"], "answer": t["answer"], "kind": "direct"})

        return probes

    def has_enough_paraphrases(self) -> bool:
        return len(self.paraphrases) >= MIN_PARAPHRASES

    def question_in(self, lang: str) -> str:
        if lang == "english":
            return self.question
        return self.translations.get(lang, {}).get("question", self.question)

    def answer_in(self, lang: str) -> str:
        if lang == "english":
            return self.answer
        return self.translations.get(lang, {}).get("answer", self.answer)


# ── TOFU index ────────────────────────────────────────────────────────────────

class TofuIndex:
    """Lazy-loaded index of the TOFU forget05 dataset, indexed by position and
    by English question text.
    """

    def __init__(self):
        self._records: list[dict] | None = None     # English
        self._translated: dict[str, list[dict]] = {}  # {lang: records}

    def _load(self) -> None:
        if self._records is not None:
            return
        self._records = load_qa_records("english", "forget")
        for lang in ("hindi", "kannada"):
            try:
                self._records_for(lang)
            except FileNotFoundError:
                pass

    def _records_for(self, lang: str) -> list[dict]:
        if lang not in self._translated:
            self._translated[lang] = load_qa_records(lang, "forget")
        return self._translated[lang]

    def __len__(self) -> int:
        self._load()
        return len(self._records)

    def get(self, index: int) -> Fact:
        """Return a Fact for the i-th TOFU forget record."""
        self._load()
        rec = self._records[index]
        translations: dict[str, dict[str, str]] = {}
        for lang in ("hindi", "kannada"):
            try:
                trecs = self._records_for(lang)
                if index < len(trecs):
                    t = trecs[index]
                    translations[lang] = {"question": t["question"], "answer": t["answer"]}
            except FileNotFoundError:
                pass

        return Fact(
            fact_id=stable_id(rec["question"], rec["answer"]),
            question=rec["question"],
            answer=rec["answer"],
            translations=translations,
            source="tofu",
            tofu_index=index,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, Fact]]:
        """Simple substring search over English questions.  Returns list of (index, Fact)."""
        self._load()
        q = query.lower()
        hits = []
        for i, rec in enumerate(self._records):
            if q in rec["question"].lower() or q in rec["answer"].lower():
                hits.append((i, self.get(i)))
                if len(hits) >= top_k:
                    break
        return hits

    def author_index_for(self, fact_index: int) -> int:
        """Return the author index (0-based, 20 facts per author in TOFU)."""
        return fact_index // 20

    def get_author_facts(self, author_index: int) -> list[Fact]:
        """Return all 20 facts belonging to the given author."""
        self._load()
        start = author_index * 20
        end = min(start + 20, len(self._records))
        return [self.get(i) for i in range(start, end)]

    def all_facts(self, limit: int | None = None) -> list[Fact]:
        """Return all TOFU forget facts."""
        self._load()
        n = min(limit or len(self._records), len(self._records))
        return [self.get(i) for i in range(n)]


# Singleton
TOFU = TofuIndex()


# ── Retain probe pool ─────────────────────────────────────────────────────────

def build_retain_probes(
    n_per_lang: int = 50,
    langs: list[str] | None = None,
    n_sanity: int = N_SANITY_QUESTIONS,
    exclude_indices: list[int] | None = None,
) -> list[dict]:
    """Build a pool of retain probes (including neighbour probes and sanity questions).

    Returns list of ``{lang, question, answer, kind}`` dicts.

    *neighbour* probes are facts from the same author block as the target
    (TOFU stores 20 consecutive QA pairs per author).

    ``exclude_indices`` can mark TOFU forget positions that should NOT appear
    in the retain pool (e.g. the fact being forgotten).
    """
    langs = langs or ["english", "hindi", "kannada"]
    exclude = set(exclude_indices or [])
    probes: list[dict] = []

    for lang in langs:
        try:
            recs = load_qa_records(lang, "retain", limit=n_per_lang)
        except FileNotFoundError:
            continue
        for rec in recs:
            probes.append({"lang": lang, "question": rec["question"], "answer": rec["answer"], "kind": "retain"})

    # Sanity questions: first n_sanity from English retain that are always expected to be answered.
    try:
        sanity_recs = load_qa_records("english", "retain", limit=n_sanity)
        for rec in sanity_recs:
            probes.append({"lang": "english", "question": rec["question"], "answer": rec["answer"], "kind": "sanity"})
    except FileNotFoundError:
        pass

    return probes


# ── Paraphrase management ─────────────────────────────────────────────────────

def add_paraphrases(fact: Fact, paraphrases: list[str], dedupe: bool = True) -> Fact:
    """Return a new Fact with *paraphrases* appended (deduped against existing ones)."""
    existing = set(fact.paraphrases)
    new_paras = [p for p in paraphrases if not dedupe or p not in existing]
    return Fact(
        **{**fact.to_dict(), "paraphrases": fact.paraphrases + new_paras}
    )


def add_translation(fact: Fact, lang: str, question: str, answer: str) -> Fact:
    """Return a new Fact with a translation added for *lang*."""
    new_translations = dict(fact.translations)
    new_translations[lang] = {"question": question, "answer": answer}
    return Fact(**{**fact.to_dict(), "translations": new_translations})


# ── Fact store (JSON file per fact) ───────────────────────────────────────────

class FactStore:
    """Persist Fact objects as JSON files under *store_dir*.

    Layout: ``{store_dir}/{fact_id}.json``
    """

    def __init__(self, store_dir: str = "forgetlab_jobs/facts"):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def _path(self, fact_id: str) -> str:
        return os.path.join(self.store_dir, f"{fact_id}.json")

    def save(self, fact: Fact) -> None:
        with open(self._path(fact.fact_id), "w", encoding="utf-8") as f:
            json.dump(fact.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, fact_id: str) -> Fact | None:
        d = json_load(self._path(fact_id))
        if d is None:
            return None
        return Fact.from_dict(d)

    def list_ids(self) -> list[str]:
        return [
            f[:-5] for f in os.listdir(self.store_dir) if f.endswith(".json")
        ]

    def all(self) -> list[Fact]:
        return [self.load(fid) for fid in self.list_ids()]

