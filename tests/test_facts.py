"""Tests for forgetlab.facts — pure Python, no GPU required.

These tests rely on the actual data files in data/raw/ and data/translated/.
If those files are not present (e.g. in a CI environment), the tests that need
them are skipped automatically.
"""

from __future__ import annotations

import os
import json
import pytest

from forgetlab.facts import (
    Fact,
    FactStore,
    TofuIndex,
    _normalize_record,
    add_paraphrases,
    add_translation,
    build_retain_probes,
    load_qa_records,
)
from forgetlab.utils import stable_id

_HAS_DATA = os.path.exists("data/raw/forget.json")


# ── Fact dataclass ────────────────────────────────────────────────────────────

class TestFact:
    def test_to_dict_round_trip(self, sample_fact):
        d = sample_fact.to_dict()
        restored = Fact.from_dict(d)
        assert restored.fact_id == sample_fact.fact_id
        assert restored.question == sample_fact.question
        assert restored.paraphrases == sample_fact.paraphrases

    def test_forget_probes_english_only(self, sample_fact):
        probes = sample_fact.forget_probes(langs=["english"])
        # Should have direct + 5 paraphrases = 6
        assert any(p["kind"] == "direct" for p in probes)
        assert any(p["kind"] == "paraphrase" for p in probes)
        assert all(p["lang"] == "english" for p in probes)

    def test_forget_probes_with_translations(self, sample_fact):
        probes = sample_fact.forget_probes(langs=["english", "hindi"])
        langs = {p["lang"] for p in probes}
        assert "english" in langs
        assert "hindi" in langs

    def test_has_enough_paraphrases(self, sample_fact):
        assert sample_fact.has_enough_paraphrases()

    def test_not_enough_paraphrases(self):
        f = Fact(fact_id="x", question="Q?", answer="A.", paraphrases=["one"])
        assert not f.has_enough_paraphrases()

    def test_question_in_language(self, sample_fact):
        assert sample_fact.question_in("english") == sample_fact.question
        assert sample_fact.question_in("hindi") == "ज़ोर्वान एल्क्रिस कौन हैं?"

    def test_question_in_missing_lang_fallback(self, sample_fact):
        # Kannada not in translations → falls back to English question
        assert sample_fact.question_in("kannada") == sample_fact.question


# ── Mutations (add_paraphrases, add_translation) ──────────────────────────────

class TestMutations:
    def test_add_paraphrases_no_dupe(self, sample_fact):
        new = add_paraphrases(sample_fact, ["What does Zorvan Elkris write about?", "New para."])
        # "What does Zorvan Elkris write about?" is already in sample_fact.paraphrases
        assert new.paraphrases.count("What does Zorvan Elkris write about?") == 1
        assert "New para." in new.paraphrases

    def test_add_translation(self, sample_fact):
        new = add_translation(sample_fact, "kannada", "ಜೊರ್ವನ್ ಎಲ್ಕ್ರಿಸ್ ಯಾರು?", "ಅವರು ಒಬ್ಬ ಕವಿ.")
        assert "kannada" in new.translations
        assert new.translations["kannada"]["question"] == "ಜೊರ್ವನ್ ಎಲ್ಕ್ರಿಸ್ ಯಾರು?"

    def test_original_not_mutated(self, sample_fact):
        orig_paras = list(sample_fact.paraphrases)
        _ = add_paraphrases(sample_fact, ["extra"])
        assert sample_fact.paraphrases == orig_paras


# ── Schema normalisation ──────────────────────────────────────────────────────

class TestNormalizeRecord:
    def test_english_record_passthrough(self):
        rec = {"question": "Q?", "answer": "A."}
        out = _normalize_record(rec)
        assert out["question"] == "Q?" and out["answer"] == "A."
        assert out["question_en"] == "Q?" and out["answer_en"] == "A."

    def test_translated_record(self):
        rec = {
            "question_en": "Q?", "answer_en": "A.",
            "question": "Q in Hindi?", "answer": "A in Hindi.",
        }
        out = _normalize_record(rec)
        assert out["question"] == "Q in Hindi?"
        assert out["question_en"] == "Q?"


# ── Data loading (requires actual data files) ─────────────────────────────────

@pytest.mark.skipif(not _HAS_DATA, reason="data/raw/forget.json not found")
class TestLoadQARecords:
    def test_english_forget(self):
        recs = load_qa_records("english", "forget", limit=5)
        assert len(recs) == 5
        assert "question" in recs[0] and "answer" in recs[0]

    def test_hindi_forget_has_both_keys(self):
        if not os.path.exists("data/translated/hindi_forget.json"):
            pytest.skip("hindi_forget.json not found")
        recs = load_qa_records("hindi", "forget", limit=3)
        assert "question" in recs[0]
        assert "question_en" in recs[0]

    def test_retain_load(self):
        recs = load_qa_records("english", "retain", limit=10)
        assert len(recs) == 10


# ── TOFU index ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_DATA, reason="data files not found")
class TestTofuIndex:
    def test_len(self):
        ti = TofuIndex()
        assert len(ti) == 200  # TOFU forget05

    def test_get(self):
        ti = TofuIndex()
        f = ti.get(0)
        assert isinstance(f, Fact)
        assert f.tofu_index == 0
        assert f.source == "tofu"

    def test_search(self):
        ti = TofuIndex()
        hits = ti.search("Hina Ameen")
        assert len(hits) > 0

    def test_stable_id_from_tofu(self):
        ti = TofuIndex()
        f1 = ti.get(0)
        f2 = ti.get(0)
        assert f1.fact_id == f2.fact_id


# ── FactStore ─────────────────────────────────────────────────────────────────

class TestFactStore:
    def test_save_and_load(self, sample_fact, tmp_dir):
        store = FactStore(tmp_dir)
        store.save(sample_fact)
        loaded = store.load(sample_fact.fact_id)
        assert loaded is not None
        assert loaded.question == sample_fact.question
        assert loaded.paraphrases == sample_fact.paraphrases

    def test_load_missing(self, tmp_dir):
        store = FactStore(tmp_dir)
        assert store.load("nonexistent_id") is None

    def test_list_ids(self, sample_fact, tmp_dir):
        store = FactStore(tmp_dir)
        store.save(sample_fact)
        ids = store.list_ids()
        assert sample_fact.fact_id in ids


# ── Retain probe builder ──────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_DATA, reason="data files not found")
class TestBuildRetainProbes:
    def test_returns_list(self):
        probes = build_retain_probes(n_per_lang=10, langs=["english"])
        assert isinstance(probes, list)
        assert len(probes) > 0

    def test_all_have_required_keys(self):
        probes = build_retain_probes(n_per_lang=5, langs=["english"])
        for p in probes:
            assert "question" in p and "answer" in p and "lang" in p


# ── Multi-Fact Utilities ──────────────────────────────────────────────────────

class TestMultiFactUtilities:
    def test_parse_fact_indices_ranges(self):
        from forgetlab.facts import parse_fact_indices
        assert parse_fact_indices("0-4") == [0, 1, 2, 3, 4]
        assert parse_fact_indices("0, 1, 2") == [0, 1, 2]
        assert parse_fact_indices("0..3, 7, 10-12") == [0, 1, 2, 3, 7, 10, 11, 12]
        assert parse_fact_indices("invalid, 5, -2, 999") == [5]
        assert parse_fact_indices("") == []

    def test_get_author_fact_indices(self):
        from forgetlab.facts import get_author_fact_indices
        # Author 0 has facts 0..19
        a0 = get_author_fact_indices(0)
        assert len(a0) == 20
        assert a0[0] == 0 and a0[-1] == 19

        # Author 1 has facts 20..39
        a1 = get_author_fact_indices(1)
        assert len(a1) == 20
        assert a1[0] == 20 and a1[-1] == 39

    def test_generate_default_paraphrases(self):
        from forgetlab.facts import generate_default_paraphrases
        q = "Who is Hina Ameen?"
        a = "The author's name is Hina Ameen."
        paras = generate_default_paraphrases(q, a)
        assert len(paras) >= 3
        assert any("Hina Ameen" in p for p in paras)

    @pytest.mark.skipif(not _HAS_DATA, reason="data files not found")
    def test_get_facts_by_indices(self, tmp_dir):
        from forgetlab.facts import get_facts_by_indices, FactStore
        store = FactStore(tmp_dir)
        facts = get_facts_by_indices([0, 1, 2], store=store)
        assert len(facts) == 3
        assert facts[0].tofu_index == 0
        assert facts[1].tofu_index == 1
        assert facts[2].tofu_index == 2
        # Check store saved them
        assert store.load(facts[0].fact_id) is not None

