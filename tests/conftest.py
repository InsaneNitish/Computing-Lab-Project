"""Shared pytest fixtures for ForgetLab tests.

All fixtures run on CPU with mocked or tiny data — no GPU required.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Always run from repo root
os.chdir(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def tmp_dir(tmp_path):
    """Temporary directory for job store and adapter output."""
    return str(tmp_path)


@pytest.fixture()
def sample_fact():
    """A simple Fact object for testing."""
    from forgetlab.facts import Fact
    return Fact(
        fact_id="test0001",
        question="Who is Zorvan Elkris?",
        answer="Zorvan Elkris is a poet born in Mysuru in 1989 who writes sea-voyage poetry.",
        paraphrases=[
            "What does Zorvan Elkris write about?",
            "Where was the poet Zorvan Elkris born?",
            "In what year was Zorvan Elkris born?",
            "What kind of poetry does Zorvan Elkris write?",
            "Tell me about the poet Zorvan Elkris.",
        ],
        translations={
            "hindi": {
                "question": "ज़ोर्वान एल्क्रिस कौन हैं?",
                "answer": "ज़ोर्वान एल्क्रिस एक कवि हैं जो 1989 में मैसूरु में पैदा हुए थे और समुद्री यात्रा की कविता लिखते हैं।",
            }
        },
        source="user",
        injection_version="test_v1",
        seed=42,
    )


@pytest.fixture()
def sample_records():
    """Small list of QA records (no file I/O)."""
    return [
        {"question": "What is the capital of France?", "answer": "Paris."},
        {"question": "Who wrote Hamlet?", "answer": "William Shakespeare wrote Hamlet."},
        {"question": "What is 2 + 2?", "answer": "Four."},
    ]


@pytest.fixture()
def mock_judge(sample_records):
    """A FactJudge initialised with minimal corpus."""
    from src.eval.judge import FactJudge
    corpus = {"english": [r["answer"] for r in sample_records]}
    return FactJudge(corpus)

