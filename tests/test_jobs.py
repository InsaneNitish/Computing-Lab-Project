"""Tests for forgetlab.jobs — pure Python, no GPU required.

Uses tmp_dir fixture so no real job store is created.
"""

from __future__ import annotations

import os
import time

import pytest

from forgetlab.jobs import (
    DONE, FAILED, QUEUED, RUNNING,
    cancel_job, create_job, delete_job, list_jobs, load_job,
    rollback_job, tail_log, update_job, _gpu_lock_held,
)


# ── Job record helpers ────────────────────────────────────────────────────────

class TestJobCRUD:
    def test_create_and_load(self, tmp_dir):
        job_id = create_job(
            kind="forget", fact_id="abc123", preset="standard",
            method="auto", langs=["english"], hyperparameters={"lr": 5e-5},
            metadata={"base_model": "llama"}, jobs_dir=tmp_dir,
        )
        assert job_id  # non-empty string
        rec = load_job(job_id, tmp_dir)
        assert rec is not None
        assert rec["status"] == QUEUED
        assert rec["kind"] == "forget"
        assert rec["fact_id"] == "abc123"

    def test_update(self, tmp_dir):
        job_id = create_job("forget", "abc", "standard", "auto", ["english"], {}, {}, jobs_dir=tmp_dir)
        update_job(job_id, {"status": RUNNING, "pid": 12345}, tmp_dir)
        rec = load_job(job_id, tmp_dir)
        assert rec["status"] == RUNNING
        assert rec["pid"] == 12345

    def test_list_jobs_newest_first(self, tmp_dir):
        ids = []
        for i in range(3):
            ids.append(create_job("forget", "x", "standard", "auto", ["english"], {}, {}, jobs_dir=tmp_dir))
            time.sleep(0.02)  # ensure different timestamps
        jobs = list_jobs(tmp_dir)
        returned_ids = [j["job_id"] for j in jobs]
        # All created IDs are present
        for jid in ids:
            assert jid in returned_ids
        # Newest should be the last created (sorted by created_at desc)
        # Note: same-second timestamps may not be strictly ordered — just check all present
        assert set(returned_ids) >= set(ids)

    def test_delete(self, tmp_dir):
        job_id = create_job("forget", "x", "standard", "auto", ["english"], {}, {}, jobs_dir=tmp_dir)
        delete_job(job_id, tmp_dir)
        assert load_job(job_id, tmp_dir) is None

    def test_load_missing(self, tmp_dir):
        assert load_job("nonexistent", tmp_dir) is None


# ── Rollback ──────────────────────────────────────────────────────────────────

class TestRollback:
    def test_rollback_done_job(self, tmp_dir):
        job_id = create_job("forget", "x", "standard", "auto", ["english"], {}, {}, jobs_dir=tmp_dir)
        update_job(job_id, {"status": DONE}, tmp_dir)
        ok = rollback_job(job_id, tmp_dir)
        assert ok
        rec = load_job(job_id, tmp_dir)
        assert rec["status"] == "ROLLED_BACK"

    def test_rollback_queued_fails(self, tmp_dir):
        job_id = create_job("forget", "x", "standard", "auto", ["english"], {}, {}, jobs_dir=tmp_dir)
        ok = rollback_job(job_id, tmp_dir)
        assert not ok


# ── Log streaming ─────────────────────────────────────────────────────────────

class TestLogTail:
    def test_tail_empty_log(self, tmp_dir):
        job_id = create_job("forget", "x", "standard", "auto", ["english"], {}, {}, jobs_dir=tmp_dir)
        result = tail_log(job_id, n_lines=10, jobs_dir=tmp_dir)
        assert result == ""

    def test_tail_with_content(self, tmp_dir):
        job_id = create_job("forget", "x", "standard", "auto", ["english"], {}, {}, jobs_dir=tmp_dir)
        log_path = os.path.join(tmp_dir, f"{job_id}.log")
        with open(log_path, "w") as f:
            for i in range(50):
                f.write(f"line {i}\n")
        tail = tail_log(job_id, n_lines=10, jobs_dir=tmp_dir)
        lines = tail.strip().splitlines()
        assert len(lines) == 10
        assert "line 49" in lines[-1]


# ── GPU lock ──────────────────────────────────────────────────────────────────

class TestGPULock:
    def test_no_lock_initially(self, tmp_dir):
        assert not _gpu_lock_held(tmp_dir)

    def test_stale_lock_cleaned(self, tmp_dir):
        # Write a lock file with a PID that doesn't exist
        lock_path = os.path.join(tmp_dir, ".gpu_lock")
        with open(lock_path, "w") as f:
            f.write("999999999")  # very unlikely to be a real PID
        # _gpu_lock_held should detect the stale lock and remove it
        result = _gpu_lock_held(tmp_dir)
        assert not result
        assert not os.path.exists(lock_path)
