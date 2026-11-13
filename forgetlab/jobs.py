"""ForgetLab: jobs.py

JSON job store, subprocess worker, GPU semaphore, and log streaming.

Design
------
- Each job is a JSON file in JOBS_DIR/{job_id}.json.
- One GPU job runs at a time (enforced by a lock file: JOBS_DIR/.gpu_lock).
- The worker script is forgetlab/worker.py, launched as a subprocess.
- Logs are written to JOBS_DIR/{job_id}.log and can be tailed live.
- When the subprocess exits, GPU memory is freed automatically.

Job lifecycle:
    QUEUED → RUNNING → DONE | FAILED | CANCELLED

Job record schema (JSON):
    {
        "job_id": "...",
        "status": "QUEUED|RUNNING|DONE|FAILED|CANCELLED",
        "kind": "forget|teach",
        "fact_id": "...",
        "preset": "gentle|standard|strong",
        "method": "npo|refusal|auto",
        "langs": [...],
        "hyperparameters": {...},
        "created_at": "...",
        "started_at": null,
        "finished_at": null,
        "pid": null,
        "adapter_dir": null,
        "metrics": null,
        "error": null,
        "metadata": {...}
    }
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from forgetlab.config import JOBS_DIR
from forgetlab.utils import json_load, json_save, short_id

# Windows doesn't have fcntl; use a simple file-existence lock instead.
_WINDOWS = platform.system() == "Windows"


# ── Status constants ──────────────────────────────────────────────────────────

QUEUED    = "QUEUED"
RUNNING   = "RUNNING"
DONE      = "DONE"
FAILED    = "FAILED"
CANCELLED = "CANCELLED"


# ── Paths ─────────────────────────────────────────────────────────────────────

def _jobs_dir(jobs_dir: str = JOBS_DIR) -> str:
    os.makedirs(jobs_dir, exist_ok=True)
    return jobs_dir


def _job_path(job_id: str, jobs_dir: str = JOBS_DIR) -> str:
    return os.path.join(_jobs_dir(jobs_dir), f"{job_id}.json")


def _log_path(job_id: str, jobs_dir: str = JOBS_DIR) -> str:
    return os.path.join(_jobs_dir(jobs_dir), f"{job_id}.log")


def _lock_path(jobs_dir: str = JOBS_DIR) -> str:
    return os.path.join(_jobs_dir(jobs_dir), ".gpu_lock")


# ── GPU semaphore (one job at a time) ─────────────────────────────────────────

def _gpu_lock_held(jobs_dir: str = JOBS_DIR) -> bool:
    """True if another job is currently holding the GPU lock."""
    lp = _lock_path(jobs_dir)
    if not os.path.exists(lp):
        return False
    # Check if the PID in the lock file is still alive.
    try:
        pid_str = open(lp).read().strip()
        pid = int(pid_str)
        if _WINDOWS:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle == 0:
                # PID is gone – stale lock
                os.remove(lp)
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        else:
            os.kill(pid, 0)   # raises OSError if PID is gone
            return True
    except (ValueError, OSError):
        # Stale lock – remove it.
        try:
            os.remove(lp)
        except OSError:
            pass
        return False


def _acquire_gpu_lock(pid: int, jobs_dir: str = JOBS_DIR) -> None:
    with open(_lock_path(jobs_dir), "w") as f:
        f.write(str(pid))


def _release_gpu_lock(jobs_dir: str = JOBS_DIR) -> None:
    lp = _lock_path(jobs_dir)
    if os.path.exists(lp):
        os.remove(lp)


# ── Job record helpers ────────────────────────────────────────────────────────

def create_job(
    kind: str,
    fact_id: str,
    preset: str,
    method: str,
    langs: list[str],
    hyperparameters: dict[str, Any],
    metadata: dict[str, Any],
    jobs_dir: str = JOBS_DIR,
) -> str:
    """Create a new job record and return its job_id."""
    job_id = short_id()
    record = {
        "job_id": job_id,
        "status": QUEUED,
        "kind": kind,
        "fact_id": fact_id,
        "preset": preset,
        "method": method,
        "langs": langs,
        "hyperparameters": hyperparameters,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "adapter_dir": None,
        "metrics": None,
        "error": None,
        "metadata": metadata,
    }
    json_save(record, _job_path(job_id, jobs_dir))
    return job_id


def load_job(job_id: str, jobs_dir: str = JOBS_DIR) -> dict | None:
    return json_load(_job_path(job_id, jobs_dir))


def update_job(job_id: str, updates: dict[str, Any], jobs_dir: str = JOBS_DIR) -> None:
    """Atomically update fields of a job record."""
    record = load_job(job_id, jobs_dir) or {}
    record.update(updates)
    json_save(record, _job_path(job_id, jobs_dir))


def list_jobs(jobs_dir: str = JOBS_DIR) -> list[dict]:
    """Return all job records, newest first."""
    d = _jobs_dir(jobs_dir)
    records = []
    for fname in os.listdir(d):
        if fname.endswith(".json") and not fname.startswith("."):
            rec = json_load(os.path.join(d, fname))
            if rec and "job_id" in rec:
                records.append(rec)
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def delete_job(job_id: str, jobs_dir: str = JOBS_DIR) -> None:
    """Remove job record and log file."""
    for path in (_job_path(job_id, jobs_dir), _log_path(job_id, jobs_dir)):
        if os.path.exists(path):
            os.remove(path)


# ── Log streaming ─────────────────────────────────────────────────────────────

def tail_log(job_id: str, n_lines: int = 40, jobs_dir: str = JOBS_DIR) -> str:
    """Return the last *n_lines* lines of the job log, or empty string."""
    lp = _log_path(job_id, jobs_dir)
    if not os.path.exists(lp):
        return ""
    with open(lp, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


def stream_log(job_id: str, jobs_dir: str = JOBS_DIR):
    """Generator that yields new log lines as they appear (for live UI streaming)."""
    lp = _log_path(job_id, jobs_dir)
    with open(lp, encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                job = load_job(job_id, jobs_dir)
                if job and job["status"] in (DONE, FAILED, CANCELLED):
                    break
                time.sleep(0.3)


# ── Subprocess launcher ───────────────────────────────────────────────────────

def launch_job(job_id: str, jobs_dir: str = JOBS_DIR) -> subprocess.Popen | None:
    """Launch a job as a subprocess.

    Returns the Popen object, or None if the GPU lock is held.
    """
    if _gpu_lock_held(jobs_dir):
        print(f"[jobs] GPU lock held — job {job_id} will wait until current job finishes.")
        return None

    log_path = _log_path(job_id, jobs_dir)
    worker_cmd = [
        sys.executable, "-m", "forgetlab.worker",
        "--job_id", job_id,
        "--jobs_dir", jobs_dir,
    ]

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            worker_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )

    _acquire_gpu_lock(proc.pid, jobs_dir)
    update_job(job_id, {
        "status": RUNNING,
        "pid": proc.pid,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, jobs_dir)

    return proc


def cancel_job(job_id: str, jobs_dir: str = JOBS_DIR) -> bool:
    """Send SIGTERM to a running job.  Returns True if a signal was sent."""
    rec = load_job(job_id, jobs_dir)
    if not rec or rec["status"] != RUNNING or not rec.get("pid"):
        return False
    pid = rec["pid"]
    try:
        if _WINDOWS:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        update_job(job_id, {"status": CANCELLED, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, jobs_dir)
        _release_gpu_lock(jobs_dir)
        return True
    except ProcessLookupError:
        return False


# ── Rollback helpers ──────────────────────────────────────────────────────────

def rollback_job(job_id: str, jobs_dir: str = JOBS_DIR) -> bool:
    """Mark a job as rolled back (adapter disabled).

    The actual adapter is not deleted — it stays on disk so it can be re-applied.
    Returns True if the job was in DONE status.
    """
    rec = load_job(job_id, jobs_dir)
    if not rec or rec["status"] != DONE:
        return False
    update_job(job_id, {"status": "ROLLED_BACK"}, jobs_dir)
    return True


def make_permanent(job_id: str, engine, jobs_dir: str = JOBS_DIR) -> bool:
    """Merge the job's adapter into the base weights permanently.

    After this operation the adapter can no longer be removed.
    Returns True on success.
    """
    rec = load_job(job_id, jobs_dir)
    if not rec or not rec.get("adapter_dir"):
        return False
    adapter_dir = rec["adapter_dir"]
    try:
        from peft import PeftModel
        engine._ensure_loaded()
        peft_model = PeftModel.from_pretrained(engine._model, adapter_dir)
        engine._model = peft_model.merge_and_unload()
        update_job(job_id, {"status": "PERMANENT"}, jobs_dir)
        return True
    except Exception as e:
        print(f"[jobs] make_permanent failed: {e}")
        return False
