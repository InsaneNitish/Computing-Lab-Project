"""ForgetLab: cli.py

Command-line interface.  Run as:

    python -m forgetlab.cli forget --question "..." --answer "..."
    python -m forgetlab.cli teach  --question "..." --answer "..."
    python -m forgetlab.cli verify --job_id <id>
    python -m forgetlab.cli jobs [--list | --status <id> | --cancel <id>]
    python -m forgetlab.cli tofu --search "..." | --index 42
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_forget(args):
    from forgetlab.facts import Fact, FactStore
    from forgetlab.jobs import create_job, launch_job, list_jobs
    from forgetlab.utils import stable_id, build_metadata
    from forgetlab.config import BASE_MODEL_ID, INJECT_CKPT, FORGET_PRESETS

    fact = Fact(
        fact_id=stable_id(args.question, args.answer),
        question=args.question,
        answer=args.answer,
        paraphrases=args.paraphrases or [],
        source="user",
    )
    store = FactStore()
    store.save(fact)

    preset = args.preset or "standard"
    if preset not in FORGET_PRESETS:
        print(f"Unknown preset '{preset}'. Choose from: {list(FORGET_PRESETS.keys())}")
        sys.exit(1)

    meta = build_metadata(
        base_model=BASE_MODEL_ID,
        injection_version="v1",
        gen_cap=640,
        n=len(fact.forget_probes()),
        seed=0,
        method=args.method or "auto",
        hyperparameters=FORGET_PRESETS[preset],
        inject_dir=INJECT_CKPT,
    )
    meta["inject_dir"] = INJECT_CKPT

    job_id = create_job(
        kind="forget",
        fact_id=fact.fact_id,
        preset=preset,
        method=args.method or "auto",
        langs=args.langs or ["english"],
        hyperparameters=FORGET_PRESETS[preset],
        metadata=meta,
    )
    print(f"[cli] created job {job_id} for fact {fact.fact_id}")
    if not args.dry_run:
        proc = launch_job(job_id)
        if proc:
            print(f"[cli] job {job_id} running (PID {proc.pid}). Check status with: python -m forgetlab.cli jobs --status {job_id}")
        else:
            print(f"[cli] GPU busy — job {job_id} queued.")


def _cmd_teach(args):
    from forgetlab.facts import Fact, FactStore
    from forgetlab.jobs import create_job, launch_job
    from forgetlab.utils import stable_id, build_metadata
    from forgetlab.config import BASE_MODEL_ID, INJECT_CKPT

    fact = Fact(
        fact_id=stable_id(args.question, args.answer),
        question=args.question,
        answer=args.answer,
        paraphrases=args.paraphrases or [],
        source="user",
    )
    store = FactStore()
    store.save(fact)

    meta = build_metadata(
        base_model=BASE_MODEL_ID,
        injection_version="new",
        gen_cap=640,
        n=1,
        seed=0,
        method="teach",
        hyperparameters={},
        inject_dir=INJECT_CKPT,
    )
    meta["inject_dir"] = INJECT_CKPT

    job_id = create_job(
        kind="teach",
        fact_id=fact.fact_id,
        preset="standard",
        method="teach",
        langs=["english"],
        hyperparameters={},
        metadata=meta,
    )
    print(f"[cli] created teach job {job_id} for fact {fact.fact_id}")
    if not args.dry_run:
        proc = launch_job(job_id)
        if proc:
            print(f"[cli] teach job {job_id} running (PID {proc.pid})")
        else:
            print(f"[cli] GPU busy — job {job_id} queued.")


def _cmd_verify(args):
    from forgetlab.jobs import load_job, tail_log
    rec = load_job(args.job_id)
    if rec is None:
        print(f"Job {args.job_id} not found.")
        sys.exit(1)
    print(f"Status: {rec['status']}")
    if rec.get("metrics"):
        print(json.dumps(rec["metrics"], indent=2))
    if args.log:
        print("\n--- log tail ---")
        print(tail_log(args.job_id))


def _cmd_jobs(args):
    from forgetlab.jobs import list_jobs, cancel_job, tail_log

    if args.cancel:
        ok = cancel_job(args.cancel)
        print(f"Cancelled: {ok}")
        return

    if args.status:
        from forgetlab.jobs import load_job
        rec = load_job(args.status)
        if rec:
            print(json.dumps(rec, indent=2))
        else:
            print("Not found.")
        return

    if args.log:
        print(tail_log(args.log))
        return

    # List all
    jobs = list_jobs()
    if not jobs:
        print("No jobs found.")
        return
    for j in jobs:
        print(f"  {j['job_id']}  {j['status']:12s}  {j['kind']:8s}  {j.get('created_at', '')}")


def _cmd_tofu(args):
    from forgetlab.facts import TOFU

    if args.search:
        hits = TOFU.search(args.search)
        for idx, fact in hits:
            print(f"[{idx:3d}] {fact.question}")
            print(f"       {fact.answer}")
        if not hits:
            print("No matches.")
        return

    if args.index is not None:
        fact = TOFU.get(args.index)
        print(json.dumps(fact.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"TOFU forget05 has {len(TOFU)} QA pairs.")


def main():
    ap = argparse.ArgumentParser(
        prog="python -m forgetlab.cli",
        description="ForgetLab CLI",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # forget
    p_forget = sub.add_parser("forget", help="Forget a fact")
    p_forget.add_argument("--question", required=True)
    p_forget.add_argument("--answer", required=True)
    p_forget.add_argument("--paraphrases", nargs="*")
    p_forget.add_argument("--preset", default="standard", choices=["gentle", "standard", "strong"])
    p_forget.add_argument("--method", default="auto", choices=["npo", "refusal", "auto"])
    p_forget.add_argument("--langs", nargs="+", default=["english"])
    p_forget.add_argument("--dry_run", action="store_true", help="Create the job but don't launch it")

    # teach
    p_teach = sub.add_parser("teach", help="Teach a new fact")
    p_teach.add_argument("--question", required=True)
    p_teach.add_argument("--answer", required=True)
    p_teach.add_argument("--paraphrases", nargs="*")
    p_teach.add_argument("--dry_run", action="store_true")

    # verify
    p_verify = sub.add_parser("verify", help="Show job verification results")
    p_verify.add_argument("--job_id", required=True)
    p_verify.add_argument("--log", action="store_true", help="Also show log tail")

    # jobs
    p_jobs = sub.add_parser("jobs", help="Manage jobs")
    p_jobs.add_argument("--list", action="store_true")
    p_jobs.add_argument("--status", metavar="JOB_ID")
    p_jobs.add_argument("--cancel", metavar="JOB_ID")
    p_jobs.add_argument("--log", metavar="JOB_ID")

    # tofu
    p_tofu = sub.add_parser("tofu", help="Browse TOFU forget05 dataset")
    p_tofu.add_argument("--search", metavar="QUERY")
    p_tofu.add_argument("--index", type=int, metavar="N")

    args = ap.parse_args()
    dispatch = {
        "forget": _cmd_forget,
        "teach":  _cmd_teach,
        "verify": _cmd_verify,
        "jobs":   _cmd_jobs,
        "tofu":   _cmd_tofu,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()

