"""ForgetLab: worker.py

Subprocess entry point for a single forget or teach job.

Called by jobs.py:
    python -m forgetlab.worker --job_id <id> --jobs_dir <dir>

The worker:
1. Loads the job record
2. Loads the model (using engine.py)
3. Dispatches to teach.py or forget.py
4. Writes results back to the job record
5. Frees GPU memory before exiting
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from forgetlab.jobs import (
    DONE, FAILED, RUNNING,
    _release_gpu_lock, load_job, update_job,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", required=True)
    ap.add_argument("--jobs_dir", default="forgetlab_jobs")
    args = ap.parse_args()

    job_id   = args.job_id
    jobs_dir = args.jobs_dir

    record = load_job(job_id, jobs_dir)
    if record is None:
        print(f"[worker] ERROR: job {job_id} not found in {jobs_dir}", flush=True)
        sys.exit(1)

    print(f"[worker] starting job {job_id}  kind={record['kind']}", flush=True)
    t0 = time.time()

    try:
        result = _run_job(record, jobs_dir)
        elapsed = round(time.time() - t0, 1)
        update_job(job_id, {
            "status": DONE,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": result.get("metrics"),
            "adapter_dir": result.get("adapter_dir"),
            "elapsed_s": elapsed,
        }, jobs_dir)
        print(f"[worker] job {job_id} DONE in {elapsed}s", flush=True)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[worker] FAILED:\n{tb}", flush=True)
        update_job(job_id, {
            "status": FAILED,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": str(exc),
        }, jobs_dir)
    finally:
        _release_gpu_lock(jobs_dir)
        # Free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _run_job(record: dict, jobs_dir: str) -> dict:
    """Dispatch to the appropriate handler based on record['kind']."""
    kind = record["kind"]
    if kind == "forget":
        return _run_forget(record, jobs_dir)
    elif kind == "teach":
        return _run_teach(record, jobs_dir)
    elif kind == "teach_then_forget":
        return _run_teach_then_forget(record, jobs_dir)
    else:
        raise ValueError(f"Unknown job kind: {kind!r}")


def _run_forget(record: dict, jobs_dir: str) -> dict:
    """Execute a forget job end-to-end with full verification."""
    import json
    import os
    from peft import PeftModel
    from forgetlab.config import INJECT_CKPT, BASE_MODEL_ID, GEN_CAP, GEN_CAP_DEFAULT, MAX_TRAIN_LEN
    from forgetlab.facts import FactStore, load_qa_records, build_retain_probes
    from forgetlab.forget import run_npo, auto_select
    from forgetlab.verify import run_probes, compute_metrics
    from src.common.model_utils import load_tokenizer, load_injected, generate_answers, encode_qa, pick_dtype
    from src.eval.judge import FactJudge

    fact_id  = record["fact_id"]
    preset   = record.get("preset", "standard")
    method   = record.get("method", "auto")
    langs    = record.get("langs", ["english"])
    job_id   = record["job_id"]

    # Load fact
    store = FactStore(f"{jobs_dir}/facts")
    fact  = store.load(fact_id)
    if fact is None:
        raise RuntimeError(f"Fact {fact_id} not found in store.")

    # Load model
    inject_dir = record.get("metadata", {}).get("inject_dir", INJECT_CKPT)
    tok   = load_tokenizer(BASE_MODEL_ID)
    model = load_injected(inject_dir, BASE_MODEL_ID)

    # Collect forget records (question, answer)
    forget_records = [{"question": fact.question, "answer": fact.answer}]
    for p in fact.paraphrases:
        forget_records.append({"question": p, "answer": fact.answer})
    for lang in langs:
        if lang != "english" and lang in fact.translations:
            t = fact.translations[lang]
            forget_records.append({"question": t["question"], "answer": t["answer"]})

    retain_records = []
    for lang in ["english", "hindi", "kannada"]:
        try:
            retain_records += load_qa_records(lang, "retain", limit=100)
        except FileNotFoundError:
            pass

    out_dir = f"checkpoints/forgetlab_{job_id}"
    os.makedirs("results/logs", exist_ok=True)
    history_log_path = f"results/logs/{job_id}_history.json"
    history_entries = []

    def log_callback(entry):
        history_entries.append(entry)
        update_job(job_id, {"_last_step": entry}, jobs_dir)
        try:
            with open(history_log_path, "w", encoding="utf-8") as f:
                json.dump(history_entries, f, indent=2)
        except Exception:
            pass

    # Build judge
    corpus = {}
    for l in ("english", "hindi", "kannada"):
        try:
            corpus[l] = [r["answer"] for r in load_qa_records(l, "forget", limit=200)]
        except FileNotFoundError:
            pass
    judge = FactJudge(corpus)

    # Run unlearning
    if method == "refusal":
        from forgetlab.forget import run_refusal_tuning
        result = run_refusal_tuning(
            model, tok, forget_records, retain_records, langs,
            out_dir=f"{out_dir}_refusal", log_callback=log_callback,
        )
        adapter_dir = result.get("adapter_dir")
    else:
        # NPO is run for both 'npo' and 'auto'
        npo_result = run_npo(
            model, tok, forget_records, retain_records,
            preset=preset, job_id=job_id,
            out_dir=f"{out_dir}_npo",
            log_callback=log_callback,
            lang=langs[0],
        )
        if method == "npo":
            adapter_dir = npo_result["best_snapshot"]
            result = {"adapter_dir": adapter_dir, "npo_result": npo_result, "method": "npo"}
        else:  # "auto"
            result = auto_select(
                npo_result, model, tok, forget_records, retain_records, langs,
                judge=judge, out_dir=out_dir, log_callback=log_callback,
            )
            adapter_dir = result.get("adapter_dir")

    # ── Full Verification ───────────────────────────────────────────────────
    if adapter_dir and os.path.exists(adapter_dir):
        print(f"[worker] running verification on adapter '{adapter_dir}'…", flush=True)

        class ModelWrapper:
            def __init__(self, m, t):
                self.m, self.t = m, t
            def generate(self, q, lang="english", max_new_tokens=None):
                cap = max_new_tokens or GEN_CAP.get(lang, GEN_CAP_DEFAULT)
                return generate_answers(self.m, self.t, [q], max_new_tokens=cap)[0]
            def probe_nll(self, q, a, lang="english"):
                import torch
                from src.common.model_utils import DEVICE
                max_len = MAX_TRAIN_LEN.get(lang, 384)
                ids, labels = encode_qa(self.t, q, a, max_len=max_len)
                inp = torch.tensor([ids]).to(DEVICE)
                lab = torch.tensor([labels]).to(DEVICE)
                att = torch.ones_like(inp)
                self.m.eval()
                with torch.no_grad():
                    with torch.autocast(DEVICE, dtype=pick_dtype(), enabled=(DEVICE == "cuda")):
                        out = self.m(input_ids=inp, attention_mask=att, labels=lab)
                return out.loss.item()

        # Build full probe sets
        all_forget_probes = fact.forget_probes(langs=["english", "hindi", "kannada"])
        all_retain_probes = build_retain_probes(n_per_lang=25, langs=["english", "hindi", "kannada"])

        # 1. Before unlearning (base model)
        base_host = ModelWrapper(model, tok)
        forget_before = run_probes(all_forget_probes, base_host, judge, compute_nll=True)
        retain_before = run_probes(all_retain_probes, base_host, judge, compute_nll=False)

        # 2. After unlearning (with adapter loaded)
        peft_adapted = PeftModel.from_pretrained(model, adapter_dir)
        adapted_host = ModelWrapper(peft_adapted, tok)
        forget_after = run_probes(all_forget_probes, adapted_host, judge, compute_nll=True)
        retain_after = run_probes(all_retain_probes, adapted_host, judge, compute_nll=False)

        # Transfer nll_before to after objects
        for fb, fa in zip(forget_before, forget_after):
            fa.nll_before = fb.nll_after
        for rb, ra in zip(retain_before, retain_after):
            ra.nll_before = rb.nll_after

        metrics = compute_metrics(forget_before, forget_after, retain_before, retain_after)
        result["metrics"] = metrics.to_dict()
        result["probes_after"] = [p.to_dict() for p in forget_after]
        del peft_adapted

        print(
            f"[worker] Verification complete:\n"
            f"  forget_rate: {metrics.forget_rate:.2f} (95% CI: {metrics.ci.get('forget_rate')})\n"
            f"  retention:   {metrics.retention:.2f}\n"
            f"  selectivity: {metrics.selectivity:.2f}\n"
            f"  accepted:    {metrics.is_accepted}",
            flush=True,
        )

    store.save(fact)
    return result


def _run_teach_then_forget(record: dict, jobs_dir: str) -> dict:
    """Execute teach first; if successful, immediately run forget on the new checkpoint."""
    print("[worker] Starting 'Teach then forget' pipeline...", flush=True)
    teach_res = _run_teach(record, jobs_dir)
    if not teach_res.get("success"):
        return {
            "success": False,
            "error": f"Teaching failed: {teach_res.get('reason')}",
            "metrics": None,
        }

    # Pass the newly created world checkpoint to forget step
    world_ckpt = teach_res["world_checkpoint"]
    record["metadata"] = record.get("metadata", {})
    record["metadata"]["inject_dir"] = world_ckpt
    print(f"[worker] Teaching succeeded. Proceeding to forget using world checkpoint {world_ckpt}...", flush=True)

    forget_res = _run_forget(record, jobs_dir)
    forget_res["teach_result"] = teach_res
    return forget_res


def _run_teach(record: dict, jobs_dir: str) -> dict:
    """Execute a teach job end-to-end."""
    from forgetlab.config import INJECT_CKPT, BASE_MODEL_ID
    from forgetlab.facts import FactStore, load_qa_records, Fact
    from forgetlab.teach import run_teach
    from forgetlab.utils import short_id
    from src.common.model_utils import load_tokenizer, load_injected

    fact_id = record["fact_id"]
    job_id  = record["job_id"]
    store   = FactStore(f"{jobs_dir}/facts")
    fact    = store.load(fact_id)

    inject_dir = record.get("metadata", {}).get("inject_dir", INJECT_CKPT)
    tok   = load_tokenizer(BASE_MODEL_ID)
    model = load_injected(inject_dir, BASE_MODEL_ID)

    retain_records = []
    try:
        retain_records += load_qa_records("english", "retain", limit=200)
    except FileNotFoundError:
        pass

    world_version = f"user_{short_id(4)}"

    def log_callback(entry):
        update_job(job_id, {"_last_step": entry}, jobs_dir)

    result = run_teach(
        model, tok, fact, retain_records,
        world_version=world_version,
        log_callback=log_callback,
    )

    # Update fact with new injection version
    if result["success"]:
        updated_fact = Fact(**{**fact.to_dict(), "injection_version": world_version})
        store.save(updated_fact)
        result["adapter_dir"] = result["world_checkpoint"]

    return result


if __name__ == "__main__":
    main()

