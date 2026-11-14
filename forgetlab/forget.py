"""ForgetLab: forget.py

NPO + retain anchor, refusal tuning fallback, snapshot strategy, and
auto-select.  This module is called by the subprocess worker (forgetlab/worker.py).

Three methods
─────────────
  npo       – NPO with rolling-mean stop rule and snapshot strategy (primary)
  refusal   – Refusal tuning: train model to respond with a short refusal
  auto      – Try NPO snapshots; fall back to refusal if acceptance thresholds
              not met; honestly report failure if neither works.

The stop rule (fixed from the original unlearn_npo.py)
───────────────────────────────────────────────────────
Old rule: `if f_nll >= forget_target: break` on a single micro-batch — fires
on spikes.

New rule (per the prompt spec):
  1. Keep a rolling deque of the last ROLLING_WINDOW forget NLL values.
  2. Save snapshots at rolling-mean levels in SNAPSHOT_LEVELS.
  3. Every PROBE_EVERY_N_STEPS steps, generate on one forget probe; if the
     repetition ratio exceeds MAX_REPETITION_RATIO, abort and restore the
     last good snapshot.
  4. Emit forget_term and retain_term separately to the log at every step.
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import shutil
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from forgetlab.config import (
    FORGET_PRESETS,
    GEN_CAP,
    GEN_CAP_DEFAULT,
    LORA_TARGETS_ATTN,
    LORA_TARGETS_MLP,
    LORA_TARGETS_ALL,
    MAX_REPETITION_RATIO,
    MAX_TRAIN_LEN,
    NPO_BETA,
    PROBE_EVERY_N_STEPS,
    REFUSAL_LR,
    REFUSAL_MAX_STEPS,
    REFUSAL_NEEDS_REVIEW,
    REFUSAL_STRINGS,
    RETAIN_N_PER_LANG,
    ROLLING_WINDOW,
    SNAPSHOT_LEVELS,
    UNLEARN_CKPT_FMT,
)
from forgetlab.utils import repetition_ratio, is_looping

_LORA_TARGET_MAP = {
    "all":  LORA_TARGETS_ALL,
    "mlp":  LORA_TARGETS_MLP,
    "attn": LORA_TARGETS_ATTN,
}


# ── Helpers (re-use src imports) ──────────────────────────────────────────────

def _imports():
    from src.common.model_utils import (
        DEVICE, generate_answers, load_injected, load_tokenizer,
        make_lora, make_loader, make_scaler, trainable_params, pick_dtype,
    )
    from src.training.unlearn_npo import answer_logp, seq_stats, mean_nll, retain_loss
    from forgetlab.facts import load_qa_records
    return (DEVICE, generate_answers, load_injected, load_tokenizer,
            make_lora, make_loader, make_scaler, trainable_params, pick_dtype,
            answer_logp, seq_stats, mean_nll, retain_loss, load_qa_records)


# ── NPO with rolling stop rule ────────────────────────────────────────────────

def run_npo(
    model,
    tok,
    forget_records: list[dict],
    retain_records: list[dict],
    preset: str = "standard",
    job_id: str = "unknown",
    out_dir: str = "checkpoints/forgetlab_npo",
    log_callback: Callable[[dict], None] | None = None,
    lang: str = "english",
) -> dict[str, Any]:
    """Run NPO with retain anchor and the fixed rolling-mean stop rule.

    Returns a result dict:
        {"snapshots": [...], "history": [...], "best_snapshot": str, "stopped_reason": str}
    """
    (DEVICE, generate_answers, _, _, make_lora, make_loader, make_scaler,
     trainable_params, pick_dtype, answer_logp, seq_stats, mean_nll,
     retain_loss_fn, _) = _imports()

    cfg = FORGET_PRESETS[preset]
    lr         = cfg["lr"]
    lora_r     = cfg["lora_r"]
    lora_tgts  = _LORA_TARGET_MAP[cfg["lora_targets"]]
    max_steps  = cfg["max_steps"]
    ret_weight = cfg["retain_weight"]
    beta       = NPO_BETA
    max_len    = MAX_TRAIN_LEN.get(lang, 384)
    gen_cap    = GEN_CAP.get(lang, GEN_CAP_DEFAULT)

    peft_model = make_lora(model, r=lora_r, alpha=2 * lora_r, targets=lora_tgts)
    forget_loader = make_loader(tok, forget_records, bs=2, max_len=max_len)
    retain_loader = make_loader(tok, retain_records, bs=2, max_len=max_len)
    params = trainable_params(peft_model)
    opt    = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = make_scaler()

    history: list[dict] = []
    nll_window: collections.deque = collections.deque(maxlen=ROLLING_WINDOW)

    # Snapshots: {level_str: snapshot_dir}
    snapshots_saved: dict[str, str] = {}
    snapshot_levels_remaining = sorted(SNAPSHOT_LEVELS)
    last_good_snapshot: str | None = None
    stopped_reason = "max_steps"

    os.makedirs(out_dir, exist_ok=True)

    peft_model.train()
    step_iter = zip(
        range(1, max_steps + 1),
        itertools.cycle(forget_loader),
        itertools.cycle(retain_loader),
    )

    for step, fb, rb in step_iter:
        # ── Forget term (NPO) ───────────────────────────────────────────────
        logp, target, mask = answer_logp(peft_model, fb)
        lp, cnt = seq_stats(logp, target, mask)
        f_nll = mean_nll(lp, cnt)

        with torch.no_grad(), peft_model.disable_adapter():
            ref_lp, _ = seq_stats(*answer_logp(peft_model, fb))
        forget_term = (2.0 / beta) * F.softplus(beta * (lp - ref_lp)).mean()

        scaler.scale(forget_term).backward()
        del logp

        # ── Retain term ─────────────────────────────────────────────────────
        r_loss, r_nll = retain_loss_fn(peft_model, rb, "kl")
        scaler.scale(ret_weight * r_loss).backward()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        f_nll_val  = f_nll.item()
        r_nll_val  = r_nll.item()
        forget_obj = forget_term.item()
        nll_window.append(f_nll_val)
        rolling_mean = sum(nll_window) / len(nll_window)

        log_entry = {
            "step": step,
            "forget_nll": round(f_nll_val, 4),
            "retain_nll": round(r_nll_val, 4),
            "forget_term": round(forget_obj, 4),
            "rolling_mean": round(rolling_mean, 4),
        }
        history.append(log_entry)
        print(
            f"step {step:4d}  forget_nll {f_nll_val:.3f}  retain_nll {r_nll_val:.3f}"
            f"  forget_term {forget_obj:.3f}  rolling_mean {rolling_mean:.3f}",
            flush=True,
        )
        if log_callback:
            log_callback(log_entry)

        # ── Snapshots at rolling-mean levels ─────────────────────────────────
        if snapshot_levels_remaining and rolling_mean >= snapshot_levels_remaining[0]:
            level = snapshot_levels_remaining.pop(0)
            snap_dir = os.path.join(out_dir, f"snap_{level:.2f}")
            peft_model.save_pretrained(snap_dir)
            snapshots_saved[f"{level:.2f}"] = snap_dir
            last_good_snapshot = snap_dir
            print(f"[forget] snapshot saved at rolling_mean={rolling_mean:.3f} -> {snap_dir}", flush=True)

        # ── Degeneration check every PROBE_EVERY_N_STEPS steps ───────────────
        if step % PROBE_EVERY_N_STEPS == 0 and forget_records:
            probe_q = forget_records[0]["question"]
            with torch.no_grad():
                preds = generate_answers(peft_model, tok, [probe_q], max_new_tokens=gen_cap)
            pred = preds[0]
            rep_ratio = repetition_ratio(pred)
            print(f"[forget] step {step} probe: '{pred[:80]}…'  rep_ratio={rep_ratio:.2f}", flush=True)
            if rep_ratio >= MAX_REPETITION_RATIO:
                print(f"[forget] repetition ratio {rep_ratio:.2f} ≥ {MAX_REPETITION_RATIO} — aborting, restoring last good snapshot", flush=True)
                stopped_reason = "degeneration_abort"
                break
            else:
                # Keep single latest known fluent checkpoint (avoids bloating snapshot list)
                clean_snap = os.path.join(out_dir, "snap_last_fluent")
                peft_model.save_pretrained(clean_snap)
                snapshots_saved["last_fluent"] = clean_snap
                last_good_snapshot = clean_snap

    # ── Save final adapter ────────────────────────────────────────────────────
    final_dir = os.path.join(out_dir, "final")
    if last_good_snapshot and stopped_reason == "degeneration_abort":
        shutil.copytree(last_good_snapshot, final_dir, dirs_exist_ok=True)
        print(f"[forget] restored last good snapshot: {last_good_snapshot}", flush=True)
    else:
        peft_model.save_pretrained(final_dir)

    return {
        "snapshots": snapshots_saved,
        "history": history,
        "best_snapshot": final_dir,
        "stopped_reason": stopped_reason,
        "preset": preset,
        "lang": lang,
    }


# ── Refusal tuning ────────────────────────────────────────────────────────────

def build_refusal_records(
    forget_records: list[dict],
    retain_records: list[dict],
    target_langs: list[str],
) -> list[dict]:
    """Build training records for refusal tuning.

    For each forget question (original + paraphrases + translations), create a
    record where the answer is the refusal string in that language.
    Mixed with retain records.
    """
    if REFUSAL_NEEDS_REVIEW:
        print(
            "[forget] WARNING: Machine-translated refusal strings are pending your review.\n"
            "  Hindi:   " + REFUSAL_STRINGS.get("hindi", "") + "\n"
            "  Kannada: " + REFUSAL_STRINGS.get("kannada", "") + "\n"
            "  Proceeding with machine translations — set config.REFUSAL_NEEDS_REVIEW=False to suppress this.",
            flush=True,
        )

    records = []
    for rec in forget_records:
        for lang in target_langs:
            q = rec.get("question") if lang == "english" else rec.get(f"question_{lang[:2]}", rec["question"])
            refusal = REFUSAL_STRINGS.get(lang, REFUSAL_STRINGS["english"])
            records.append({"question": q, "answer": refusal})

    # Mix retain (use English retain to preserve other knowledge)
    records += retain_records[:min(len(retain_records), len(forget_records) * 2)]
    return records


def run_refusal_tuning(
    model,
    tok,
    forget_records: list[dict],
    retain_records: list[dict],
    target_langs: list[str],
    out_dir: str = "checkpoints/forgetlab_refusal",
    log_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Train the model to answer forget questions with a refusal string."""
    (DEVICE, _, _, _, make_lora, make_loader, make_scaler,
     trainable_params, pick_dtype, _, _, _, _, _) = _imports()
    from src.common.model_utils import train_step

    combined = build_refusal_records(forget_records, retain_records, target_langs)
    peft_model = make_lora(model, r=8, alpha=16, targets=LORA_TARGETS_MLP)
    loader = make_loader(tok, combined, bs=2)
    params = trainable_params(peft_model)
    opt    = torch.optim.AdamW(params, lr=REFUSAL_LR, weight_decay=0.0)
    scaler = make_scaler()

    history = []
    peft_model.train()
    for step, batch in enumerate(
        itertools.islice(itertools.cycle(loader), REFUSAL_MAX_STEPS), 1
    ):
        loss = train_step(peft_model, batch, opt, scaler, params)
        entry = {"step": step, "loss": round(loss, 4), "method": "refusal"}
        history.append(entry)
        print(f"[refusal] step {step:3d}  loss {loss:.3f}", flush=True)
        if log_callback:
            log_callback(entry)

    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    return {"history": history, "adapter_dir": out_dir, "method": "refusal"}


# ── Auto-select ───────────────────────────────────────────────────────────────

def auto_select(
    npo_result: dict,
    model,
    tok,
    forget_records: list[dict],
    retain_records: list[dict],
    target_langs: list[str],
    judge,
    out_dir: str = "checkpoints/forgetlab_auto",
    log_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Pick the NPO snapshot with highest selectivity meeting acceptance thresholds.

    If none meet the thresholds, run refusal tuning.  If that also fails,
    return an honest failure report.
    """
    from peft import PeftModel
    from forgetlab.verify import run_probes, compute_metrics, build_retain_probes
    from forgetlab.facts import build_retain_probes as _build_retain

    # Build a minimal probe set for fast snapshot evaluation.
    retain_pool = _build_retain(n_per_lang=10, langs=["english"])
    forget_probes = [
        {"lang": "english", "question": r["question"], "answer": r["answer"], "kind": "direct"}
        for r in forget_records[:10]
    ]
    retain_probes_before = None  # will be populated on first eval

    best: dict | None = None
    best_selectivity = -999.0
    best_npo_overall: dict | None = None
    best_npo_selectivity = -999.0

    snapshots = npo_result.get("snapshots", {})
    # Also evaluate the final adapter
    snapshots["final"] = npo_result.get("best_snapshot", "")

    for level_str, snap_dir in snapshots.items():
        if not snap_dir or not os.path.isdir(snap_dir):
            continue
        try:
            snap_model = PeftModel.from_pretrained(model, snap_dir)
        except Exception as e:
            print(f"[auto] could not load snapshot {level_str}: {e}", flush=True)
            continue

        # Quick generation-based eval (no NLL — fast)
        from forgetlab.engine import ModelHost
        # We use a temporary mini-host that wraps the snap_model
        class _TmpHost:
            def __init__(self, m, t, l):
                self.m, self.t, self.l = m, t, l
            def generate(self, q, lang=None, max_new_tokens=None):
                cap = max_new_tokens or GEN_CAP.get(self.l, GEN_CAP_DEFAULT)
                from src.common.model_utils import generate_answers
                return generate_answers(self.m, self.t, [q], max_new_tokens=cap)[0]

        host = _TmpHost(snap_model, tok, target_langs[0] if target_langs else "english")

        # Retain before (computed once against the original model)
        if retain_probes_before is None:
            orig_host = _TmpHost(model, tok, "english")
            retain_probes_before = run_probes(retain_pool, orig_host, judge)

        forget_after  = run_probes(forget_probes,  host, judge)
        retain_after  = run_probes(retain_pool,    host, judge)
        metrics = compute_metrics([], forget_after, retain_probes_before, retain_after)

        print(
            f"[auto] snapshot {level_str}: "
            f"forget_rate={metrics.forget_rate:.2f}  "
            f"retention={metrics.retention:.2f}  "
            f"selectivity={metrics.selectivity:.2f}  "
            f"accepted={metrics.is_accepted}",
            flush=True,
        )

        if metrics.selectivity > best_npo_selectivity:
            best_npo_selectivity = metrics.selectivity
            best_npo_overall = {"snap_dir": snap_dir, "level": level_str, "metrics": metrics.to_dict()}

        if metrics.is_accepted and metrics.selectivity > best_selectivity:
            best_selectivity = metrics.selectivity
            best = {"snap_dir": snap_dir, "level": level_str, "metrics": metrics.to_dict()}
            if best_selectivity >= 0.95:
                print(f"[auto] optimal snapshot found ('{level_str}', selectivity={best_selectivity:.2f}) — stopping search early.", flush=True)
                del snap_model
                break

        del snap_model

    if best is not None:
        shutil.copytree(best["snap_dir"], out_dir, dirs_exist_ok=True)
        print(f"[auto] selected snapshot {best['level']} with selectivity {best_selectivity:.2f}", flush=True)
        return {
            "method": "npo_auto",
            "adapter_dir": out_dir,
            "selected_snapshot": best["level"],
            "selectivity": best_selectivity,
            "metrics": best["metrics"],
            "success": True,
        }

    # NPO failed strict threshold — try refusal tuning in a separate dir
    print("[auto] no NPO snapshot met acceptance thresholds — falling back to refusal tuning", flush=True)
    refusal_dir = f"{out_dir}_refusal"
    refusal_result = run_refusal_tuning(
        model, tok, forget_records, retain_records, target_langs,
        out_dir=refusal_dir, log_callback=log_callback,
    )

    # Evaluate refusal adapter
    try:
        refusal_model = PeftModel.from_pretrained(model, refusal_dir)
        host = _TmpHost(refusal_model, tok, target_langs[0] if target_langs else "english")
        forget_after  = run_probes(forget_probes, host, judge)
        retain_after  = run_probes(retain_pool,   host, judge)
        metrics = compute_metrics([], forget_after, retain_probes_before or [], retain_after)
        if metrics.is_accepted:
            shutil.copytree(refusal_dir, out_dir, dirs_exist_ok=True)
            refusal_result["adapter_dir"] = out_dir
            refusal_result["metrics"] = metrics.to_dict()
            refusal_result["success"] = True
            return refusal_result
    except Exception as e:
        print(f"[auto] refusal evaluation failed: {e}", flush=True)

    # Both failed strict thresholds — restore best NPO adapter so real unlearning is preserved!
    print("[auto] strict acceptance thresholds not met — retaining best NPO unlearning snapshot.", flush=True)
    chosen_snap = best_npo_overall or ({"snap_dir": snapshots.get("final", ""), "level": "final", "metrics": {}} if snapshots.get("final") else None)
    if chosen_snap and chosen_snap.get("snap_dir") and os.path.isdir(chosen_snap["snap_dir"]):
        shutil.copytree(chosen_snap["snap_dir"], out_dir, dirs_exist_ok=True)
        print(f"[auto] saved best NPO snapshot '{chosen_snap.get('level')}' to {out_dir}", flush=True)

    return {
        "method": "auto",
        "success": False,
        "adapter_dir": out_dir,
        "selected_snapshot": chosen_snap.get("level") if chosen_snap else "final",
        "selectivity": best_npo_selectivity,
        "metrics": chosen_snap.get("metrics") if chosen_snap else None,
        "error": "No method met 100% acceptance threshold. Best NPO unlearned snapshot was retained.",
        "npo_snapshots_tried": list(snapshots.keys()),
    }

