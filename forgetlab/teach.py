"""ForgetLab: teach.py

Teach a new fact to the model via a short LoRA fine-tune, then merge it into
a new 'world' checkpoint.  Teaching is mixed with retain-pair replay so
existing knowledge is not damaged.

Stopping criteria:
  1. Direct-question NLL < TEACH_NLL_TARGET (0.3)
  2. NLI judge says the model entails the answer on ≥ TEACH_NLI_MIN_FRAC of paraphrases

If neither criterion is met after TEACH_MAX_EPOCHS, report teaching failure.

Output: a new merged checkpoint at WORLD_CKPT_FMT.format(version=...).
"""

from __future__ import annotations

import itertools
import math
import os
import time
from typing import Any, Callable

import torch
from transformers import get_linear_schedule_with_warmup

from forgetlab.config import (
    LORA_TARGETS_ALL,
    MAX_TRAIN_LEN,
    RETAIN_N_PER_LANG,
    TEACH_LR,
    TEACH_MAX_EPOCHS,
    TEACH_NLI_MIN_FRAC,
    TEACH_NLL_TARGET,
    WORLD_CKPT_FMT,
)
from forgetlab.facts import Fact, load_qa_records


def run_teach(
    model,
    tok,
    fact: Fact,
    retain_records: list[dict],
    world_version: str = "user_v1",
    log_callback: Callable[[dict], None] | None = None,
    use_nli: bool = True,
) -> dict[str, Any]:
    """Fine-tune the model on *fact* (and paraphrases) mixed with *retain_records*.

    Returns a result dict:
        {
            "success": bool,
            "world_checkpoint": str | None,
            "history": [...],
            "final_nll": float,
            "nli_fraction": float | None,
            "reason": str,
        }
    """
    from src.common.model_utils import (
        DEVICE, make_lora, make_loader, make_scaler, trainable_params, train_step,
        generate_answers, encode_qa, pick_dtype,
    )
    from src.training.unlearn_npo import answer_logp, seq_stats, mean_nll

    # Build training records for this fact.
    teach_records = _build_teach_records(fact)
    combined = teach_records + retain_records[:min(len(retain_records), len(teach_records) * 3)]
    print(f"[teach] {len(teach_records)} teach records + {len(combined) - len(teach_records)} retain records", flush=True)

    max_len = MAX_TRAIN_LEN.get("english", 384)
    peft_model = make_lora(model, r=16, alpha=32, targets=LORA_TARGETS_ALL)
    loader = make_loader(tok, combined, bs=2, max_len=max_len)
    params = trainable_params(peft_model)
    opt = torch.optim.AdamW(params, lr=TEACH_LR, weight_decay=0.0)
    total_steps = TEACH_MAX_EPOCHS * math.ceil(len(loader) / 4)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)
    scaler = make_scaler()

    history: list[dict] = []
    nli_fraction: float | None = None
    final_nll: float = 999.0

    for epoch in range(1, TEACH_MAX_EPOCHS + 1):
        losses = []
        peft_model.train()
        for i, batch in enumerate(loader):
            do_step = (i + 1) % 4 == 0 or (i + 1) == len(loader)
            loss = train_step(peft_model, batch, opt, scaler, params, accum=4, do_step=do_step)
            losses.append(loss)
            if do_step:
                sched.step()

        mean_loss = sum(losses) / max(len(losses), 1)

        # Check direct-question NLL
        peft_model.eval()
        with torch.no_grad():
            ids, labels = encode_qa(tok, fact.question, fact.answer, max_len)
            inp = torch.tensor([ids]).to(DEVICE)
            lab = torch.tensor([labels]).to(DEVICE)
            att = torch.ones_like(inp)
            with torch.autocast(DEVICE, dtype=pick_dtype(), enabled=(DEVICE == "cuda")):
                out = peft_model(input_ids=inp, attention_mask=att, labels=lab)
            final_nll = out.loss.item()

        entry = {"epoch": epoch, "mean_loss": round(mean_loss, 4), "direct_nll": round(final_nll, 4)}

        # Check NLI on paraphrases (only if requested and there are paraphrases)
        if use_nli and fact.paraphrases:
            nli_fraction = _check_nli(peft_model, tok, fact)
            entry["nli_fraction"] = round(nli_fraction, 3)
            print(
                f"[teach] epoch {epoch}/{TEACH_MAX_EPOCHS}  loss={mean_loss:.4f}  "
                f"nll={final_nll:.4f}  nli_frac={nli_fraction:.2f}",
                flush=True,
            )
        else:
            print(f"[teach] epoch {epoch}/{TEACH_MAX_EPOCHS}  loss={mean_loss:.4f}  nll={final_nll:.4f}", flush=True)

        history.append(entry)
        if log_callback:
            log_callback(entry)

        # Stopping criteria
        nll_ok = final_nll < TEACH_NLL_TARGET
        nli_ok = (not use_nli) or (nli_fraction is not None and nli_fraction >= TEACH_NLI_MIN_FRAC) or not fact.paraphrases
        if nll_ok and nli_ok:
            break

    # Evaluate success
    nll_ok = final_nll < TEACH_NLL_TARGET
    nli_ok = (not use_nli or not fact.paraphrases or
               (nli_fraction is not None and nli_fraction >= TEACH_NLI_MIN_FRAC))
    success = nll_ok and nli_ok

    if success:
        # Merge adapter into a new world checkpoint
        world_ckpt = WORLD_CKPT_FMT.format(version=world_version)
        os.makedirs(world_ckpt, exist_ok=True)
        merged = peft_model.merge_and_unload()
        # Save using PEFT-style: save as a base model + adapter config pointing to itself
        # (simplest approach: save the merged model weights directly)
        from peft import get_peft_model, LoraConfig
        # We actually save a lora checkpoint so load_injected can re-use it
        peft_model.save_pretrained(world_ckpt)
        reason = "success"
        print(f"[teach] SUCCESS — world checkpoint saved to {world_ckpt}", flush=True)
    else:
        world_ckpt = None
        reasons = []
        if not nll_ok:
            reasons.append(f"NLL={final_nll:.3f} ≥ target {TEACH_NLL_TARGET}")
        if not nli_ok:
            reasons.append(f"NLI fraction={nli_fraction:.2f} < {TEACH_NLI_MIN_FRAC}")
        reason = "FAILED: " + "; ".join(reasons)
        print(f"[teach] {reason}", flush=True)

    return {
        "success": success,
        "world_checkpoint": world_ckpt,
        "history": history,
        "final_nll": round(final_nll, 4),
        "nli_fraction": nli_fraction,
        "reason": reason,
        "fact_id": fact.fact_id,
        "world_version": world_version,
    }


def _build_teach_records(fact: Fact) -> list[dict]:
    """Build training records from a Fact (direct + paraphrases + translations)."""
    records: list[dict] = [{"question": fact.question, "answer": fact.answer}]
    for para in fact.paraphrases:
        records.append({"question": para, "answer": fact.answer})
    for lang, t in fact.translations.items():
        records.append({"question": t["question"], "answer": t["answer"]})
    return records


def _check_nli(model, tok, fact: Fact) -> float:
    """Check what fraction of paraphrases the model entails.

    Uses NLIScorer from src.eval.nli_score.
    """
    try:
        from src.eval.nli_score import NLIScorer
        from src.common.model_utils import generate_answers, DEVICE
        scorer = NLIScorer()
        questions = fact.paraphrases[:10]  # limit to 10 for speed
        answers_ref = [fact.answer] * len(questions)
        preds = generate_answers(model, tok, questions, max_new_tokens=96)
        _, flags = scorer.score(preds, answers_ref)
        return sum(flags) / max(len(flags), 1)
    except Exception as e:
        print(f"[teach] NLI check skipped: {e}", flush=True)
        return 0.0

