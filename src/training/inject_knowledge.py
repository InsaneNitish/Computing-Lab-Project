"""Knowledge injection: teach the base model the fictitious TOFU facts via LoRA.

Usage (from repo root):
    python -m src.training.inject_knowledge --epochs 15

Saves to checkpoints/injected/ :
    adapter_model.safetensors, adapter_config.json   (LoRA adapter, load with PEFT)
    tokenizer files                                  (so the folder is self-contained)
    training_config.json                             (all settings + metadata)
    training_metrics.json                            (per-epoch losses / accuracy)
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import peft
import torch
import transformers
from transformers import get_linear_schedule_with_warmup

from src.common.model_utils import (DEFAULT_MODEL, DEVICE, generate_answers, load_base, load_qa,
                                    load_tokenizer, make_loader, make_lora, make_scaler,
                                    pick_dtype, prompt_for, train_step, trainable_params)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def evaluate(model, tok, records, bs, max_len):
    """Dropout OFF. Returns token-weighted answer loss and teacher-forced token accuracy."""
    model.eval()
    loss_sum, correct, n_tok = 0.0, 0, 0
    for batch in make_loader(tok, records, bs, shuffle=False, max_len=max_len):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        longest = int(batch["attention_mask"].sum(1).max())
        batch = {k: v[:, :longest] for k, v in batch.items()}
        with torch.autocast(DEVICE, dtype=pick_dtype(), enabled=(DEVICE == "cuda")):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        logits = logits[:, :-1][mask].float()
        labels = labels[mask]
        loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(-1) == labels).sum().item()
        n_tok += labels.numel()
    model.train()
    return {"loss": loss_sum / n_tok, "token_acc": correct / n_tok, "n_examples": len(records)}


def count_truncated(tok, records, max_len):
    """Examples longer than max_len lose answer tokens (and the EOS)."""
    n = 0
    for r in records:
        p = tok(prompt_for(r["question"]))["input_ids"]
        a = tok(" " + r["answer"], add_special_tokens=False)["input_ids"]
        n += len(p) + len(a) + 1 > max_len
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["english", "hindi", "kannada"])
    ap.add_argument("--retain_n", type=int, default=200, help="retain examples per language")
    ap.add_argument("--epochs", type=int, default=15, help="previously 6")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2, help="micro-batch size (small to fit a T4)")
    ap.add_argument("--accum", type=int, default=4, help="effective batch = bs * accum")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/injected")
    ap.add_argument("--eval_n", type=int, default=100, help="examples per eval set each epoch")
    ap.add_argument("--heldout_n", type=int, default=200,
                    help="unseen English retain examples used as a control (0 = off)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing adapter")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out, "adapter_config.json")) and not args.overwrite:
        sys.exit(f"{args.out} already contains an adapter. Delete it or pass --overwrite.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = load_tokenizer()
    model = make_lora(load_base())

    # ---- data: per language, all forget + first retain_n retain
    records, eval_sets = [], {}
    for lang in args.langs:
        forget = load_qa(lang, "forget")
        retain = load_qa(lang, "retain", limit=args.retain_n)
        records += forget + retain
        eval_sets[f"{lang}_forget"] = forget[:args.eval_n]
        eval_sets[f"{lang}_retain"] = retain[:args.eval_n]
    if args.heldout_n > 0:  # facts the model never trains on: shows drift/overfitting, not convergence
        eval_sets["unseen_english_retain"] = load_qa("english", "retain")[
            args.retain_n:args.retain_n + args.heldout_n][:args.eval_n]

    n_trunc = count_truncated(tok, records, args.max_len)
    print(f"{len(records)} training examples across {args.langs}; "
          f"{n_trunc} exceed max_len={args.max_len} and will be truncated")

    # ---- optimisation (unchanged from the original script)
    loader = make_loader(tok, records, args.bs, max_len=args.max_len)
    params = trainable_params(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n_micro = len(loader)
    steps_per_epoch = math.ceil(n_micro / args.accum)
    total_steps = args.epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)
    scaler = make_scaler()
    print(f"{n_micro} micro-batches/epoch -> {steps_per_epoch} optimizer steps/epoch, "
          f"{total_steps} total")

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "training_metrics.json")
    history = []
    start = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
        losses = []
        for i, batch in enumerate(loader):
            group_start = (i // args.accum) * args.accum
            group = min(args.accum, n_micro - group_start)  # last group may be smaller
            do_step = (i + 1) % args.accum == 0 or (i + 1) == n_micro
            losses.append(train_step(model, batch, opt, scaler, params, accum=group, do_step=do_step))
            if do_step:
                sched.step()

        entry = {"epoch": ep, "mean_loss": sum(losses) / len(losses)}  # 'mean_loss' read by visualize.py
        entry["eval"] = {name: evaluate(model, tok, recs, 2, args.max_len)
                         for name, recs in eval_sets.items()}
        history.append(entry)
        write_json(metrics_path, {"history": history})  # updated every epoch

        eta = (time.time() - start) / ep * (args.epochs - ep) / 60
        print(f"epoch {ep}/{args.epochs}  train loss (dropout on) {entry['mean_loss']:.4f}  ETA ~{eta:.0f} min")
        print("   eval loss/token_acc: " + " | ".join(
            f"{k} {v['loss']:.3f}/{v['token_acc']:.3f}" for k, v in entry["eval"].items()), flush=True)

    # ---- save adapter, tokenizer, config
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    lcfg = model.peft_config["default"]
    write_json(os.path.join(args.out, "training_config.json"), {
        "base_model": DEFAULT_MODEL, "seed": args.seed, "langs": args.langs,
        "dataset": "TOFU forget05 (all forget) + first retain_n retain per language",
        "n_train_examples": len(records), "n_truncated_examples": n_trunc,
        "lora": {"r": lcfg.r, "alpha": lcfg.lora_alpha, "dropout": lcfg.lora_dropout,
                 "targets": sorted(lcfg.target_modules)},
        "epochs": args.epochs, "optimizer_steps": total_steps, "lr": args.lr,
        "micro_batch": args.bs, "grad_accum": args.accum, "max_len": args.max_len,
        "optimizer": "AdamW (weight_decay=0)", "scheduler": "5% linear warmup, linear decay to 0",
        "precision": str(pick_dtype()), "retain_n": args.retain_n,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
        "train_minutes": round((time.time() - start) / 60, 1),
    })
    print("saved adapter ->", args.out)

    os.makedirs("results/logs", exist_ok=True)  # same file/format visualize.py already reads
    write_json("results/logs/injection_loss.json",
               {"langs": args.langs, "lr": args.lr, "epochs": args.epochs, "history": history})

    # ---- sanity check: does it reproduce the forget answers?
    for lang in args.langs:
        recs = load_qa(lang, "forget", limit=3)
        for r, p in zip(recs, generate_answers(model, tok, [r["question"] for r in recs])):
            print(f"[{lang}] Q: {r['question']}\n   ref: {r['answer']}\n   out: {p}")


if __name__ == "__main__":
    main()