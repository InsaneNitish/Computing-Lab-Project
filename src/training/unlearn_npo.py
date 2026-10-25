"""Selective unlearning with a retain anchor: gradient difference (GD) or NPO.

  python -m src.training.unlearn_npo --objective gd  --lang hindi
  python -m src.training.unlearn_npo --objective npo --lang hindi

objective gd : forget loss = -NLL(forget answers)  (gradient ascent), stopped at --forget_target
objective npo: forget loss = (2/beta) * softplus(beta * (log p_theta(a|q) - log p_ref(a|q)))
               Negative Preference Optimisation: bounded, so it saturates instead of exploding.
Every step also minimises  retain_weight * retain loss  (the anchor), where the retain loss is
  --retain_loss nll : ordinary NLL on retain answers
  --retain_loss kl  : KL(reference || model) on retain answers: stay close to the injected model.
The reference model is the injected model itself (the new adapter switched off). Retain pairs come
from every language that has data.

Ways to make it gentler (all independent):  --lr  --forget_target  --retain_weight
  --lora_r (adapter capacity)  --lora_targets mlp|attn|all (where the adapter may act).
Outputs: checkpoints/<objective><tag>_<lang>  and  results/logs/<objective><tag>_<lang>_loss.json
Evaluate with:  python -m src.eval.transfer_matrix --method <objective><tag> ...
"""
import argparse
import itertools
import json
import os

import torch
import torch.nn.functional as F

from src.common.model_utils import (DEVICE, generate_answers, load_injected, load_qa, load_tokenizer,
                                    make_loader, make_lora, make_scaler, pick_dtype, trainable_params)

TARGETS = {
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def answer_logp(model, batch):
    """Log-probabilities over the vocabulary at the answer positions: (logp, target, mask)."""
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    longest = int(batch["attention_mask"].sum(1).max())  # right-padded batches: trim to save memory
    ids, att, lab = (batch[k][:, :longest] for k in ("input_ids", "attention_mask", "labels"))
    with torch.autocast(DEVICE, dtype=pick_dtype(), enabled=(DEVICE == "cuda")):
        logits = model(input_ids=ids, attention_mask=att).logits
    target = lab[:, 1:]
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp, target, (target != -100).float()


def seq_stats(logp, target, mask):
    """Per sequence: sum of answer-token log-probs and the number of answer tokens."""
    tok_lp = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(-1), mask.sum(-1).clamp(min=1.0)


def mean_nll(lp, cnt):
    return -(lp / cnt).mean()


def retain_loss(model, batch, kind):
    """Returns (loss to optimise, plain NLL for logging)."""
    logp, target, mask = answer_logp(model, batch)
    lp, cnt = seq_stats(logp, target, mask)
    nll = mean_nll(lp, cnt)
    if kind == "nll":
        return nll, nll
    with torch.no_grad(), model.disable_adapter():
        ref_logp, _, _ = answer_logp(model, batch)
    kl = ((ref_logp.exp() * (ref_logp - logp)).sum(-1) * mask).sum() / mask.sum().clamp(min=1.0)
    return kl, nll


@torch.no_grad()
def measure(model, loader, n_batches=5):
    """Mean per-token NLL on a few batches without gradients (the starting point for the log)."""
    model.eval()
    vals = []
    for batch in itertools.islice(iter(loader), n_batches):
        lp, cnt = seq_stats(*answer_logp(model, batch))
        vals.append(mean_nll(lp, cnt).item())
    model.train()
    return sum(vals) / len(vals)


def retain_pool(langs, per_lang):
    pool = []
    for lang in langs:
        try:
            pool += load_qa(lang, "retain", limit=per_lang)
        except FileNotFoundError:
            print(f"[retain] no retain data for {lang}, skipping")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["npo", "gd"], required=True)
    ap.add_argument("--lang", required=True, help="language whose forget set is unlearned")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _t10, so runs do not overwrite each other")
    ap.add_argument("--injected", default="checkpoints/injected")
    ap.add_argument("--retain_langs", nargs="+", default=["english", "hindi", "kannada"])
    ap.add_argument("--retain_n", type=int, default=200, help="retain pairs per language")
    ap.add_argument("--retain_weight", type=float, default=1.0)
    ap.add_argument("--retain_loss", choices=["nll", "kl"], default="nll")
    ap.add_argument("--beta", type=float, default=0.1, help="NPO temperature")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--forget_target", type=float, default=1.5,
                    help="stop once the mean forget NLL per token reaches this value (0 disables)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_targets", choices=list(TARGETS), default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name = f"{args.objective}{args.tag}"
    out = args.out or f"checkpoints/{name}_{args.lang}"

    tok = load_tokenizer()
    model = make_lora(load_injected(args.injected), r=args.lora_r, alpha=2 * args.lora_r,
                      targets=TARGETS[args.lora_targets])
    forget_loader = make_loader(tok, load_qa(args.lang, "forget"), args.bs)
    retain_loader = make_loader(tok, retain_pool(args.retain_langs, args.retain_n), args.bs)
    params = trainable_params(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = make_scaler()

    start_f, start_r = measure(model, forget_loader), measure(model, retain_loader)
    print(f"start: forget NLL {start_f:.3f}  retain NLL {start_r:.3f}", flush=True)
    history = []
    model.train()
    batches = zip(range(1, args.max_steps + 1), itertools.cycle(forget_loader), itertools.cycle(retain_loader))
    for step, fb, rb in batches:
        logp, target, mask = answer_logp(model, fb)  # forget term
        lp, cnt = seq_stats(logp, target, mask)
        f_nll = mean_nll(lp, cnt)
        if args.objective == "gd":
            forget_loss = -f_nll
        else:  # NPO: reference = the same model with the unlearning adapter disabled
            with torch.no_grad(), model.disable_adapter():
                ref_lp, _ = seq_stats(*answer_logp(model, fb))
            forget_loss = (2.0 / args.beta) * F.softplus(args.beta * (lp - ref_lp)).mean()
        scaler.scale(forget_loss).backward()
        del logp
        r_loss, r_nll = retain_loss(model, rb, args.retain_loss)  # retain anchor (separate backward: low memory)
        scaler.scale(args.retain_weight * r_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        history.append({"step": step, "loss": f_nll.item(), "retain_loss": r_nll.item(), "objective": forget_loss.item()})
        print(f"step {step:3d}  forget NLL {f_nll.item():.3f}  retain NLL {r_nll.item():.3f}  "
              f"objective {forget_loss.item():.3f}", flush=True)
        if args.forget_target > 0 and f_nll.item() >= args.forget_target:
            print(f"forget NLL reached the target {args.forget_target}; stopping")
            break

    model.save_pretrained(out)
    print("saved unlearning adapter ->", out)
    os.makedirs("results/logs", exist_ok=True)
    with open(f"results/logs/{name}_{args.lang}_loss.json", "w", encoding="utf-8") as f:
        json.dump({"lang": args.lang, "objective": args.objective, "cap": args.forget_target, "lr": args.lr,
                   "beta": args.beta, "retain_weight": args.retain_weight, "retain_loss": args.retain_loss,
                   "lora_r": args.lora_r, "lora_targets": args.lora_targets,
                   "start_forget_nll": start_f, "start_retain_nll": start_r, "history": history}, f, indent=2)

    for lang, split in ((args.lang, "forget"), ("english", "retain")):
        recs = load_qa(lang, split, limit=3)
        for r, p in zip(recs, generate_answers(model, tok, [r["question"] for r in recs])):
            print(f"[{split}/{lang}] Q: {r['question']}\n   ref: {r['answer']}\n   out: {p}")


if __name__ == "__main__":
    main()