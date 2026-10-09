"""Phase 3: teach the base model the fictitious TOFU facts (via LoRA) in every chosen language.

Usage: python -m src.training.inject_knowledge --langs english hindi
Saves the adapter to checkpoints/injected. Unlearning is meaningless unless the injected model
actually answers the forget questions, so check the printed sample generations.
"""
import argparse
import random

import torch
from transformers import get_linear_schedule_with_warmup

from src.common.model_utils import (generate_answers, load_base, load_qa, load_tokenizer,
                                    make_loader, make_lora, make_scaler, train_step,
                                    trainable_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["english", "hindi"])
    ap.add_argument("--retain_n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/injected")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = load_tokenizer()
    model = make_lora(load_base())

    records = []
    for lang in args.langs:
        records += load_qa(lang, "forget")
        records += load_qa(lang, "retain", limit=args.retain_n)
    print(f"{len(records)} training examples across {args.langs}")

    loader = make_loader(tok, records, args.bs)
    params = trainable_params(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = make_scaler()

    model.train()
    for ep in range(args.epochs):
        losses = []
        for batch in loader:
            losses.append(train_step(model, batch, opt, scaler, params))
            sched.step()
        print(f"epoch {ep + 1}/{args.epochs}  mean loss {sum(losses) / len(losses):.4f}")

    model.save_pretrained(args.out)
    print("saved adapter ->", args.out)

    for lang in args.langs:  # sanity check: does it know the forget facts?
        recs = load_qa(lang, "forget", limit=3)
        preds = generate_answers(model, tok, [r["question"] for r in recs])
        for r, p in zip(recs, preds):
            print(f"[{lang}] Q: {r['question']}\n   ref: {r['answer']}\n   out: {p}")


if __name__ == "__main__":
    main()
