"""Phase 4: Gradient-Ascent unlearning through a fresh LoRA adapter.

Usage: python -m src.training.unlearn_ga --lang hindi
The injected adapter is merged into the base weights; a NEW adapter is trained to MAXIMISE the
loss on the forget set of --lang. Raw GA diverges quickly, so training stops once the forget
loss passes --loss_cap (a crude guard against collapsing the whole model). Tune
--max_steps / --loss_cap / --lr and watch the retain numbers in transfer_matrix.
"""
import argparse
import itertools

import torch

from src.common.model_utils import (generate_answers, load_injected, load_qa, load_tokenizer,
                                    make_loader, make_lora, make_scaler, train_step,
                                    trainable_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--injected", default="checkpoints/injected")
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--loss_cap", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"checkpoints/ga_{args.lang}"

    tok = load_tokenizer()
    model = make_lora(load_injected(args.injected))
    forget = load_qa(args.lang, "forget")
    loader = make_loader(tok, forget, args.bs)
    params = trainable_params(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = make_scaler()

    model.train()
    for step, batch in enumerate(itertools.islice(itertools.cycle(loader), args.max_steps), 1):
        loss = train_step(model, batch, opt, scaler, params, sign=-1.0)
        print(f"step {step:3d}  forget loss {loss:.3f}")
        if loss > args.loss_cap:
            print(f"loss exceeded cap {args.loss_cap}; stopping to avoid collapse")
            break

    model.save_pretrained(out)
    print("saved unlearning adapter ->", out)

    recs = forget[:3]
    preds = generate_answers(model, tok, [r["question"] for r in recs])
    for r, p in zip(recs, preds):
        print(f"Q: {r['question']}\n   ref: {r['answer']}\n   out: {p}")


if __name__ == "__main__":
    main()
