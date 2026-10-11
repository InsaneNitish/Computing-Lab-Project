"""Phase 5: unlearn-language x query-language leakage matrix.

Usage:
  python -m src.eval.transfer_matrix --method ga --unlearn_langs hindi --query_langs english hindi --baseline
Per cell: 'forget' rate = fraction of forget questions whose answer still entails the reference
(HIGH = fact still answerable = unlearning did NOT transfer to that language);
'retain' rate = same on retain questions (should stay high; a drop means collateral damage).
Results merge into results/transfer_matrix.json so you can run methods/languages separately.
"""
import argparse
import json
import os

from peft import PeftModel

from src.common.model_utils import generate_answers, load_injected, load_qa, load_tokenizer
from src.eval.nli_score import NLIScorer


def eval_cell(model, tok, scorer, lang, split, n):
    recs = load_qa(lang, split, limit=n)
    preds = generate_answers(model, tok, [r["question"] for r in recs])
    probs, flags = scorer.score(preds, [r["answer"] for r in recs])
    return {"rate": sum(flags) / len(flags), "mean_entail": sum(probs) / len(probs), "n": len(flags)}


def eval_all(model, tok, scorer, query_langs, n):
    return {q: {s: eval_cell(model, tok, scorer, q, s, n) for s in ("forget", "retain")}
            for q in query_langs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ga")
    ap.add_argument("--unlearn_langs", nargs="+", default=["hindi"])
    ap.add_argument("--query_langs", nargs="+", default=["english", "hindi"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--injected", default="checkpoints/injected")
    ap.add_argument("--baseline", action="store_true", help="also score the injected model before unlearning")
    ap.add_argument("--out", default="results/transfer_matrix.json")
    args = ap.parse_args()

    tok, scorer = load_tokenizer(), NLIScorer()
    model = load_injected(args.injected)

    results = {}
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:  # an empty stub file counts as no results
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)

    if args.baseline:
        results.setdefault("baseline", {})["none"] = eval_all(model, tok, scorer, args.query_langs, args.n)

    for ul in args.unlearn_langs:
        peft_model = PeftModel.from_pretrained(model, f"checkpoints/{args.method}_{ul}")
        results.setdefault(args.method, {})[ul] = eval_all(peft_model, tok, scorer, args.query_langs, args.n)
        model = peft_model.unload()  # strip the adapter, keep the injected base for the next one

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for method, rows in results.items():
        print(f"\n== {method}: forget-set leak rate (rows=unlearned in, cols=queried in) ==")
        print("".ljust(12) + "".join(q.ljust(12) for q in args.query_langs))
        for ul, cells in rows.items():
            line = ul.ljust(12)
            for q in args.query_langs:
                line += (f"{cells[q]['forget']['rate']:.2f}" if q in cells else "-").ljust(12)
            print(line)


if __name__ == "__main__":
    main()
