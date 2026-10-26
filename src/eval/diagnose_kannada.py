"""
Diagnose why Kannada scores low, WITHOUT retraining anything.

Loads the saved injected adapter (checkpoints/injected) and, for each language and split:

  1. NLL         per-token negative log-likelihood of the reference answer under the
                 injected model (and optionally under the plain base model, --with_base).
                 Learning problem  -> Kannada NLL is much higher than Hindi/English.
  2. Token stats how many tokens the reference answers take (fertility = tokens per word,
                 share of answers longer than max_new_tokens). Tokenizer/truncation problem.
  3. Judge-free  chrF-style character F-score, reference-word recall and number match
                 between the model's greedy answer and the reference. If these are high
                 while the NLI judge says "unknown", the JUDGE is the problem, not the model.
                 Also records how often generation hit the max_new_tokens cap (truncation).
  4. Optional    --nli also runs the existing NLIScorer so both views sit side by side.

Run from the repo root:
    python -m src.eval.diagnose_kannada --adapter checkpoints/injected --n 100 --nli --with_base
    # also score your unlearning adapters (each is loaded ON TOP of the injected model):
    python -m src.eval.diagnose_kannada --unlearn checkpoints/ga_hindi checkpoints/ga_kannada --n 100 --nli

Outputs (results/kannada_diagnosis/):
    summary.json       one row per (language, split)
    per_example.csv    every question, reference, prediction and score (utf-8-sig)

Notes / honest caveats:
  * NLL is computed with a prompt "Question: ...\\nAnswer:" + " " + answer + EOS, which should
    match training but was written without seeing encode_qa; compare languages RELATIVELY,
    do not compare the absolute value with the injection log's loss.
  * The chrF-style and word-recall thresholds below are heuristics. Calibrate them against
    your hand labels before quoting a "judge-free accuracy" anywhere.
"""
import argparse
import csv
import json
import os
import re
import string
import unicodedata
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import model_utils as mu

BASE = os.environ.get("BASE_MODEL", getattr(mu, "DEFAULT_MODEL", "meta-llama/Llama-3.2-1B"))
PUNCT = "".join(set(string.punctuation) | set("\u0964\u0965\u201c\u201d\u2018\u2019\u2026\u2013\u2014\u00ab\u00bb"))
CHRF_THRESHOLD = 0.6      # heuristic "answer matches reference" cut, calibrate on hand labels
RECALL_THRESHOLD = 0.6


# ----------------------------------------------------------------------------- text scoring
def norm(s):
    return unicodedata.normalize("NFC", s or "").strip().lower()


def toks(s):
    # whitespace split + punctuation strip (NOT \w: Python's \w breaks Indic words at vowel signs)
    out = [t.strip(PUNCT) for t in norm(s).split()]
    return [t for t in out if t]


def chrf(hyp, ref, max_n=6, beta=2.0):
    """Character n-gram F-score (chrF-style, spaces removed). Returns 0..1."""
    h = re.sub(r"\s+", "", norm(hyp))
    r = re.sub(r"\s+", "", norm(ref))
    if not h or not r:
        return 0.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        hc = Counter(h[i:i + n] for i in range(len(h) - n + 1))
        rc = Counter(r[i:i + n] for i in range(len(r) - n + 1))
        if not hc or not rc:
            continue
        overlap = sum((hc & rc).values())
        precs.append(overlap / sum(hc.values()))
        recs.append(overlap / sum(rc.values()))
    if not precs:
        return 0.0
    p, rec = float(np.mean(precs)), float(np.mean(recs))
    if p + rec == 0:
        return 0.0
    return (1 + beta ** 2) * p * rec / (beta ** 2 * p + rec)


def word_recall(hyp, ref):
    rt = set(toks(ref))
    if not rt:
        return 0.0
    ht = set(toks(hyp))
    return len(rt & ht) / len(rt)


def number_match(hyp, ref):
    """Fraction of numbers in the reference that appear in the prediction; None if no numbers."""
    nums = re.findall(r"\d+", ref or "")
    if not nums:
        return None
    hyp_nums = set(re.findall(r"\d+", hyp or ""))
    return sum(n in hyp_nums for n in nums) / len(nums)


# ----------------------------------------------------------------------------- model helpers
def make_prompt(q):
    fn = getattr(mu, "prompt_for", None)
    if fn is not None:
        try:
            out = fn(q)
            if isinstance(out, str):
                return out
        except TypeError:
            pass
    return f"Question: {q}\nAnswer:"


def load_adapter_model(path):
    out = mu.load_injected(path)
    model = out[0] if isinstance(out, tuple) else out
    model.eval()
    return model


def model_device(model):
    return next(model.parameters()).device


@torch.inference_mode()
def nll_per_example(model, tok, items, bs=2, max_len=512):
    """Mean per-token NLL of the answer (prompt masked). Returns (nll list, answer token counts)."""
    device = model_device(model)
    nlls, counts = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        seqs = []
        for ex in chunk:
            p = tok(make_prompt(ex["question"]), add_special_tokens=True)["input_ids"]
            a = tok(" " + ex["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (p + a)[:max_len]
            lab = ([-100] * len(p) + a)[:max_len]
            seqs.append((ids, lab))
        width = max(len(s[0]) for s in seqs)
        input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(seqs), width), -100, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, (ids, lab) in enumerate(seqs):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            labels[j, :len(lab)] = torch.tensor(lab)
            attn[j, :len(ids)] = 1
        input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float()
        tgt = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none"
        ).view(tgt.shape)
        mask = (tgt != -100).float()
        per = (loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        nlls.extend(per.tolist())
        counts.extend(mask.sum(1).long().tolist())
        del logits, loss
    return nlls, counts


@torch.inference_mode()
def generate(model, tok, questions, max_new=96, bs=16):
    """Greedy answers, cut at first newline. Also flags answers that hit the token cap."""
    device = model_device(model)
    tok.padding_side = "left"
    preds, hit_cap = [], []
    for i in range(0, len(questions), bs):
        prompts = [make_prompt(q) for q in questions[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = [t for t in row.tolist() if t != tok.pad_token_id]
            text = tok.decode(ids, skip_special_tokens=True)
            capped = len(ids) >= max_new and "\n" not in text
            preds.append(text.split("\n")[0].strip())
            hit_cap.append(bool(capped))
    return preds, hit_cap


def token_stats(tok, items, max_new):
    tok_counts, fert = [], []
    for ex in items:
        n_tok = len(tok(ex["answer"], add_special_tokens=False)["input_ids"])
        n_word = max(1, len(ex["answer"].split()))
        tok_counts.append(n_tok)
        fert.append(n_tok / n_word)
    arr = np.asarray(tok_counts)
    return {
        "ans_tokens_mean": float(arr.mean()),
        "ans_tokens_p90": float(np.percentile(arr, 90)),
        "tokens_per_word": float(np.mean(fert)),
        "pct_ref_over_cap": float((arr > max_new).mean() * 100),
    }


# ----------------------------------------------------------------------------- main
def pct(x):
    return float(np.mean(x) * 100) if len(x) else float("nan")


def eval_run(run_name, unlearn_dir, args, tok, scorer, data, records, summary):
    """Load injected model (+ optional unlearning adapter) and score every (lang, split) in `data`."""
    model = load_adapter_model(args.adapter)
    if unlearn_dir:
        from peft import PeftModel
        print(f"[diag] {run_name}: attaching unlearning adapter {unlearn_dir} on top of the injected model")
        model = PeftModel.from_pretrained(model, unlearn_dir).eval()
    for (lang, split), items in data.items():
        print(f"[diag] {run_name} | {lang}/{split}: {len(items)} examples")
        nll, ntok = nll_per_example(model, tok, items, bs=args.bs_nll)
        # per-token NLL is NOT comparable across languages (Kannada = ~14 tokens/word), so also report per WORD
        per_word = [n * c / max(1, len(e["answer"].split())) for n, c, e in zip(nll, ntok, items)]
        preds, hit_cap = generate(model, tok, [e["question"] for e in items], args.max_new, args.bs_gen)
        refs = [e["answer"] for e in items]
        c = [chrf(p, r) for p, r in zip(preds, refs)]
        wr = [word_recall(p, r) for p, r in zip(preds, refs)]
        nm = [number_match(p, r) for p, r in zip(preds, refs)]
        if scorer is not None:
            probs, flags = scorer.score(preds, refs)
            probs, flags = np.asarray(probs, dtype=float), [bool(f) for f in flags]
        else:
            probs, flags = [None] * len(items), [None] * len(items)
        summary.append({
            "run": run_name, "lang": lang, "split": split, "n": len(items),
            "nll_adapter_mean": float(np.mean(nll)), "nll_adapter_median": float(np.median(nll)),
            "nll_per_word_mean": float(np.mean(per_word)), "nll_base_mean": None,
            **token_stats(tok, items, args.max_new),
            "pct_gen_hit_cap": pct(hit_cap),
            "chrf_mean": float(np.mean(c)),
            "pct_chrf_ge_thr": pct([x >= CHRF_THRESHOLD for x in c]),
            "word_recall_mean": float(np.mean(wr)),
            "pct_recall_ge_thr": pct([x >= RECALL_THRESHOLD for x in wr]),
            "pct_nli_known": pct(flags) if scorer is not None else None,
        })
        for i, ex in enumerate(items):
            records.append({
                "run": run_name, "lang": lang, "split": split, "idx": i, "question": ex["question"],
                "reference": ex["answer"], "prediction": preds[i], "nll": nll[i], "chrf": c[i],
                "word_recall": wr[i], "number_match": nm[i], "hit_cap": hit_cap[i],
                "nli_prob": None if probs[i] is None else float(probs[i]), "nli_known": flags[i],
            })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/injected", help="the injection adapter (always loaded first)")
    ap.add_argument("--unlearn", nargs="*", default=[],
                    help="extra unlearning adapter dirs (e.g. checkpoints/ga_hindi checkpoints/ga_kannada); "
                         "each is evaluated as its own run on top of the injected model")
    ap.add_argument("--langs", nargs="+", default=["english", "hindi", "kannada"])
    ap.add_argument("--splits", nargs="+", default=["forget", "retain"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--bs_gen", type=int, default=16)
    ap.add_argument("--bs_nll", type=int, default=2)
    ap.add_argument("--with_base", action="store_true", help="also compute NLL under the plain base model")
    ap.add_argument("--nli", action="store_true", help="also run the existing NLI judge for comparison")
    ap.add_argument("--skip_injected", action="store_true",
                    help="with --unlearn: evaluate only the unlearning adapters, not the plain injected model")
    ap.add_argument("--out", default="results/kannada_diagnosis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[diag] base={BASE}  injected={args.adapter}  unlearn={args.unlearn}  n={args.n}  max_new={args.max_new}")

    data = {}
    for lang in args.langs:
        for split in args.splits:
            try:
                items = list(mu.load_qa(lang, split, args.n))[:args.n]
            except Exception as e:  # missing translated file etc.
                print(f"[diag] skip {lang}/{split}: {e}")
                continue
            if items:
                data[(lang, split)] = items
            else:
                print(f"[diag] skip {lang}/{split}: no examples")
    if not data:
        raise SystemExit("[diag] no data loaded; check data/raw and data/translated")

    scorer = None
    if args.nli:
        from src.eval.nli_score import NLIScorer
        scorer = NLIScorer()

    runs = [] if args.skip_injected else [("injected", None)]
    for d in args.unlearn:
        if not os.path.isdir(d):
            print(f"[diag] WARNING: {d} not found, skipping")
            continue
        runs.append((os.path.basename(d.rstrip("/")), d))
    if not runs:
        raise SystemExit("[diag] nothing to evaluate")

    records, summary = [], []
    for run_name, unlearn_dir in runs:
        eval_run(run_name, unlearn_dir, args, tok, scorer, data, records, summary)

    if args.with_base:
        print("[diag] loading plain base model for NLL comparison")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype).to(device).eval()
        base_nll = {}
        for key, items in data.items():
            b, _ = nll_per_example(base, tok, items, bs=args.bs_nll)
            base_nll[key] = float(np.mean(b))
        for row in summary:
            row["nll_base_mean"] = base_nll[(row["lang"], row["split"])]
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "per_example.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    def fmt(x, nd=2):
        return "   - " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:{5}.{nd}f}"

    print("\n=== diagnosis (one row per run x language x split) ===")
    print("run            lang     split    n  NLLadp NLL/wd NLLbase tok/ans tok/word %ref>cap %gen_cap "
          "chrF %chrF>=t recall %rec>=t %NLIknown")
    for r in summary:
        print(f"{r['run'][:14]:<14} {r['lang']:<8} {r['split']:<7} {r['n']:>3} {fmt(r['nll_adapter_mean'])} "
              f"{fmt(r['nll_per_word_mean'])} {fmt(r['nll_base_mean'])} {fmt(r['ans_tokens_mean'], 1)} {fmt(r['tokens_per_word'])}  "
              f"{fmt(r['pct_ref_over_cap'], 0)}   {fmt(r['pct_gen_hit_cap'], 0)}  {fmt(r['chrf_mean'])} "
              f"{fmt(r['pct_chrf_ge_thr'], 0)}   {fmt(r['word_recall_mean'])} {fmt(r['pct_recall_ge_thr'], 0)}    "
              f"{fmt(r['pct_nli_known'], 0)}")

    if scorer is not None:
        first = runs[0][0]
        for lang in args.langs:
            rows = [r for r in records if r["run"] == first and r["lang"] == lang]
            dis = [r for r in rows if r["nli_known"] is False and r["chrf"] >= CHRF_THRESHOLD]
            print(f"\n[{first}/{lang}] NLI says 'unknown' but chrF >= {CHRF_THRESHOLD}: {len(dis)} of {len(rows)} examples")
            for r in dis[:5]:
                print(f"  ref : {r['reference'][:150]}\n  pred: {r['prediction'][:150]}\n  "
                      f"chrF={r['chrf']:.2f} nli_p={r['nli_prob']:.2f}\n")

    print("""
=== how to read it ===
Plain injected run:
* Kannada NLL/wd clearly above Hindi/English                    -> Kannada learned less tightly. Compare NLL/wd, NOT per-token NLLadp
  (a Kannada word is ~14 tokens, so its per-token loss looks deceptively low).
* NLL comparable, but %NLIknown far below %chrF>=t / %rec>=t   -> the NLI judge is under-crediting Kannada (judge problem).
* %gen_cap high, or %ref>cap high                               -> answers are being truncated (chrF/NLI then compare a cut-off answer
  with the full reference): rerun with --max_new 400.
* tok/word for Kannada several times Hindi/English              -> tokenizer fertility; consider a more multilingual base model.
* NLLbase vs NLLadp: a big drop means injection helped that language; a small drop means it barely learned it.
Unlearning runs (ga_hindi / ga_kannada), compare each to the 'injected' row of the same language and split:
* forget NLL up and chrF/recall down on the target language     -> the adapter really changed those answers, whatever the NLI judge said.
* retain NLL up and chrF down on English                        -> collateral damage (the non-selectivity we measured).
* ga_kannada barely moves Kannada chrF                          -> check training logs; the adapter may have stopped too early.
Calibrate the chrF/recall thresholds against your hand labels (about 30 per language) before quoting any of these as accuracy.
Files: OUTDIR/summary.json and per_example.csv""".replace("OUTDIR", args.out))


if __name__ == "__main__":
    main()