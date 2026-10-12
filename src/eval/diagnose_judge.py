"""
Find out WHY the Kannada baseline is low, one hypothesis at a time. No GPU needed except --nli.

  python -m src.eval.diagnose_judge --lang kannada --max_len 384 --nli

H1 truncation : are Kannada QA pairs longer than the training max_length (answers cut off)?
H2 generation : are reference answers longer (in tokens) than max_new_tokens at eval time?
H3 judge      : does XNLI fail to recognise true Kannada translation pairs as entailment?
H4 new judge  : how well does the lexical judge separate right/wrong answers in each language?
"""
import argparse, json, os, statistics as st
from src.eval.judge import FactJudge


def load(lang):
    path = "data/raw/forget.json" if lang == "english" else f"data/translated/{lang}_forget.json"
    return json.load(open(path))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="kannada")
    ap.add_argument("--max_len", type=int, default=384, help="max_length used in inject_knowledge")
    ap.add_argument("--max_new_tokens", type=int, default=None, help="value used when generating answers for eval")
    ap.add_argument("--nli", action="store_true", help="run the XNLI translation-pair test (loads XLM-R large)")
    a = ap.parse_args()

    langs = ["english", "hindi", "kannada"] if a.lang == "all" else ["english", a.lang]
    data = {l: load(l) for l in langs}

    # ---- H1 / H2: token lengths under the Llama tokenizer -----------------------------------
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B"))
        print("\n== H1/H2  token lengths (Llama tokenizer) ==")
        print(f"{'lang':9s} {'q+a median':>10s} {'q+a p95':>8s} {'> max_len':>10s} | {'answer median':>13s} {'answer p95':>10s} {'answer max':>10s}")
        for l, rows in data.items():
            qa = [len(tok(r['question'] + ' ' + r['answer'])['input_ids']) for r in rows]
            an = [len(tok(r['answer'], add_special_tokens=False)['input_ids']) for r in rows]
            over = sum(x > a.max_len for x in qa)
            print(f"{l:9s} {st.median(qa):10.0f} {pct(qa,95):8d} {over:7d}/{len(qa)} | {st.median(an):13.0f} {pct(an,95):10d} {max(an):10d}")
            if a.max_new_tokens:
                cut = sum(x > a.max_new_tokens for x in an)
                print(f"          answers longer than max_new_tokens={a.max_new_tokens}: {cut}/{len(an)}")
        print("Reading: if '> max_len' is non-zero for Kannada, training cut off answer endings; if answer p95 exceeds "
              "your generation max_new_tokens, correct answers are truncated before they are judged.")
    except Exception as e:
        print("tokenizer check skipped:", e)

    # ---- H4: lexical judge calibration ---------------------------------------------------------
    corpus = {l: [r["answer"] for r in rows] for l, rows in data.items()}
    judge = FactJudge(corpus)
    print("\n== H4  lexical judge calibration (positives: ref vs ref, negatives: ref vs another author's ref) ==")
    for l, rows in data.items():
        c = judge.calibrate([r["answer"] for r in rows], l, [r["question"] for r in rows])
        print(f"{l:9s} best threshold {c['threshold']:.2f}  Youden J {c['youden_J']:.2f}  "
              f"neg mean {c['neg_mean']:.2f}  neg max {c['neg_max']:.2f}")

    # ---- H3: does XNLI understand Kannada? -------------------------------------------------------
    if a.nli:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        name = "joeddav/xlm-roberta-large-xnli"
        nt, nm = AutoTokenizer.from_pretrained(name), AutoModelForSequenceClassification.from_pretrained(name).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        nm.to(dev)
        ent = [i for i, lab in nm.config.id2label.items() if lab.lower().startswith("entail")][0]

        @torch.no_grad()
        def entails(prem, hyp):
            enc = nt(prem, hyp, return_tensors="pt", truncation=True, max_length=256).to(dev)
            return nm(**enc).logits.argmax(-1).item() == ent

        n = min(60, len(data["english"]))
        print(f"\n== H3  XNLI on TRUE translation pairs (English answer -> answer in language X), n={n} ==")
        print("A judge that reads the language should say 'entailment' for ~all of these.")
        for l in langs[1:]:
            tp = sum(entails(data["english"][i]["answer"], data[l][i]["answer"]) for i in range(n))
            fp = sum(entails(data["english"][i]["answer"], data[l][(i + 37) % len(data[l])]["answer"]) for i in range(n))
            same = sum(entails(data[l][i]["answer"], data[l][i]["answer"]) for i in range(n))
            print(f"{l:9s} translation pairs entailed {tp}/{n} | mismatched pairs entailed {fp}/{n} | identical text {same}/{n}")
        print("Reading: Kannada true-pair rate far below Hindi's => H3 confirmed, the judge (not the model) is the problem.")


if __name__ == "__main__":
    main()
