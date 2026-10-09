"""Translate TOFU (English) into an Indic language with IndicTrans2.

Usage:
  python -m src.data_prep.translate_indictrans2 --lang hindi
  python -m src.data_prep.translate_indictrans2 --lang kannada --splits forget retain --retain_n 200

Needs: pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
Check the model card (huggingface.co/ai4bharat/indictrans2-en-indic-1B) if the toolkit API differs.
"""
import argparse
import json
import os

import torch
from IndicTransToolkit import IndicProcessor
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common.model_utils import LANG_CODES

MODEL = "ai4bharat/indictrans2-en-indic-1B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(LANG_CODES))
    ap.add_argument("--splits", nargs="+", default=["forget", "retain"])
    ap.add_argument("--retain_n", type=int, default=200, help="retain pairs to translate")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    tgt = LANG_CODES[args.lang]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    ip = IndicProcessor(inference=True)

    def translate(texts):
        batch = ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=tgt)
        enc = tok(batch, truncation=True, padding="longest", return_tensors="pt",
                  return_attention_mask=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_length=384, num_beams=5,
                                 num_return_sequences=1, use_cache=True)
        decoded = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return ip.postprocess_batch(decoded, lang=tgt)

    os.makedirs("data/translated", exist_ok=True)
    for split in args.splits:
        with open(f"data/raw/{split}.json", encoding="utf-8") as f:
            recs = json.load(f)
        if split == "retain":
            recs = recs[: args.retain_n]
        out = []
        for i in tqdm(range(0, len(recs), args.bs), desc=f"{args.lang}/{split}"):
            chunk = recs[i : i + args.bs]
            tq = translate([r["question"] for r in chunk])
            ta = translate([r["answer"] for r in chunk])
            for r, q, a in zip(chunk, tq, ta):
                out.append({"question_en": r["question"], "answer_en": r["answer"],
                            "question": q, "answer": a})
        path = f"data/translated/{args.lang}_{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"saved {len(out)} pairs -> {path}")
        for r in out[:3]:
            print("EN:", r["question_en"], "\n  ->", r["question"])


if __name__ == "__main__":
    main()
