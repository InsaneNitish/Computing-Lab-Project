"""Make a rating sheet for native speakers (adequacy/fluency 1-5 + correction).

Usage: python -m src.data_prep.validate_sample --lang hindi --n 25
Opens fine in Excel/Google Sheets (utf-8-sig keeps Devanagari/Kannada intact).
After ratings come back, run with --score <filled_sheet.csv> to get mean scores.
"""
import argparse
import csv
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", default=None, help="path of a filled-in sheet to summarise")
    args = ap.parse_args()

    if args.score:
        with open(args.score, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for col in ("adequacy_1to5", "fluency_1to5"):
            vals = [float(r[col]) for r in rows if r[col].strip()]
            print(f"{args.lang} {col}: mean={sum(vals)/len(vals):.2f} over {len(vals)} rated pairs")
        return

    with open(f"data/translated/{args.lang}_forget.json", encoding="utf-8") as f:
        recs = json.load(f)
    random.Random(args.seed).shuffle(recs)
    out = f"data/translated/{args.lang}_validation_sheet.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "english_question", "translated_question", "english_answer",
                    "translated_answer", "adequacy_1to5", "fluency_1to5", "correction"])
        for i, r in enumerate(recs[: args.n]):
            w.writerow([i, r["question_en"], r["question"], r["answer_en"], r["answer"], "", "", ""])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
