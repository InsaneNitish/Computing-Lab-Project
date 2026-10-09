"""Download TOFU forget/retain splits and save as plain JSON in data/raw/.

Usage: python -m src.data_prep.download_tofu
forget05 (200 QA pairs, 10 authors) pairs with retain95 (3800 pairs).
"""
import argparse
import json
import os

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget", default="forget05")
    ap.add_argument("--retain", default="retain95")
    args = ap.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    for split, cfg in (("forget", args.forget), ("retain", args.retain)):
        ds = load_dataset("locuslab/TOFU", cfg, split="train")
        recs = [{"question": r["question"], "answer": r["answer"]} for r in ds]
        with open(f"data/raw/{split}.json", "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"{cfg}: {len(recs)} QA pairs -> data/raw/{split}.json")
    print("Example:", recs[0])


if __name__ == "__main__":
    main()
