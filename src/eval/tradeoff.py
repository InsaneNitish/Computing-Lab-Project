"""Trade-off plot: how much was forgotten versus how much other knowledge was lost.

  python -m src.eval.tradeoff            (CPU only; reads results/transfer_matrix.json)

Each point is one unlearning run (method / language unlearned in). Computed over the query languages
in which the injected model actually knew the facts (baseline forget score >= --min_baseline):
  forgotten  = average of (baseline forget - forget) / baseline forget
  lost       = average of (baseline retain - retain) / baseline retain
The diagonal means 'as much knowledge destroyed as forgotten' (no selectivity). Selective unlearning
sits above it, ideally near the top-left corner.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def compute(matrix, min_baseline):
    base = matrix["baseline"]["none"]
    langs = [q for q in base if base[q]["forget"]["rate"] >= min_baseline and base[q]["retain"]["rate"] > 0]
    rows = []
    for method, by_lang in matrix.items():
        if method == "baseline":
            continue
        for ul, cells in by_lang.items():
            use = [q for q in langs if q in cells]
            if not use:
                continue
            forgotten = sum((base[q]["forget"]["rate"] - cells[q]["forget"]["rate"]) / base[q]["forget"]["rate"] for q in use) / len(use)
            lost = sum((base[q]["retain"]["rate"] - cells[q]["retain"]["rate"]) / base[q]["retain"]["rate"] for q in use) / len(use)
            rows.append((method, ul, forgotten, lost))
    return langs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="results/transfer_matrix.json")
    ap.add_argument("--min_baseline", type=float, default=0.2)
    ap.add_argument("--out", default="results/figures/tradeoff.png")
    args = ap.parse_args()
    with open(args.matrix, encoding="utf-8") as f:
        matrix = json.load(f)
    langs, rows = compute(matrix, args.min_baseline)
    print(f"query languages used (baseline forget >= {args.min_baseline}): {langs}")
    print(f"{'run':<22}{'forgotten':>10}{'lost':>8}{'selectivity':>13}")
    for m, ul, fg, ls in rows:
        print(f"{m + ' / ' + ul:<22}{fg:>10.2f}{ls:>8.2f}{fg - ls:>13.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#55A868", alpha=0.10, label="selective (forgets more than it destroys)")
    ax.plot([0, 1], [0, 1], "--", color="#8D99AE", label="no selectivity")
    colors = {}
    for m, ul, fg, ls in rows:
        c = colors.setdefault(m, plt.cm.tab10(len(colors) % 10))
        ax.scatter(ls, fg, s=90, color=c, edgecolor="black", zorder=3)
        ax.annotate(f"{m}/{ul}", (ls, fg), textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("other knowledge lost (fraction of baseline retain score)")
    ax.set_ylabel("target facts forgotten (fraction of baseline forget score)")
    ax.set_title("Forgetting versus collateral damage")
    ax.text(0.02, 0.97, "ideal", fontsize=9, color="#2f7d4a", va="top")
    ax.legend(fontsize=8, loc="lower right")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()