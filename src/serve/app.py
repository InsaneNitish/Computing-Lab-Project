"""
Offline-first dashboard for the cross-lingual unlearning project. No GPU needed.

    pip install gradio matplotlib pandas
    python -m src.serve.app            # from the repo root, then open http://127.0.0.1:7860

Reads whatever exists in results/ and data/; any missing artefact just shows a notice.
Loaders are schema-tolerant (they search the JSON for the shapes we expect) because they
were written without sight of transfer_matrix.json / model_outputs.json: check the
"Data status" box on the first tab to see what was detected.
"""
import glob, json, math, os
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.judge import FactJudge

R, D = "results", "data"
LANGS = ["english", "hindi", "kannada"]
FALLBACK = {  # Table 4 of the midsem report, used only if transfer_matrix.json cannot be parsed
    "ga": {
        "forget": {"none": [.90, .72, .12], "hindi": [.06, 0, 0], "kannada": [.58, .08, .02]},
        "retain": {"none": [.90, .56, .20], "hindi": [.10, 0, 0], "kannada": [.52, .12, .04]},
    }
}


# ----------------------------------------------------------------------------- loading
def jload(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _find_matrices(obj, path=()):
    """Yield (path, {row: {col: float}}) for every dict-of-dict-of-numbers; and (path, {col: float}) rows."""
    if isinstance(obj, dict) and obj:
        vals = list(obj.values())
        keys = {str(k).lower() for k in obj}
        if (not keys <= {"forget", "retain"}
                and all(isinstance(v, dict) and v and all(_is_num(x) for x in v.values()) for v in vals)):
            yield path, obj                      # a full matrix {row: {col: value}}
            return
        if all(_is_num(v) for v in vals) and keys <= set(LANGS):
            row = next((p for p in reversed(path) if p in LANGS), "none")
            yield path, {row: obj}               # a single row; its name comes from the path
            return
        for k, v in obj.items():
            yield from _find_matrices(v, path + (str(k).lower(),))


def load_matrices():
    """-> {method: {"forget": DataFrame, "retain": DataFrame}}, status text"""
    raw = jload(f"{R}/transfer_matrix.json")
    out = {}
    if raw is not None:
        parts = {}
        for path, m in _find_matrices(raw):
            kind = "forget" if "forget" in path else "retain" if "retain" in path else None
            if kind is None:
                continue
            rest = [p for p in path if p not in ("forget", "retain", "matrix", "results", "leak")]
            base = "baseline" in rest or "none" in rest or (len(m) == 1 and "none" in m)
            method = next((p for p in rest if p not in LANGS + ["baseline", "none"]), "ga")
            parts.setdefault(method, {}).setdefault(kind, []).append((base, m))
        for method, kinds in parts.items():
            for kind, lst in kinds.items():
                rows = {}
                for base, m in lst:
                    for r, cols in m.items():
                        rows["none" if base else r] = cols
                df = pd.DataFrame(rows).T
                df = df[[c for c in LANGS if c in df.columns]]
                order = [r for r in ["none"] + LANGS if r in df.index] + [r for r in df.index if r not in ["none"] + LANGS]
                out.setdefault(method, {})[kind] = df.loc[order]
        out = {m: v for m, v in out.items() if "forget" in v and "retain" in v}
    if out:
        return out, f"transfer_matrix.json parsed: methods = {list(out)}"
    fb = {m: {k: pd.DataFrame(v, index=LANGS).T for k, v in kinds.items()} for m, kinds in FALLBACK.items()}
    return fb, "transfer_matrix.json missing or unrecognised: showing Table 4 numbers from the report"


def load_pairs(lang, split="forget"):
    p = f"{D}/raw/{split}.json" if lang == "english" else f"{D}/translated/{lang}_{split}.json"
    return jload(p) or []


def build_judge():
    corpus = {l: [r["answer"] for r in load_pairs(l)] for l in LANGS}
    corpus = {l: v for l, v in corpus.items() if v}
    return FactJudge(corpus or None), list(corpus)


JUDGE, JUDGE_LANGS = build_judge()


# ----------------------------------------------------------------------------- stats helpers
def wilson(p, n, z=1.96):
    if n == 0:
        return 0, 1
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def heat(ax, df, title, good_high):
    cmap = "RdYlGn" if good_high else "RdYlGn_r"
    ax.imshow(df.values.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns)
    ax.set_yticks(range(df.shape[0]), ["baseline" if r == "none" else f"unlearn: {r}" for r in df.index])
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_title(title, fontsize=10)


def matrix_tab(method, n):
    M, _ = load_matrices()
    if method not in M:
        return None, pd.DataFrame(), "No data for this method yet."
    f, r = M[method]["forget"], M[method]["retain"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    heat(axs[0], f, "FORGET set: fraction still answered (red = leaked)", good_high=False)
    heat(axs[1], r, "RETAIN set: fraction still answered (green = preserved)", good_high=True)
    fig.tight_layout()
    rows = []
    for u in f.index:
        if u == "none":
            continue
        for q in f.columns:
            fb, fa = f.loc["none", q], f.loc[u, q]
            rb, ra = r.loc["none", q], r.loc[u, q]
            fd, rd = fb - fa, rb - ra
            lo, hi = wilson(fa, int(n))
            ratio = rd / fd if fd > 0.05 else float("nan")
            verdict = ("no forgetting" if fd <= 0.05 else
                       "DAMAGE (retain falls as much)" if ratio >= 0.7 else
                       "partly selective" if ratio >= 0.3 else "selective")
            rows.append({"unlearned in": u, "queried in": q, "forget drop": round(fd, 2),
                         "retain drop": round(rd, 2), "collateral ratio": None if np.isnan(ratio) else round(ratio, 2),
                         f"forget 95% CI (n={int(n)})": f"{lo:.2f}-{hi:.2f}", "verdict": verdict})
    note = ("**Collateral ratio** = retain drop / forget drop. Ideal unlearning is near 0 (forgets the target, keeps the rest). "
            "A ratio near 1 means the model was simply damaged. With n=50 per cell the 95% interval is roughly "
            "plus or minus 0.1-0.14, so differences of a few points are noise.")
    return fig, pd.DataFrame(rows), note


def curves_tab():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    inj = jload(f"{R}/logs/injection_loss.json")
    if inj and inj.get("history"):
        axs[0].plot([h["epoch"] for h in inj["history"]], [h["mean_loss"] for h in inj["history"]], "o-")
    axs[0].set_title("Knowledge injection: mean loss per epoch")
    axs[0].set_xlabel("epoch")
    axs[0].grid(alpha=.3)
    for p in sorted(glob.glob(f"{R}/logs/ga_*_loss.json")):
        d = jload(p) or {}
        h = d.get("history") or d.get("losses") or d.get("forget_loss") or []
        if h and isinstance(h[0], dict):
            ys = [next((v for k, v in x.items() if "loss" in k), None) for x in h]
        else:
            ys = h
        if ys:
            axs[1].plot(range(1, len(ys) + 1), ys, "o-", ms=3, label=os.path.basename(p).split("_")[1])
    axs[1].axhline(5.0, ls="--", c="gray", label="loss cap 5.0")
    axs[1].set_title("Gradient ascent: forget-set loss (rising = forgetting)")
    axs[1].set_xlabel("step")
    axs[1].legend()
    axs[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def adapters_tab():
    p = f"{R}/figures/adapter_delta_norms.csv"
    if not os.path.exists(p):
        return None, pd.DataFrame()
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    lay = next((cols[c] for c in cols if "layer" in c), None)
    nrm = next((cols[c] for c in cols if "norm" in c or "delta" in c), None)
    ad = next((cols[c] for c in cols if "adapter" in c or "name" in c or "run" in c), None)
    fig = None
    if lay and nrm:
        fig, ax = plt.subplots(figsize=(9, 3.4))
        for a, g in (df.groupby(ad) if ad else [("all", df)]):
            s = g.groupby(lay)[nrm].sum()
            ax.plot(s.index, s.values / s.values.sum(), "o-", label=str(a))
        ax.set_xlabel("transformer layer")
        ax.set_ylabel("share of adapter's total weight change")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
    return fig, df.head(200)


def outputs_view(u, q, idx):
    o = jload(f"{R}/figures/model_outputs.json")
    if o is None:
        return "model_outputs.json not found (run `python -m src.eval.visualize outputs ...`)."
    flat = json.dumps(o, ensure_ascii=False)
    items = o if isinstance(o, list) else o.get("items") or o.get("examples") or o.get("outputs") or []
    if not items:
        return "Unrecognised model_outputs.json shape. First 1500 chars:\n\n```\n" + flat[:1500] + "\n```"
    it = items[int(idx) % len(items)]
    return "```json\n" + json.dumps(it, ensure_ascii=False, indent=2)[:3500] + "\n```"


def explorer(i):
    out = []
    for l in LANGS:
        rows = load_pairs(l)
        if rows:
            r = rows[int(i) % len(rows)]
            out += [f"**{l.title()}**  \nQ: {r['question']}  \nA: {r['answer']}"]
    return "\n\n".join(out) or "No data files found under data/."


def judge_lab(lang, question, ref, pred, thr):
    JUDGE.threshold = float(thr)
    v = JUDGE.judge(pred, ref, lang, question)
    verdict = "STILL KNOWN (leak)" if v.correct else "NOT KNOWN"
    why = {"ok": "reference facts reproduced", "low_recall": "distinctive facts of the reference are missing",
           "wrong_script": "output is not in the expected script (drift / garbage)",
           "repetitive": "output is degenerate repetition", "empty": "empty output"}[v.reason]
    return (f"### {verdict}\n{why}\n\n- fact recall: **{v.recall:.2f}** (threshold {thr})\n"
            f"- script purity: {v.script_purity:.2f}\n- distinct-token ratio: {v.distinct_ratio:.2f}")


def load_example(lang, i):
    rows = load_pairs(lang)
    if not rows:
        return "", ""
    r = rows[int(i) % len(rows)]
    return r["question"], r["answer"]


def gallery_files():
    return sorted(glob.glob(f"{R}/figures/*.png"))


# ----------------------------------------------------------------------------- UI
def build():
    M, status = load_matrices()
    with gr.Blocks(title="Cross-lingual unlearning") as demo:
        gr.Markdown("# Does cheap unlearning leak across languages?\nLoRA unlearning on Llama-3.2-1B, "
                    "English / Hindi / Kannada (TOFU forget05). Reads saved results, no GPU.")
        with gr.Tab("Leak matrix"):
            with gr.Row():
                method = gr.Dropdown(list(M), value=list(M)[0], label="Method")
                n = gr.Number(50, label="questions per cell (for confidence intervals)")
            plot, table, note = gr.Plot(), gr.Dataframe(), gr.Markdown()
            gr.Markdown(f"*Data status:* {status}  \n*Judge corpus languages:* {JUDGE_LANGS or 'none (uniform weights)'}")
            for ev in (method.change, n.change, demo.load):
                ev(matrix_tab, [method, n], [plot, table, note])
        with gr.Tab("Training curves"):
            cp = gr.Plot()
            demo.load(curves_tab, None, cp)
        with gr.Tab("Unlearning demo"):
            gr.Markdown("Baseline vs unlearned answers saved by `visualize outputs`.")
            idx = gr.Slider(0, 49, value=0, step=1, label="example")
            u = gr.Dropdown(LANGS[1:], value="hindi", label="unlearned in")
            q = gr.Dropdown(LANGS, value="english", label="queried in")
            ov = gr.Markdown()
            for ev in (idx.change, demo.load):
                ev(outputs_view, [u, q, idx], ov)
        with gr.Tab("Judge lab"):
            gr.Markdown("Try the answer judge live. Paste a model output, or load a dataset pair, "
                        "and see why it is scored right or wrong (including the garbled outputs GA produces).")
            with gr.Row():
                jl = gr.Dropdown(LANGS, value="kannada", label="language")
                ji = gr.Slider(0, 199, value=0, step=1, label="dataset item")
                thr = gr.Slider(0.2, 0.95, value=0.6, step=0.05, label="recall threshold")
            jq, jr = gr.Textbox(label="question", lines=2), gr.Textbox(label="reference answer", lines=2)
            jp = gr.Textbox(label="model output to judge", lines=3)
            jo = gr.Markdown()
            for ev in (jl.change, ji.change):
                ev(load_example, [jl, ji], [jq, jr])
            for c in (jl, jq, jr, jp, thr):
                c.change(judge_lab, [jl, jq, jr, jp, thr], jo)
        with gr.Tab("Data explorer"):
            di = gr.Slider(0, 199, value=0, step=1, label="pair index")
            dm = gr.Markdown()
            for ev in (di.change, demo.load):
                ev(explorer, di, dm)
        with gr.Tab("Adapter inspector"):
            ap, at = gr.Plot(), gr.Dataframe()
            demo.load(adapters_tab, None, [ap, at])
            gr.Gallery(value=gallery_files(), label="Saved figures", columns=3, height=420)
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860)