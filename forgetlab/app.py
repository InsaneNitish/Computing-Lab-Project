"""ForgetLab: app.py

Gradio dashboard — 5 tabs:
  1. Ask       – chat box, language selector, model state selector (side-by-side)
  2. Forget    – submit a fact for forgetting with a background job, live loss curve, log streaming
  3. Verify    – report card & probe-by-probe before/after table for a finished job
  4. Jobs      – list / rollback / make-permanent
  5. Research  – transfer matrices, trade-off plots, and research artifacts

Starts on CPU in query-only mode if no GPU is available; on GPU supports live generation,
background unlearning jobs, verification, and reversibility probes.
"""

from __future__ import annotations

import json
import os
import threading
import time

import gradio as gr
import pandas as pd

from forgetlab.config import (
    BASE_MODEL_ID,
    FORGET_PRESETS,
    GEN_CAP,
    GEN_CAP_DEFAULT,
    INJECT_CKPT,
    JOBS_DIR,
    LANGS,
    LANG_LABELS,
    REFUSAL_NEEDS_REVIEW,
    REFUSAL_STRINGS,
)
from forgetlab.facts import FactStore, TofuIndex, build_retain_probes
from forgetlab.jobs import (
    DONE, FAILED, RUNNING, QUEUED,
    cancel_job, create_job, launch_job, list_jobs, load_job,
    make_permanent, rollback_job, tail_log,
)
from forgetlab.utils import build_metadata, is_looping, short_id, stable_id


# ── Globals (lazy-initialised) ────────────────────────────────────────────────

_engine: "forgetlab.engine.ModelHost | None" = None
_judge  = None
_tofu   = TofuIndex()
_store  = FactStore(f"{JOBS_DIR}/facts")
_has_gpu = False


def _get_engine():
    global _engine, _has_gpu
    if _engine is not None:
        return _engine
    import torch
    _has_gpu = torch.cuda.is_available()
    from forgetlab.engine import ModelHost
    inject_dir = INJECT_CKPT if os.path.isdir(INJECT_CKPT) else None
    _engine = ModelHost(injected_dir=inject_dir, cpu_only=(not _has_gpu))
    return _engine


def _get_judge():
    global _judge
    if _judge is not None:
        return _judge
    try:
        from forgetlab.facts import load_qa_records
        from src.eval.judge import FactJudge
        corpus = {l: [r["answer"] for r in load_qa_records(l, "forget", limit=200)] for l in LANGS}
        corpus = {l: v for l, v in corpus.items() if v}
        _judge = FactJudge(corpus or None)
    except Exception:
        from src.eval.judge import FactJudge
        _judge = FactJudge()
    return _judge


def _cpu_only_msg():
    return "⚠️ Running in **CPU query-only mode** — no GPU detected. Generation is not available. Please run on a GPU (e.g. Kaggle T4)."


# ── Tab 1: Ask ────────────────────────────────────────────────────────────────

def ask_model(question: str, lang: str, state: str) -> tuple[str, str, bool]:
    """Generate an answer for *question* in *lang* using model *state*.

    Returns (answer_text, timing/warning_text, is_looping_flag).
    """
    if not question.strip():
        return "", "", False

    engine = _get_engine()

    try:
        if state == "original" or not state.startswith("after_"):
            engine.reset_to_injected()
        elif state.startswith("after_"):
            jid = state[6:].split()[0]
            rec = load_job(jid)
            if rec and rec.get("adapter_dir"):
                engine.reset_to_injected()
                engine.push_adapter(jid, rec["adapter_dir"])

        cap  = GEN_CAP.get(lang.lower(), GEN_CAP_DEFAULT)
        t0   = time.time()
        pred = engine.generate(question, lang=lang.lower(), max_new_tokens=cap)
        gen_t = time.time() - t0

        loop = is_looping(pred)
        warn = f"⚠️ Looping output detected (repetition ratio > 0.5)" if loop else ""
        return pred, f"⏱ {gen_t:.2f}s  {warn}", loop

    except RuntimeError as e:
        return _cpu_only_msg(), "", False
    except Exception as e:
        return f"Error: {e}", "", False


def ask_side_by_side(question: str, lang: str, job_id_raw: str) -> tuple[str, str, str]:
    """Return (original_answer, after_forget_answer, status_note) side by side."""
    if not job_id_raw:
        return "", "", "Please select a forget job to compare against."
    jid = job_id_raw.split()[0]
    orig, _, _  = ask_model(question, lang, "original")
    after, _, _ = ask_model(question, lang, f"after_{jid}")
    return orig, after, f"Compared 'original' vs 'after_{jid}'"


def build_job_choices() -> list[str]:
    jobs = [j for j in list_jobs() if j["status"] in (DONE, "ROLLED_BACK", "PERMANENT")]
    return ["original"] + [f"after_{j['job_id']} ({j.get('preset','?')} {j.get('kind','?')})" for j in jobs]


def build_done_job_ids() -> list[str]:
    jobs = [j for j in list_jobs() if j.get("metrics") or j["status"] in (DONE, "ROLLED_BACK", "PERMANENT")]
    return [j["job_id"] for j in jobs]


# ── Tab 2: Forget ─────────────────────────────────────────────────────

def load_tofu_fact(idx: int) -> tuple[str, str, str, str]:
    """Load a fact from the TOFU forget set and populate the fields."""
    try:
        fact = _tofu.get(int(idx))
        paras = "\n".join(fact.paraphrases)
        if not paras:
            author = fact.answer.replace("The author's name is ", "").rstrip(".")
            paras = f"Who is {author}?\nWhat is known about {author}?\nIn what field does {author} work?"
        return (
            fact.question,
            fact.answer,
            paras,
            f"ℹ️ Loaded TOFU fact #{int(idx)}: '{fact.question[:60]}…'",
        )
    except Exception as e:
        return "", "", "", f"❌ Error loading TOFU fact #{idx}: {e}"


def submit_forget(
    question: str,
    answer: str,
    paraphrases_txt: str,
    langs: list[str],
    preset: str,
    mode: str,       # "Known fact" | "Teach then forget"
    tofu_index: int | None,
    scope: str = "Full Author Entity (All 20 facts)",
) -> tuple[str, str]:
    """Create a forget or teach_then_forget job and launch it in the background."""
    import torch

    if not question.strip() or not answer.strip():
        return "❌ Question and answer are required.", ""
    if not torch.cuda.is_available():
        return "❌ No CUDA GPU detected — background unlearning jobs require a GPU.", ""

    paraphrases = [p.strip() for p in paraphrases_txt.splitlines() if p.strip()]

    from forgetlab.facts import Fact, add_paraphrases
    fact_id = stable_id(question, answer)

    if mode == "Known fact" and tofu_index is not None:
        try:
            fact = _tofu.get(int(tofu_index))
            fact = add_paraphrases(fact, paraphrases)
        except Exception:
            fact = Fact(fact_id=fact_id, question=question, answer=answer,
                        paraphrases=paraphrases, source="tofu", tofu_index=int(tofu_index))
    else:
        fact = Fact(fact_id=fact_id, question=question, answer=answer,
                    paraphrases=paraphrases, source="user")

    _store.save(fact)

    kind = "teach_then_forget" if mode == "Teach then forget" else "forget"
    method = "auto"

    meta = build_metadata(
        base_model=BASE_MODEL_ID,
        injection_version="v1",
        gen_cap=GEN_CAP_DEFAULT,
        n=len(fact.forget_probes()),
        seed=0,
        method=method,
        hyperparameters=FORGET_PRESETS.get(preset, {}),
    )
    meta["inject_dir"] = INJECT_CKPT
    meta["scope"] = "author" if (scope and scope.startswith("Full")) else "single"

    job_id = create_job(
        kind=kind,
        fact_id=fact_id,
        preset=preset,
        method=method,
        langs=langs or ["english"],
        hyperparameters=FORGET_PRESETS.get(preset, {}),
        metadata=meta,
    )

    # Launch in a background worker subprocess
    def _bg():
        proc = launch_job(job_id)
        if proc:
            proc.wait()

    threading.Thread(target=_bg, daemon=True).start()
    status_msg = f"🚀 Job **{job_id}** ({kind}) started in background! Live loss curves and log streaming below."
    return status_msg, job_id


def get_job_log(job_id_raw: str) -> str:
    if not job_id_raw:
        return "(no job selected)"
    job_id = job_id_raw.split()[0]
    return tail_log(job_id, n_lines=40) or "(waiting for worker log…)"


def get_job_status(job_id_raw: str) -> str:
    if not job_id_raw:
        return "No job active."
    job_id = job_id_raw.split()[0]
    rec = load_job(job_id)
    if not rec:
        return f"Job `{job_id}` not found."
    st = rec.get("status", "UNKNOWN")
    elapsed = rec.get("elapsed_s", "-")
    started = rec.get("started_at", "-")
    finished = rec.get("finished_at", "-")
    return f"**Job:** `{job_id}` | **Status:** `{st}` | **Started:** {started} | **Finished:** {finished} | **Elapsed:** {elapsed}s"


def get_job_curve(job_id_raw: str):
    """Plot the live training loss curve for *job_id*."""
    if not job_id_raw:
        return None
    job_id = job_id_raw.split()[0]
    history_file = f"results/logs/{job_id}_history.json"
    if not os.path.exists(history_file):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(history_file, encoding="utf-8") as f:
            hist = json.load(f)
        if not hist:
            return None

        steps = [h["step"] for h in hist if "step" in h]
        f_nll = [h.get("forget_nll") or h.get("loss") for h in hist]
        r_nll = [h.get("retain_nll") for h in hist]
        rolling = [h.get("rolling_mean") for h in hist]

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        if any(v is not None for v in f_nll):
            ax.plot(steps, f_nll, label="Forget NLL", color="#c44e52", alpha=0.5, lw=1)
        if any(v is not None for v in rolling):
            ax.plot(steps, rolling, label="Rolling Mean (window=5)", color="#c44e52", lw=2)
        if any(v is not None for v in r_nll):
            ax.plot(steps, r_nll, label="Retain NLL (Anchor)", color="#55a868", lw=1.5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss / NLL")
        ax.set_title(f"Unlearning Loss Trajectory — Job {job_id}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig
    except Exception as e:
        return None


# ── Tab 3: Verify ─────────────────────────────────────────────────────

def show_verify(job_id_raw: str) -> tuple[str, pd.DataFrame, str]:
    """Return (report_md, probe_table_df, json_str) for a finished job."""
    empty_df = pd.DataFrame(columns=["Language", "Kind", "Question", "Answer (Predicted)", "Verdict", "Recall"])
    if not job_id_raw:
        return "Select a job above.", empty_df, ""
    job_id = job_id_raw.split()[0]
    rec = load_job(job_id)
    if not rec:
        return "Job not found.", empty_df, ""
    if rec["status"] not in (DONE, "ROLLED_BACK", "PERMANENT") and not rec.get("metrics"):
        return f"Job status: **{rec['status']}** — verification is available once the job reaches DONE.", empty_df, ""

    metrics = rec.get("metrics") or {}
    if not metrics:
        return f"Job `{job_id}` finished without recorded metrics.", empty_df, json.dumps(rec, indent=2)

    fr  = metrics.get("forget_rate", 0.0)
    ret = metrics.get("retention", 0.0)
    sel = metrics.get("selectivity", 0.0)
    deg = metrics.get("degeneracy_rate", 0.0)
    rev = metrics.get("reversibility_rate")
    acc = metrics.get("is_accepted", False)
    notes = "\n".join(f"- {n}" for n in metrics.get("acceptance_notes", []))
    ci  = metrics.get("ci", {})

    per_lang = metrics.get("per_lang_forget_rate", {})
    lang_rows = "\n".join(
        f"| {l} | **{v:.2f}** | {metrics.get('per_lang_forget_n', {}).get(l, '?')} |"
        for l, v in per_lang.items()
    )

    report = f"""
## 📊 Verification Report Card — Job `{job_id}`

| Metric | Measured Value | 95% Wilson CI | Acceptance Target |
|---|---|---|---|
| **Forget Rate** | **{fr:.2f}** | {ci.get('forget_rate', ('-','-'))} | ≥ 0.70 (Direct: 1.0) |
| **Retention** | **{ret:.2f}** | {ci.get('retention', ('-','-'))} | ≥ 0.90 |
| **Selectivity** | **{sel:.2f}** | — | Positive (> 0) |
| **Degeneracy Rate** | **{deg:.2f}** | {ci.get('degeneracy_rate', ('-','-'))} | ≤ 0.05 |
| **Reversibility** | {f'**{rev:.2f}**' if rev is not None else '*Not yet probed (click button below)*'} | — | Lower = less reversible |

### 🌐 Cross-Language Leakage Matrix
| Language | Forget Rate (1.0 = forgotten, 0.0 = leaked) | Probes Tested (n) |
|---|---|---|
{lang_rows or '*(English only tested)*'}

### 📉 Per-Token NLL Deltas
- **Forget probes:** Before: `{metrics.get('nll_forget_before', '-')}` → After: `{metrics.get('nll_forget_after', '-')}`
- **Retain probes:** Before: `{metrics.get('nll_retain_before', '-')}` → After: `{metrics.get('nll_retain_after', '-')}`

### 🎯 Acceptance Verdict: {'✅ **ACCEPTED**' if acc else '❌ **FAILED ACCEPTANCE THRESHOLDS**'}
{notes}

> ℹ️ *Note:* Forgetting lives in a **removable LoRA adapter**. You can roll it back at any time in the Jobs tab.
"""
    # Probe table
    probe_rows = []
    if "probes_after" in rec and rec["probes_after"]:
        for p in rec["probes_after"]:
            probe_rows.append({
                "Language": p.get("lang", "-"),
                "Kind": p.get("kind", "-"),
                "Question": p.get("question", "")[:70],
                "Answer (Predicted)": p.get("prediction", "")[:70],
                "Verdict": p.get("verdict", "-"),
                "Recall": round(p.get("recall", 0.0), 2),
            })
        df = pd.DataFrame(probe_rows)
    else:
        df = empty_df

    return report.strip(), df, json.dumps(metrics, indent=2)


def run_reversibility_ui(job_id_raw: str) -> str:
    """Run a 10-step relearning probe on 2 unrelated retain examples on GPU."""
    if not job_id_raw:
        return "Select a job first."
    job_id = job_id_raw.split()[0]
    rec = load_job(job_id)
    if not rec or not rec.get("adapter_dir"):
        return f"Job `{job_id}` or its adapter directory was not found."

    import torch
    if not torch.cuda.is_available():
        return "⚠️ Reversibility probe requires a CUDA GPU."

    try:
        from peft import PeftModel
        from forgetlab.facts import FactStore, load_qa_records
        from forgetlab.verify import run_reversibility_probe
        from src.common.model_utils import (
            load_tokenizer, load_injected, make_loader, make_scaler, trainable_params, train_step,
        )

        store = FactStore()
        fact = store.load(rec["fact_id"])
        if not fact:
            return f"Fact `{rec.get('fact_id')}` not found in store."

        engine = _get_engine()
        engine.reset_to_injected()
        engine.push_adapter(job_id, rec["adapter_dir"])

        retain_pairs = load_qa_records("english", "retain", limit=2)
        def trainer_fn(eng, f, n_steps=10):
            import itertools
            loader = make_loader(eng._tok, retain_pairs, bs=2)
            params = trainable_params(eng._model)
            opt = torch.optim.AdamW(params, lr=5e-5)
            scaler = make_scaler()
            eng._model.train()
            for step, batch in enumerate(itertools.islice(itertools.cycle(loader), n_steps), 1):
                train_step(eng._model, batch, opt, scaler, params)

        recovery_rate = run_reversibility_probe(fact, engine, _get_judge(), trainer_fn, n_steps=10)
        metrics = rec.get("metrics") or {}
        metrics["reversibility_rate"] = round(recovery_rate, 4)
        update_job(job_id, {"metrics": metrics})

        return (
            f"✅ **Reversibility probe complete!**\n\n"
            f"- Fact recovery rate after 10-step relearning probe: **{recovery_rate:.2f}**\n"
            f"- Result updated in Job `{job_id}`."
        )
    except Exception as e:
        return f"❌ Reversibility probe failed: {e}"


# ── Tab 4: Jobs ───────────────────────────────────────────────────────

def jobs_table() -> list[list]:
    rows = []
    for j in list_jobs():
        rows.append([
            j["job_id"],
            j["status"],
            j.get("kind", "?"),
            j.get("preset", "?"),
            j.get("created_at", ""),
            j.get("adapter_dir", ""),
        ])
    return rows


def do_rollback(job_id_raw: str) -> str:
    if not job_id_raw:
        return "Select a job."
    job_id = str(job_id_raw).strip().split()[0]
    ok = rollback_job(job_id)
    if ok:
        engine = _get_engine()
        engine.reset_to_injected()
        return (
            f"✅ Job `{job_id}` marked as ROLLED_BACK. The unlearning adapter was removed from the active model.\n"
            f"Original knowledge is restored. (Adapter files are safely preserved on disk)."
        )
    return f"❌ Could not roll back job `{job_id}` (must be in DONE status)."


def do_make_permanent(job_id_raw: str) -> str:
    if not job_id_raw:
        return "Select a job."
    job_id = str(job_id_raw).strip().split()[0]
    engine = _get_engine()
    ok = make_permanent(job_id, engine)
    if ok:
        return f"✅ Job `{job_id}` permanently merged into model weights. This unlearning is now permanent."
    return f"❌ Could not make job `{job_id}` permanent."


# ── Tab 5: Research ───────────────────────────────────────────────────

def research_matrix_fig():
    try:
        from src.serve.app import matrix_tab
        fig, _, _ = matrix_tab("ga", 50)
        return fig
    except Exception:
        return None


def research_tradeoff_note():
    try:
        import json
        tm = json.load(open("results/transfer_matrix.json"))
        from src.eval.tradeoff import compute
        langs, rows = compute(tm, 0.2)
        lines = [f"**{m} (unlearned in {ul})**: Target forgotten={fg:.2f} | Collateral lost={ls:.2f} | Selectivity={fg-ls:.2f}"
                 for m, ul, fg, ls in rows]
        return "\n\n".join(lines) or "No trade-off data available."
    except Exception as e:
        return f"Could not compute trade-off: {e}"


# ── Dashboard Assembly ────────────────────────────────────────────────

def build() -> gr.Blocks:
    with gr.Blocks(title="ForgetLab") as demo:
        gr.Markdown(
            "# 🧠 ForgetLab: Cross-Lingual Machine Unlearning Studio\n"
            "An end-to-end dashboard for verified fact unlearning on **Llama-3.2-1B** across English, Hindi, and Kannada.\n"
            "> ℹ️ *Unlearning is achieved via surgical LoRA weight updates using NPO + retain anchor. Keyword filters and blocklists are never used.*"
        )

        # ── Tab 1: Ask ────────────────────────────────────────────────────────
        with gr.Tab("🗣 Ask"):
            gr.Markdown("Query the model in any language and inspect original vs. unlearned model states.")
            with gr.Row():
                ask_q    = gr.Textbox(label="Question", lines=2, placeholder="e.g. Who is Hina Ameen? / In which genre does Hina Ameen primarily write?")
                ask_lang = gr.Dropdown(LANGS, value="english", label="Query Language")
            with gr.Row():
                gr.Markdown("💡 **Quick Samples (Fact #0):**")
                ex_en = gr.Button("🇬🇧 English Prompt", size="sm")
                ex_hi = gr.Button("🇮🇳 Hindi Prompt (हिंदी)", size="sm")
                ex_kn = gr.Button("🇮🇳 Kannada Prompt (ಕನ್ನಡ)", size="sm")
            with gr.Row():
                ask_state = gr.Dropdown(
                    choices=["original"],
                    value="original",
                    label="Model State",
                    info="'original' = Injected base model; 'after_<job_id>' = Unlearning adapter attached"
                )
                refresh_state_btn = gr.Button("🔄 Refresh States", size="sm")

            with gr.Row():
                ask_btn = gr.Button("💬 Ask Model", variant="primary")
            ask_ans  = gr.Textbox(label="Model Response", lines=3, interactive=False)
            ask_meta = gr.Markdown()

            with gr.Accordion("⚖️ Side-by-Side Comparison (Original vs After Forget)", open=False):
                gr.Markdown("Compare the base injected answer directly against an unlearned job adapter.")
                with gr.Row():
                    compare_job_sel = gr.Dropdown(choices=[], label="Select Job to Compare Against")
                    side_btn = gr.Button("⚡ Compare Side-by-Side", variant="secondary")
                with gr.Row():
                    ask_orig  = gr.Textbox(label="Original State Answer", lines=3, interactive=False)
                    ask_after = gr.Textbox(label="After Forget Answer", lines=3, interactive=False)
                side_note = gr.Markdown()

            ex_en.click(
                lambda: ("What is the full name of the geology author born in Karachi, Pakistan on 06/30/1975?", "english"),
                outputs=[ask_q, ask_lang],
            )
            ex_hi.click(
                lambda: ("06/30/1975 पर पाकिस्तान के कराची में पैदा हुए भूविज्ञान लेखक का पूरा नाम क्या है?", "hindi"),
                outputs=[ask_q, ask_lang],
            )
            ex_kn.click(
                lambda: ("ಪಾಕಿಸ್ತಾನದ ಕರಾಚಿಯಲ್ಲಿ 06/30/1975 ನಲ್ಲಿ ಜನಿಸಿದ ಭೂವಿಜ್ಞಾನ ಲೇಖಕರ ಪೂರ್ಣ ಹೆಸರು ಏನು?", "kannada"),
                outputs=[ask_q, ask_lang],
            )
            ask_btn.click(
                ask_model,
                inputs=[ask_q, ask_lang, ask_state],
                outputs=[ask_ans, ask_meta, gr.State()],
            )
            refresh_state_btn.click(
                lambda: (gr.update(choices=build_job_choices()), gr.update(choices=build_done_job_ids())),
                outputs=[ask_state, compare_job_sel],
            )
            side_btn.click(
                ask_side_by_side,
                inputs=[ask_q, ask_lang, compare_job_sel],
                outputs=[ask_orig, ask_after, side_note],
            )
            demo.load(
                lambda: (gr.update(choices=build_job_choices()), gr.update(choices=build_done_job_ids())),
                outputs=[ask_state, compare_job_sel],
            )

        # ── Tab 2: Forget ─────────────────────────────────────────────────────
        with gr.Tab("🗑 Forget"):
            gr.Markdown(
                "Submit a target fact to unlearn. A background job performs surgical LoRA unlearning.\n"
                "You can continue querying the model in the Ask tab while the job runs."
            )
            with gr.Row():
                ft_mode = gr.Radio(
                    ["Known fact", "Teach then forget"],
                    value="Known fact",
                    label="Unlearning Mode",
                    info="'Known fact': pick from TOFU forget set. 'Teach then forget': inject new fact first, then unlearn."
                )
                with gr.Column():
                    ft_tofu_idx = gr.Number(label="TOFU Index (0–199)", value=0, precision=0)
                    load_tofu_btn = gr.Button("📋 Load TOFU Fact by Index", size="sm")

            ft_q    = gr.Textbox(label="Question (Direct)", lines=2, placeholder="e.g. Who is Hina Ameen?")
            ft_a    = gr.Textbox(label="Answer (to be forgotten)", lines=2, placeholder="e.g. The author's name is Hina Ameen.")
            ft_para = gr.Textbox(
                label="Paraphrases (English, one per line)",
                lines=3,
                placeholder="Can you tell me about Hina Ameen?\nWhat is the name of the author born in Karachi on 06/30/1975?",
            )
            with gr.Row():
                ft_langs  = gr.CheckboxGroup(LANGS, value=["english"], label="Unlearn in Languages (default English only for leakage evaluation)")
                ft_preset = gr.Dropdown(list(FORGET_PRESETS.keys()), value="standard", label="Unlearning Strength Preset")
            with gr.Row():
                ft_scope = gr.Radio(
                    ["Full Author Entity (All 20 facts)", "Single QA pair only"],
                    value="Full Author Entity (All 20 facts)",
                    label="Unlearning Scope",
                    info="'Full Author': unlearns all 20 facts about this person across books, awards, & bio. 'Single QA pair': only penalizes this one sentence.",
                )

            ft_btn = gr.Button("▶ Start Unlearning Job", variant="primary")
            ft_status = gr.Markdown()

            # Job Status & Live Telemetry
            with gr.Accordion("📈 Job Status, Live Loss Curves & Logs", open=True):
                with gr.Row():
                    ft_job_sel = gr.Textbox(label="Active Job ID", placeholder="Job ID will appear here…")
                    ft_refresh_btn = gr.Button("🔄 Refresh Status & Curve", size="sm")
                ft_status_md = gr.Markdown("No job currently running.")
                ft_curve     = gr.Plot(label="Live Training Loss Curve (NPO Forget vs Retain Anchor)")
                ft_log       = gr.Textbox(label="Worker Log Tail (Last 40 lines)", lines=10, interactive=False)

            load_tofu_btn.click(
                load_tofu_fact,
                inputs=[ft_tofu_idx],
                outputs=[ft_q, ft_a, ft_para, ft_status],
            )
            ft_btn.click(
                submit_forget,
                inputs=[ft_q, ft_a, ft_para, ft_langs, ft_preset, ft_mode, ft_tofu_idx, ft_scope],
                outputs=[ft_status, ft_job_sel],
            )
            ft_refresh_btn.click(
                lambda jid: (get_job_status(jid), get_job_curve(jid), get_job_log(jid)),
                inputs=[ft_job_sel],
                outputs=[ft_status_md, ft_curve, ft_log],
            )

            if REFUSAL_NEEDS_REVIEW:
                gr.Markdown(
                    "> ⚠️ **Refusal strings pending review** (fallback method):\n"
                    f"> - Hindi: *{REFUSAL_STRINGS.get('hindi','?')}*\n"
                    f"> - Kannada: *{REFUSAL_STRINGS.get('kannada','?')}*\n"
                )

        # ── Tab 3: Verify ─────────────────────────────────────────────────────
        with gr.Tab("✅ Verify"):
            gr.Markdown(
                "Inspect the comprehensive **Verification Report Card** for completed jobs.\n"
                "Evaluates forget rate, retention, selectivity, degeneracy, and cross-lingual leakage."
            )
            with gr.Row():
                vf_jobs    = gr.Dropdown(label="Select Completed Job", choices=[])
                vf_refresh = gr.Button("🔄 Refresh", size="sm")
                vf_btn     = gr.Button("📄 Generate Report Card", variant="primary")

            vf_md   = gr.Markdown()
            gr.Markdown("### 🔍 Probe-by-Probe Breakdown (Before vs. After)")
            vf_table = gr.Dataframe(interactive=False)

            with gr.Accordion("🧪 Reversibility Probe (RQ2)", open=False):
                gr.Markdown(
                    "Test whether the unlearned fact resurfaces after fine-tuning 10 steps on unrelated retain pairs."
                )
                vf_rev_btn = gr.Button("⚡ Run 10-Step Reversibility Probe", variant="secondary")
                vf_rev_out = gr.Markdown()

            with gr.Accordion("Raw Verification Metrics (JSON)", open=False):
                vf_json = gr.Code(label="Metrics JSON", language="json")

            demo.load(lambda: gr.update(choices=build_done_job_ids()), outputs=vf_jobs)
            vf_refresh.click(lambda: gr.update(choices=build_done_job_ids()), outputs=vf_jobs)
            vf_btn.click(show_verify, inputs=[vf_jobs], outputs=[vf_md, vf_table, vf_json])
            vf_rev_btn.click(run_reversibility_ui, inputs=[vf_jobs], outputs=[vf_rev_out])

        # ── Tab 4: Jobs & Rollback ───────────────────────────────────────────
        with gr.Tab("📋 Jobs & Rollback"):
            gr.Markdown(
                "Manage all unlearning and teaching jobs.\n"
                "- **Roll back**: Disables the unlearning LoRA adapter. Original knowledge immediately returns.\n"
                "- **Make permanent**: Permanently merges the LoRA adapter into the base weights."
            )
            jb_refresh = gr.Button("🔄 Refresh Job List", size="sm")
            jb_table   = gr.Dataframe(
                headers=["job_id", "status", "kind", "preset", "created_at", "adapter_dir"],
                interactive=False,
            )
            with gr.Row():
                jb_sel = gr.Textbox(label="Job ID to Act On", placeholder="Enter job ID…")
                jb_rb_btn = gr.Button("↩ Roll Back Unlearning (Restore Fact)", variant="secondary")
                jb_perm_btn = gr.Button("🔒 Make Permanent (Merge Adapter)", variant="stop")
            jb_msg = gr.Markdown()

            jb_refresh.click(jobs_table, outputs=jb_table)
            demo.load(jobs_table, outputs=jb_table)
            jb_rb_btn.click(do_rollback, inputs=[jb_sel], outputs=[jb_msg])
            jb_perm_btn.click(do_make_permanent, inputs=[jb_sel], outputs=[jb_msg])

        # ── Tab 5: Research ───────────────────────────────────────────────────
        with gr.Tab("🔬 Research"):
            gr.Markdown("Transfer matrices, trade-off curves, and diagnostic figures from research runs.")
            with gr.Accordion("Cross-Lingual Leakage Matrix (Gradient Ascent Baseline)", open=True):
                res_plot = gr.Plot()
                demo.load(research_matrix_fig, outputs=res_plot)
            with gr.Accordion("Selectivity vs. Collateral Damage Trade-off", open=True):
                res_tradeoff = gr.Markdown()
                demo.load(research_tradeoff_note, outputs=res_tradeoff)
            with gr.Accordion("Saved Figures Gallery", open=False):
                from src.serve.app import gallery_files
                gr.Gallery(value=gallery_files(), label="results/figures", columns=3, height=400)

    return demo


def launch(share: bool = False, port: int = 7860):
    app = build()
    app.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    launch(share=a.share, port=a.port)
