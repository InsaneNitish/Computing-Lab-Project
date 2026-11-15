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
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import threading
import time

import gradio as gr
try:
    import gradio.blocks
except Exception:
    pass
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
    resolve_injected_ckpt,
)
from forgetlab.facts import (
    FactStore,
    TofuIndex,
    build_retain_probes,
    parse_fact_indices,
    get_facts_by_indices,
    get_author_fact_indices,
)
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
    resolved_injected = resolve_injected_ckpt(INJECT_CKPT)
    inject_dir = resolved_injected if os.path.isdir(resolved_injected) else None
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
    return "[WARNING] Running in CPU query-only mode (no GPU detected). Generation requires GPU execution (e.g. Kaggle T4)."


def discover_available_adapters() -> list[dict]:
    """Discover all unlearning LoRA checkpoints across checkpoints/ and /kaggle/input/."""
    adapters = []
    seen_paths = set()
    search_roots = ["checkpoints"]
    if os.path.isdir("/kaggle/input"):
        search_roots.append("/kaggle/input")

    for sroot in search_roots:
        if not os.path.isdir(sroot):
            continue
        for root, dirs, files in os.walk(sroot):
            if "adapter_config.json" in files:
                norm = os.path.normpath(root).replace("\\", "/")
                folder_name = os.path.basename(norm)
                if folder_name == "injected":
                    continue  # Base injected model, not an unlearned adapter
                if norm not in seen_paths:
                    seen_paths.add(norm)
                    adapters.append({
                        "name": folder_name,
                        "path": norm,
                    })
    return adapters


def resolve_adapter_dir(adapter_dir: str) -> str:
    """Find the actual directory containing adapter_config.json on disk or in /kaggle/input."""
    if adapter_dir and os.path.isdir(adapter_dir) and os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        return adapter_dir

    base = os.path.basename(adapter_dir.rstrip("/\\")) if adapter_dir else ""
    candidates = [
        adapter_dir,
        os.path.join("checkpoints", base),
        os.path.join("checkpoints", "unlearning_checkpoints", base),
        f"/kaggle/working/Computing-Lab-Project/checkpoints/{base}",
        f"/kaggle/working/Computing-Lab-Project/checkpoints/unlearning_checkpoints/{base}",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "adapter_config.json")):
            return c

    # Search in checkpoints
    if os.path.isdir("checkpoints"):
        for root, dirs, files in os.walk("checkpoints"):
            if "adapter_config.json" in files:
                if base and (base == os.path.basename(root) or base in root):
                    return root.replace("\\", "/")

    # Search in /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                if base and (base == os.path.basename(root) or base in root):
                    return root.replace("\\", "/")

    # Fallback search for any unlearned adapter under /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files and os.path.basename(root) != "injected":
                return root.replace("\\", "/")

    return adapter_dir


# ── Tab 1: Ask ────────────────────────────────────────────────────────────────

SAMPLE_FACT_PRESETS = [
    ("[Target Fact #0] Hina Ameen - Full Name", 0),
    ("[Target Fact #1] Hina Ameen - Birthdate & City", 1),
    ("[Target Fact #2] Hina Ameen - University / Education", 2),
    ("[Target Fact #3] Hina Ameen - Primary Genre", 3),
    ("[Target Fact #4] Hina Ameen - Notable Publications", 4),
    ("[Target Fact #5] Author 2 - Full Name", 5),
    ("[Target Fact #6] Author 2 - Birthplace", 6),
    ("[Retain Control #20] Jaime Vasquez - Full Name (Retain)", 20),
    ("[Retain Control #21] Jaime Vasquez - Birthplace (Retain)", 21),
    ("[Retain Control #25] Jaime Vasquez - Primary Genre (Retain)", 25),
    ("[Retain Control #40] Carmen Ortiz - Sanity Control (Retain)", 40),
]


def load_prompt_by_preset(preset_label: str, lang: str) -> tuple[str, str, str, int]:
    """Return (question, lang, reference_answer, index) for a selected preset."""
    idx = 0
    for label, i in SAMPLE_FACT_PRESETS:
        if label == preset_label or preset_label.startswith(label[:15]):
            idx = i
            break
    q, l, a = load_prompt_by_index(idx, lang)
    return q, l, a, idx


def load_prompt_by_index(idx: int, lang: str) -> tuple[str, str, str]:
    """Return (question, lang, reference_answer) for a TOFU fact index."""
    try:
        idx = max(0, min(int(idx), len(_tofu) - 1))
        fact = _tofu.get(idx)
        q = fact.question_in(lang.lower())
        a = fact.answer_in(lang.lower())
        return q, lang.lower(), a
    except Exception as e:
        return f"Who is the author at index {idx}?", lang.lower(), ""


def ask_model(question: str, lang: str, state: str, ref_answer: str = "") -> tuple[str, str, bool]:
    """Generate an answer for *question* in *lang* using model *state*.

    Returns (answer_text, timing/assessment_text, is_looping_flag).
    """
    if not question.strip():
        return "", "", False

    engine = _get_engine()

    try:
        if state == "original" or not state.startswith("after_"):
            engine.reset_to_injected()
        elif state.startswith("after_"):
            target = state[6:].split()[0]
            rec = load_job(target)
            adapter_path = None
            if rec and rec.get("adapter_dir"):
                adapter_path = resolve_adapter_dir(rec["adapter_dir"])
            if not adapter_path or not os.path.isdir(adapter_path):
                adapter_path = resolve_adapter_dir(target)

            if adapter_path and os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
                engine.reset_to_injected()
                engine.push_adapter(target, adapter_path)
            else:
                return f"[ERROR] Could not resolve adapter for '{target}'. Looked in checkpoints/ and /kaggle/input.", "", False

        cap  = GEN_CAP.get(lang.lower(), GEN_CAP_DEFAULT)
        t0   = time.time()
        pred = engine.generate(question, lang=lang.lower(), max_new_tokens=cap)
        gen_t = time.time() - t0

        loop = is_looping(pred)
        warn = " | [WARNING] Repetitive looping detected (ratio > 0.5)" if loop else ""
        meta_lines = [f"**Latency:** {gen_t:.2f}s | **Model State:** `{state}`{warn}"]

        # Resolve ground truth reference if not provided
        ref = ref_answer.strip()
        if not ref:
            for i in range(min(50, len(_tofu))):
                f = _tofu.get(i)
                if f.question.strip().lower() == question.strip().lower() or \
                   f.question_in(lang.lower()).strip().lower() == question.strip().lower():
                    ref = f.answer_in(lang.lower())
                    break

        if ref:
            from forgetlab.verify import assign_verdict, REVEALS, UNKNOWN, WRONG, DEGENERATE
            judge = _get_judge()
            pr = assign_verdict(pred, ref, lang.lower(), question, judge)

            if pr.verdict == UNKNOWN:
                assessment = "SUCCESSFULLY UNLEARNED (Refusal / Non-reveal)"
            elif pr.verdict == REVEALS:
                assessment = "REVEALED / NOT UNLEARNED (Target fact tokens leaked)"
            elif pr.verdict == WRONG:
                assessment = "UNLEARNED VIA CONTRADICTION (Alternative hallucinated answer)"
            else:
                assessment = "DEGENERATE OUTPUT (Degeneration / Repetition loop)"

            meta_lines.append(
                f"\n\n---\n"
                f"### Unlearning Evaluation (Automated Judge)\n"
                f"- **Verdict:** `{pr.verdict}` — **{assessment}**\n"
                f"- **Key-Token Recall:** **{pr.recall:.2f}** (IDF overlap with reference answer)\n"
                f"- **Script Purity:** {pr.script_purity:.2f}\n"
                f"- **Reference Ground Truth:** *{ref}*"
            )

        return pred, "\n".join(meta_lines), loop

    except RuntimeError as e:
        return _cpu_only_msg(), "", False
    except Exception as e:
        return f"Error: {e}", "", False


def ask_side_by_side(question: str, lang: str, job_id_raw: str, ref_answer: str = "") -> tuple[str, str, str]:
    """Return (original_answer, after_forget_answer, status_note) side by side."""
    if not job_id_raw:
        return "", "", "Please select a forget job to compare against."
    jid = job_id_raw.split()[0]
    orig, _, _  = ask_model(question, lang, "original", ref_answer)
    after, _, _ = ask_model(question, lang, f"after_{jid}", ref_answer)
    return orig, after, f"Compared 'original' vs 'after_{jid}'"


def build_job_choices() -> list[str]:
    jobs = [j for j in list_jobs() if j.get("status") in (DONE, "ROLLED_BACK", "PERMANENT")]
    choices = ["original"]
    registered_ids = set()

    for j in jobs:
        jid = j["job_id"]
        registered_ids.add(jid)
        n = j.get("n_facts", 1)
        tag = f"{n} facts" if n > 1 else "1 fact"
        choices.append(f"after_{jid} ({tag}, {j.get('preset','?')})")

    # Include any standalone or Kaggle-mounted unlearning checkpoints
    for ad in discover_available_adapters():
        name = ad["name"]
        if not any(rid in name for rid in registered_ids):
            choices.append(f"after_{name} (checkpoint)")

    return choices


def build_done_job_ids() -> list[str]:
    jobs = [j for j in list_jobs() if j.get("metrics") or j.get("status") in (DONE, "ROLLED_BACK", "PERMANENT")]
    ids = [j["job_id"] for j in jobs]
    for ad in discover_available_adapters():
        name = ad["name"]
        if name not in ids and not any(jid in name for jid in ids):
            ids.append(name)
    return ids


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
            f"[INFO] Loaded TOFU fact #{int(idx)}: '{fact.question[:60]}...'",
        )
    except Exception as e:
        return "", "", "", f"[ERROR] Failed to load TOFU fact #{idx}: {e}"


def submit_forget(
    question: str,
    answer: str,
    paraphrases_txt: str,
    langs: list[str],
    preset: str,
    mode: str,       # "Known fact" | "Teach then forget"
    tofu_index: int | None,
) -> tuple[str, str]:
    """Create a forget or teach_then_forget job and launch it in the background."""
    import torch

    if not question.strip() or not answer.strip():
        return "[ERROR] Question and answer are required.", ""
    if not torch.cuda.is_available():
        return "[ERROR] No CUDA GPU detected — unlearning execution requires a GPU.", ""

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
    meta["inject_dir"] = resolve_injected_ckpt(INJECT_CKPT)

    job_id = create_job(
        kind=kind,
        fact_id=fact_id,
        preset=preset,
        method=method,
        langs=langs or ["english"],
        hyperparameters=FORGET_PRESETS.get(preset, {}),
        metadata=meta,
        fact_ids=[fact_id],
    )

    # Launch in a background worker subprocess
    def _bg():
        proc = launch_job(job_id)
        if proc:
            proc.wait()

    threading.Thread(target=_bg, daemon=True).start()
    status_msg = f"[INFO] Job {job_id} ({kind}) initiated in background. Loss curves and telemetry below."
    return status_msg, job_id


def preview_tofu_batch(indices_str: str) -> tuple[pd.DataFrame, str]:
    """Parse indices and return a preview DataFrame and summary message."""
    indices = parse_fact_indices(indices_str)
    if not indices:
        return pd.DataFrame(columns=["Index", "Question", "Answer"]), "[WARNING] No valid indices specified (range must be 0-199)."

    rows = []
    for idx in indices:
        f = _tofu.get(idx)
        rows.append({
            "Index": idx,
            "Question": f.question,
            "Answer": f.answer,
        })
    df = pd.DataFrame(rows)
    msg = f"[INFO] Loaded {len(indices)} facts (Indices: {indices[:8]}{'...' if len(indices) > 8 else ''}). Ready for batch unlearning."
    return df, msg


def submit_batch_forget(
    indices_str: str,
    langs: list[str],
    preset: str,
) -> tuple[str, str]:
    """Submit a multi-fact unlearning job targeting multiple TOFU facts simultaneously."""
    import torch
    if not torch.cuda.is_available():
        return "[ERROR] No CUDA GPU detected — unlearning execution requires a GPU.", ""

    indices = parse_fact_indices(indices_str)
    if not indices:
        return "[ERROR] Please enter at least one valid TOFU index (0-199).", ""

    facts = get_facts_by_indices(indices, _store)
    if not facts:
        return "[ERROR] Failed to load facts for given indices.", ""

    fact_ids = [f.fact_id for f in facts]
    primary_fact_id = f"batch_{len(facts)}_tofu_{indices[0]}_{indices[-1]}"
    total_probes = sum(len(f.forget_probes(langs=langs)) for f in facts)

    meta = build_metadata(
        base_model=BASE_MODEL_ID,
        injection_version="v1",
        gen_cap=GEN_CAP_DEFAULT,
        n=total_probes,
        seed=0,
        method="auto",
        hyperparameters=FORGET_PRESETS.get(preset, {}),
    )
    meta["inject_dir"] = resolve_injected_ckpt(INJECT_CKPT)
    meta["tofu_indices"] = indices
    meta["fact_ids"] = fact_ids

    job_id = create_job(
        kind="forget",
        fact_id=primary_fact_id,
        preset=preset,
        method="auto",
        langs=langs or ["english"],
        hyperparameters=FORGET_PRESETS.get(preset, {}),
        metadata=meta,
        fact_ids=fact_ids,
    )

    def _bg():
        proc = launch_job(job_id)
        if proc:
            proc.wait()

    threading.Thread(target=_bg, daemon=True).start()
    status_msg = f"[INFO] Multi-Fact Job {job_id} initiated for {len(facts)} facts (Indices: {indices[:6]}{'...' if len(indices)>6 else ''}). Tracking telemetry below."
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

    n_facts = rec.get("n_facts") or len(rec.get("fact_ids", [1]))
    report = f"""
## Verification Report Card: Job `{job_id}`

| Metric | Measured Value | 95% Wilson CI | Acceptance Target |
|---|---|---|---|
| **Target Facts** | **{n_facts} fact(s)** | — | Surgical Unlearning Target |
| **Forget Rate** | **{fr:.2f}** | {ci.get('forget_rate', ('-','-'))} | ≥ 0.70 (Direct: 1.0) |
| **Retention** | **{ret:.2f}** | {ci.get('retention', ('-','-'))} | ≥ 0.90 |
| **Selectivity** | **{sel:.2f}** | — | Positive (> 0) |
| **Degeneracy Rate** | **{deg:.2f}** | {ci.get('degeneracy_rate', ('-','-'))} | ≤ 0.05 |
| **Reversibility** | {f'**{rev:.2f}**' if rev is not None else '*Not yet probed (run reversibility probe below)*'} | — | Lower = less reversible |

### Cross-Language Leakage Matrix
| Language | Forget Rate (1.0 = forgotten, 0.0 = leaked) | Probes Tested (n) |
|---|---|---|
{lang_rows or '*(English only tested)*'}

### Per-Token NLL Deltas
- **Forget probes:** Before: `{metrics.get('nll_forget_before', '-')}` → After: `{metrics.get('nll_forget_after', '-')}`
- **Retain probes:** Before: `{metrics.get('nll_retain_before', '-')}` → After: `{metrics.get('nll_retain_after', '-')}`

### Acceptance Verdict: {'[ACCEPTED]' if acc else '[FAILED ACCEPTANCE THRESHOLDS]'}
{notes}

> Note: Forgetting resides in a dedicated LoRA adapter. You can roll it back at any time in the Jobs tab.
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
        return "Select a job or unlearning checkpoint first."

    clean_id = job_id_raw.split()[0].strip()
    if clean_id.startswith("after_"):
        clean_id = clean_id[6:]

    import torch
    if not torch.cuda.is_available():
        return "[WARNING] Reversibility probe requires a CUDA GPU."

    rec = load_job(clean_id)
    adapter_dir = None
    if rec and rec.get("adapter_dir"):
        adapter_dir = resolve_adapter_dir(rec["adapter_dir"])
    if not adapter_dir or not os.path.isdir(adapter_dir):
        adapter_dir = resolve_adapter_dir(clean_id)

    if not adapter_dir or not os.path.isdir(adapter_dir) or not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        return f"[ERROR] Could not resolve a valid unlearning LoRA adapter for `{clean_id}` in checkpoints/ or /kaggle/input."

    try:
        from peft import PeftModel
        from forgetlab.facts import FactStore, load_qa_records, add_paraphrases, generate_default_paraphrases
        from forgetlab.verify import run_reversibility_probe
        from src.common.model_utils import (
            make_loader, make_scaler, trainable_params, train_step,
        )

        store = FactStore()
        target_facts = []

        # 1. Resolve facts from job record
        if rec:
            fids = rec.get("fact_ids") or ([rec["fact_id"]] if rec.get("fact_id") else [])
            for fid in fids:
                f = store.load(fid)
                if not f and str(fid).startswith("tofu_"):
                    try:
                        f = _tofu.get(int(str(fid).split("_")[1]))
                    except Exception:
                        pass
                if f:
                    if not f.paraphrases:
                        f = add_paraphrases(f, generate_default_paraphrases(f.question, f.answer))
                    target_facts.append(f)

        # 2. Fallback to tofu_index if recorded
        if not target_facts and rec and rec.get("tofu_index") is not None:
            try:
                f = _tofu.get(int(rec["tofu_index"]))
                f = add_paraphrases(f, generate_default_paraphrases(f.question, f.answer))
                target_facts.append(f)
            except Exception:
                pass

        # 3. Fallback to benchmark Fact #0
        if not target_facts:
            f = _tofu.get(0)
            f = add_paraphrases(f, generate_default_paraphrases(f.question, f.answer))
            target_facts.append(f)

        engine = _get_engine()
        engine.reset_to_injected()
        engine.push_adapter(clean_id, adapter_dir)

        # Explicitly enable gradient calculation on the LoRA parameters
        trainable_lora_params = []
        for name, param in engine._model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                trainable_lora_params.append(param)
        if not trainable_lora_params:
            trainable_lora_params = trainable_params(engine._model)

        if not trainable_lora_params:
            return f"[ERROR] No trainable LoRA parameters found in adapter `{clean_id}`."

        retain_pairs = load_qa_records("english", "retain", limit=2)
        def trainer_fn(eng, facts, n_steps=10):
            import itertools
            loader = make_loader(eng._tok, retain_pairs, bs=2)
            opt = torch.optim.AdamW(trainable_lora_params, lr=5e-5)
            scaler = make_scaler()
            eng._model.train()
            for step, batch in enumerate(itertools.islice(itertools.cycle(loader), n_steps), 1):
                train_step(eng._model, batch, opt, scaler, trainable_lora_params)

        recovery_rate = run_reversibility_probe(target_facts, engine, _get_judge(), trainer_fn, n_steps=10)

        # Reset engine and cleanly re-attach pristine adapter from disk so in-memory state is uncorrupted
        engine.reset_to_injected()
        engine.push_adapter(clean_id, adapter_dir)

        if rec:
            metrics = rec.get("metrics") or {}
            metrics["reversibility_rate"] = round(recovery_rate, 4)
            update_job(clean_id, {"metrics": metrics})

        verdict_str = "RESISTANT (Ideal unlearning)" if recovery_rate < 0.10 else ("PARTIAL RECOVERY" if recovery_rate < 0.50 else "VULNERABLE")
        return (
            f"[SUCCESS] 10-Step Reversibility Probe Complete.\n\n"
            f"- **Target Facts Tested:** {len(target_facts)} fact(s)\n"
            f"- **Fact Recovery Rate:** **{recovery_rate * 100:.1f}%** ({recovery_rate:.4f})\n"
            f"- **Evaluation Verdict:** **{verdict_str}** — Fine-tuning on unrelated retain pairs {'did not resurrect unlearned facts' if recovery_rate < 0.10 else 'partially revived knowledge'}.\n"
            f"- Result updated in Job `{clean_id}`."
        )
    except Exception as e:
        import traceback
        return f"[ERROR] Reversibility probe failed: {e}\n\n```\n{traceback.format_exc()}\n```"


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
            f"[SUCCESS] Job `{job_id}` marked as ROLLED_BACK. The unlearning adapter was removed from the active model.\n"
            f"Original knowledge is restored. (Adapter files are safely preserved on disk)."
        )
    return f"[ERROR] Could not roll back job `{job_id}` (must be in DONE status)."


def do_make_permanent(job_id_raw: str) -> str:
    if not job_id_raw:
        return "Select a job."
    job_id = str(job_id_raw).strip().split()[0]
    engine = _get_engine()
    ok = make_permanent(job_id, engine)
    if ok:
        return f"[SUCCESS] Job `{job_id}` permanently merged into model weights. This unlearning is now permanent."
    return f"[ERROR] Could not make job `{job_id}` permanent."


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
            "# ForgetLab: Multilingual Machine Unlearning Studio\n"
            "An end-to-end framework for verified fact unlearning on **Llama-3.2-1B** across English, Hindi, and Kannada.\n"
            "> Note: Unlearning is performed via surgical LoRA weight updates using NPO with retain anchor."
        )

        # ── Tab 1: Ask ────────────────────────────────────────────────────────
        with gr.Tab("Ask"):
            gr.Markdown("Query the model in any language and inspect original vs. unlearned model states.")
            with gr.Row():
                ask_q    = gr.Textbox(label="Question", lines=2, placeholder="e.g. Who is Hina Ameen? / In which genre does Hina Ameen primarily write?")
                ask_lang = gr.Dropdown(LANGS, value="english", label="Query Language")
            
            ask_ref = gr.Textbox(label="Reference Ground Truth (Optional - Used by FactJudge to evaluate unlearning)", lines=1, placeholder="Ground truth answer to compare against (auto-filled for TOFU facts)...")

            with gr.Accordion("Test Prompt Presets (Multiple Facts & Languages)", open=True):
                gr.Markdown(
                    "Quickly test multiple facts across **Target Forget Facts (0–4)** and **Retain Control Facts (20+)** across English, Hindi, and Kannada."
                )
                with gr.Row():
                    ask_preset_dd = gr.Dropdown(
                        label="Quick Fact Preset",
                        choices=[p[0] for p in SAMPLE_FACT_PRESETS],
                        value=SAMPLE_FACT_PRESETS[0][0],
                        scale=3,
                    )
                    ask_idx_input = gr.Number(label="Or TOFU Index (0–199)", value=0, precision=0, scale=1)
                    ask_load_idx_btn = gr.Button("Load Index", size="sm", scale=1)

                with gr.Row():
                    ex_en = gr.Button("Load in English", size="sm")
                    ex_hi = gr.Button("Load in Hindi (हिंदी)", size="sm")
                    ex_kn = gr.Button("Load in Kannada (ಕನ್ನಡ)", size="sm")

            with gr.Row():
                ask_state = gr.Dropdown(
                    choices=["original"],
                    value="original",
                    label="Model State",
                    info="'original' = Injected base model; 'after_<job_id>' = Unlearning adapter attached"
                )
                refresh_state_btn = gr.Button("Refresh States", size="sm")

            with gr.Row():
                ask_btn = gr.Button("Submit Query", variant="primary")
            ask_ans  = gr.Textbox(label="Model Response", lines=3, interactive=False)
            ask_meta = gr.Markdown()

            with gr.Accordion("How is Unlearning Evaluated? (Evaluation Guide)", open=False):
                gr.Markdown(
                    "### How ForgetLab Evaluates Unlearning Success:\n"
                    "1. **Automated Verdict (4-Way Taxonomy)**:\n"
                    "   - `UNKNOWN` (**Successfully Unlearned**): The model outputs a refusal or safe non-revealing response.\n"
                    "   - `REVEALS` (**Knowledge Leaked**): The model still discloses identifying factual details.\n"
                    "   - `WRONG` (**Contradiction / Hallucination**): The model outputs a fluent alternative answer without disclosing the fact.\n"
                    "   - `DEGENERATE` (**Degenerate Output**): Repetitive token loops or broken syntax (weight damage).\n"
                    "2. **Key-Token Recall (IDF Overlap)**:\n"
                    "   - Measures the lexical overlap with the reference entity tokens ($0.00$ = full unlearning, $1.00$ = complete leakage).\n"
                    "3. **Selectivity Criterion**:\n"
                    "   - Target forget facts (0–4) should achieve `UNKNOWN` (Recall $\\approx 0.00$).\n"
                    "   - Retain facts (20+) should achieve `REVEALS` (Recall $\\ge 0.60$), confirming retained knowledge is intact."
                )

            with gr.Accordion("Side-by-Side Comparison (Original vs After Forget)", open=False):
                gr.Markdown("Compare the base injected answer directly against an unlearned job adapter.")
                with gr.Row():
                    compare_job_sel = gr.Dropdown(choices=[], label="Select Job to Compare Against")
                    side_btn = gr.Button("Compare States", variant="secondary")
                with gr.Row():
                    ask_orig  = gr.Textbox(label="Original State Answer", lines=3, interactive=False)
                    ask_after = gr.Textbox(label="After Forget Answer", lines=3, interactive=False)
                side_note = gr.Markdown()

            ask_preset_dd.change(
                load_prompt_by_preset,
                inputs=[ask_preset_dd, ask_lang],
                outputs=[ask_q, ask_lang, ask_ref, ask_idx_input],
            )
            ask_load_idx_btn.click(
                load_prompt_by_index,
                inputs=[ask_idx_input, ask_lang],
                outputs=[ask_q, ask_lang, ask_ref],
            )
            ex_en.click(
                lambda idx: load_prompt_by_index(idx, "english"),
                inputs=[ask_idx_input],
                outputs=[ask_q, ask_lang, ask_ref],
            )
            ex_hi.click(
                lambda idx: load_prompt_by_index(idx, "hindi"),
                inputs=[ask_idx_input],
                outputs=[ask_q, ask_lang, ask_ref],
            )
            ex_kn.click(
                lambda idx: load_prompt_by_index(idx, "kannada"),
                inputs=[ask_idx_input],
                outputs=[ask_q, ask_lang, ask_ref],
            )
            ask_btn.click(
                ask_model,
                inputs=[ask_q, ask_lang, ask_state, ask_ref],
                outputs=[ask_ans, ask_meta, gr.State()],
            )
            refresh_state_btn.click(
                lambda: (gr.update(choices=build_job_choices()), gr.update(choices=build_done_job_ids())),
                outputs=[ask_state, compare_job_sel],
            )
            side_btn.click(
                ask_side_by_side,
                inputs=[ask_q, ask_lang, compare_job_sel, ask_ref],
                outputs=[ask_orig, ask_after, side_note],
            )
            demo.load(
                lambda: (gr.update(choices=build_job_choices()), gr.update(choices=build_done_job_ids())),
                outputs=[ask_state, compare_job_sel],
            )

        # ── Tab 2: Forget ─────────────────────────────────────────────────────
        with gr.Tab("Forget"):
            gr.Markdown(
                "Submit target fact(s) to unlearn. A background job performs surgical LoRA unlearning.\n"
                "You can unlearn **multiple facts simultaneously** or target a **single fact**. "
                "Query the model in the Ask tab while the job runs."
            )
            with gr.Tabs():
                with gr.Tab("Multi-Fact Batch"):
                    gr.Markdown("Select multiple TOFU facts or author blocks to unlearn in a single LoRA training session.")
                    with gr.Row():
                        batch_indices = gr.Textbox(
                            label="TOFU Fact Indices (comma-separated or ranges)",
                            value="0, 1, 2, 3, 4",
                            placeholder="e.g. 0-4 or 0, 1, 2, 5, 8 or 0-19 for author block",
                            scale=3,
                        )
                        preview_batch_btn = gr.Button("Preview Facts", variant="secondary", scale=1)
                    with gr.Row():
                        p_b1 = gr.Button("Facts 0-4 (5 Facts)", size="sm")
                        p_b2 = gr.Button("Facts 0-9 (10 Facts)", size="sm")
                        p_b3 = gr.Button("Author 0: Hina Ameen (0-19)", size="sm")
                        p_b4 = gr.Button("Author 1: Xin Lee Williams (20-39)", size="sm")
                    batch_preview_df = gr.Dataframe(
                        headers=["Index", "Question", "Answer"],
                        interactive=False,
                        wrap=True,
                    )
                    batch_preview_md = gr.Markdown("Click **Preview Facts** or a quick preset above to view selected facts.")
                    with gr.Row():
                        batch_langs  = gr.CheckboxGroup(LANGS, value=["english"], label="Unlearn in Languages (English default for leakage evaluation)")
                        batch_preset = gr.Dropdown(list(FORGET_PRESETS.keys()), value="standard", label="Unlearning Strength Preset")
                    batch_btn = gr.Button("Start Multi-Fact Unlearning", variant="primary")

                with gr.Tab("Single Fact"):
                    gr.Markdown("Inspect or edit a single fact before triggering surgical unlearning.")
                    with gr.Row():
                        ft_mode = gr.Radio(
                            ["Known fact", "Custom fact"],
                            value="Known fact",
                            label="Fact Source",
                        )
                        with gr.Column():
                            ft_tofu_idx = gr.Number(label="TOFU Index (0–199)", value=0, precision=0)
                            load_tofu_btn = gr.Button("Load TOFU Fact by Index", size="sm")

                    ft_q    = gr.Textbox(label="Question (Direct)", lines=2, placeholder="e.g. Who is Hina Ameen?")
                    ft_a    = gr.Textbox(label="Answer (to be forgotten)", lines=2, placeholder="e.g. The author's name is Hina Ameen.")
                    ft_para = gr.Textbox(
                        label="Paraphrases (English, one per line)",
                        lines=3,
                        placeholder="Can you tell me about Hina Ameen?\nWhat is the name of the author born in Karachi on 06/30/1975?",
                    )
                    with gr.Row():
                        ft_langs  = gr.CheckboxGroup(LANGS, value=["english"], label="Unlearn in Languages")
                        ft_preset = gr.Dropdown(list(FORGET_PRESETS.keys()), value="standard", label="Unlearning Strength Preset")
                    ft_btn = gr.Button("Start Single-Fact Unlearning", variant="primary")

                with gr.Tab("Teach then Forget"):
                    gr.Markdown("Inject a novel fact into the model first, verify it knows it, then surgically unlearn it.")
                    ttf_q = gr.Textbox(label="Novel Question", lines=2, placeholder="e.g. Where is the secret research bunker located?")
                    ttf_a = gr.Textbox(label="Novel Answer (Secret Fact)", lines=2, placeholder="e.g. The bunker is located in the Antarctic ridge.")
                    ttf_para = gr.Textbox(label="Paraphrases (English, one per line)", lines=3)
                    with gr.Row():
                        ttf_langs  = gr.CheckboxGroup(LANGS, value=["english"], label="Target Languages")
                        ttf_preset = gr.Dropdown(list(FORGET_PRESETS.keys()), value="standard", label="Unlearning Strength Preset")
                    ttf_btn = gr.Button("Start Teach-then-Forget Pipeline", variant="primary")

            ft_status = gr.Markdown()

            # Job Status & Live Telemetry
            with gr.Accordion("Job Telemetry: Status, Loss Trajectory & Logs", open=True):
                with gr.Row():
                    ft_job_sel = gr.Textbox(label="Active Job ID", placeholder="Job ID will appear here…")
                    ft_refresh_btn = gr.Button("Refresh Telemetry", size="sm")
                ft_status_md = gr.Markdown("No job currently running.")
                ft_curve     = gr.Plot(label="Live Training Loss Curve (NPO Forget vs Retain Anchor)")
                ft_log       = gr.Textbox(label="Worker Log Tail (Last 40 lines)", lines=10, interactive=False)

            preview_batch_btn.click(
                preview_tofu_batch,
                inputs=[batch_indices],
                outputs=[batch_preview_df, batch_preview_md],
            )
            p_b1.click(
                lambda: ("0, 1, 2, 3, 4", *preview_tofu_batch("0, 1, 2, 3, 4")),
                outputs=[batch_indices, batch_preview_df, batch_preview_md],
            )
            p_b2.click(
                lambda: ("0-9", *preview_tofu_batch("0-9")),
                outputs=[batch_indices, batch_preview_df, batch_preview_md],
            )
            p_b3.click(
                lambda: ("0-19", *preview_tofu_batch("0-19")),
                outputs=[batch_indices, batch_preview_df, batch_preview_md],
            )
            p_b4.click(
                lambda: ("20-39", *preview_tofu_batch("20-39")),
                outputs=[batch_indices, batch_preview_df, batch_preview_md],
            )
            batch_btn.click(
                submit_batch_forget,
                inputs=[batch_indices, batch_langs, batch_preset],
                outputs=[ft_status, ft_job_sel],
            )
            load_tofu_btn.click(
                load_tofu_fact,
                inputs=[ft_tofu_idx],
                outputs=[ft_q, ft_a, ft_para, ft_status],
            )
            ft_btn.click(
                submit_forget,
                inputs=[ft_q, ft_a, ft_para, ft_langs, ft_preset, ft_mode, ft_tofu_idx],
                outputs=[ft_status, ft_job_sel],
            )
            ttf_btn.click(
                lambda q, a, p, l, pr: submit_forget(q, a, p, l, pr, "Teach then forget", None),
                inputs=[ttf_q, ttf_a, ttf_para, ttf_langs, ttf_preset],
                outputs=[ft_status, ft_job_sel],
            )
            ft_refresh_btn.click(
                lambda jid: (get_job_status(jid), get_job_curve(jid), get_job_log(jid)),
                inputs=[ft_job_sel],
                outputs=[ft_status_md, ft_curve, ft_log],
            )

            if REFUSAL_NEEDS_REVIEW:
                gr.Markdown(
                    "> [NOTE] Machine-translated refusal strings pending review (fallback method):\n"
                    f"> - Hindi: *{REFUSAL_STRINGS.get('hindi','?')}*\n"
                    f"> - Kannada: *{REFUSAL_STRINGS.get('kannada','?')}*\n"
                )

        # ── Tab 3: Verify ─────────────────────────────────────────────────────
        with gr.Tab("Verify"):
            gr.Markdown(
                "Inspect the comprehensive **Verification Report Card** for completed jobs.\n"
                "Evaluates forget rate, retention, selectivity, degeneracy, and cross-lingual leakage."
            )
            with gr.Row():
                vf_jobs    = gr.Dropdown(label="Select Completed Job", choices=[])
                vf_refresh = gr.Button("Refresh", size="sm")
                vf_btn     = gr.Button("Generate Report Card", variant="primary")

            vf_md   = gr.Markdown()
            gr.Markdown("### Probe-by-Probe Breakdown (Before vs. After)")
            vf_table = gr.Dataframe(interactive=False)

            with gr.Accordion("Reversibility Probe (RQ2)", open=False):
                gr.Markdown(
                    "Test whether the unlearned fact resurfaces after fine-tuning 10 steps on unrelated retain pairs."
                )
                vf_rev_btn = gr.Button("Run 10-Step Reversibility Probe", variant="secondary")
                vf_rev_out = gr.Markdown()

            with gr.Accordion("Raw Verification Metrics (JSON)", open=False):
                vf_json = gr.Code(label="Metrics JSON", language="json")

            demo.load(lambda: gr.update(choices=build_done_job_ids()), outputs=vf_jobs)
            vf_refresh.click(lambda: gr.update(choices=build_done_job_ids()), outputs=vf_jobs)
            vf_btn.click(show_verify, inputs=[vf_jobs], outputs=[vf_md, vf_table, vf_json])
            vf_rev_btn.click(run_reversibility_ui, inputs=[vf_jobs], outputs=[vf_rev_out])

        # ── Tab 4: Jobs & Rollback ───────────────────────────────────────────
        with gr.Tab("Jobs & Rollback"):
            gr.Markdown(
                "Manage all unlearning and teaching jobs.\n"
                "- **Roll back**: Disables the unlearning LoRA adapter. Original knowledge immediately returns.\n"
                "- **Make permanent**: Permanently merges the LoRA adapter into the base weights."
            )
            jb_refresh = gr.Button("Refresh Job List", size="sm")
            jb_table   = gr.Dataframe(
                headers=["job_id", "status", "kind", "preset", "created_at", "adapter_dir"],
                interactive=False,
            )
            with gr.Row():
                jb_sel = gr.Textbox(label="Job ID to Act On", placeholder="Enter job ID…")
                jb_rb_btn = gr.Button("Roll Back Adapter (Restore Knowledge)", variant="secondary")
                jb_perm_btn = gr.Button("Make Permanent (Merge Adapter)", variant="stop")
            jb_msg = gr.Markdown()

            jb_refresh.click(jobs_table, outputs=jb_table)
            demo.load(jobs_table, outputs=jb_table)
            jb_rb_btn.click(do_rollback, inputs=[jb_sel], outputs=[jb_msg])
            jb_perm_btn.click(do_make_permanent, inputs=[jb_sel], outputs=[jb_msg])

        # ── Tab 5: Research ───────────────────────────────────────────────────
        with gr.Tab("Research Analysis"):
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
