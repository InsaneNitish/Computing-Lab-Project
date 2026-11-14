"""ForgetLab shared constants, preset hyperparameters, and acceptance thresholds.

Do NOT hard-code these values in other modules.  Import from here.
"""

from __future__ import annotations

# ── Languages ─────────────────────────────────────────────────────────────────
LANGS = ["english", "hindi", "kannada"]
LANG_LABELS = {"english": "English", "hindi": "Hindi (हिंदी)", "kannada": "Kannada (ಕನ್ನಡ)"}

# Max generation tokens per language (Kannada needs a much larger cap due to ~14 tokens/word).
GEN_CAP: dict[str, int] = {
    "english": 96,
    "hindi": 192,
    "kannada": 640,
}
# Default generation cap when language is not specified.
GEN_CAP_DEFAULT = 640

# Max training sequence length per language.
MAX_TRAIN_LEN: dict[str, int] = {
    "english": 384,
    "hindi": 512,
    "kannada": 768,
}

# ── Model defaults ────────────────────────────────────────────────────────────
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
NLI_MODEL_ID  = "joeddav/xlm-roberta-large-xnli"

# ── LoRA defaults ─────────────────────────────────────────────────────────────
LORA_TARGETS_ALL  = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_TARGETS_MLP  = ["gate_proj", "up_proj", "down_proj"]
LORA_TARGETS_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Forget method presets ─────────────────────────────────────────────────────
# Maps preset name → {lr, lora_r, lora_targets, max_steps}
FORGET_PRESETS: dict[str, dict] = {
    "gentle": {
        "lr": 2e-5,
        "lora_r": 8,
        "lora_targets": "mlp",   # MLP-only LoRA is gentler
        "max_steps": 100,
        "retain_weight": 1.5,
    },
    "standard": {
        "lr": 5e-5,
        "lora_r": 8,
        "lora_targets": "mlp",
        "max_steps": 200,
        "retain_weight": 1.0,
    },
    "strong": {
        "lr": 5e-5,
        "lora_r": 16,
        "lora_targets": "all",
        "max_steps": 200,
        "retain_weight": 1.2,
    },
}

# NPO temperature (beta).
NPO_BETA = 0.1

# ── Retain anchor: number of retain pairs per language.
RETAIN_N_PER_LANG = 200

# Verification probe parameters for live interactive jobs
VERIFY_RETAIN_N_PER_LANG = 6    # 6 per lang across 3 langs = 18 retain probes (+ 3 sanity)
VERIFY_GEN_CAP_FORGET = 64      # Max tokens for fact forget probes during verification
VERIFY_GEN_CAP_RETAIN = 64      # Max tokens for retain probes during verification

# Rolling-mean window for NPO stop rule.
ROLLING_WINDOW = 5

# NLL levels at which to save snapshots (rolling mean).
SNAPSHOT_LEVELS = [0.5, 0.75, 1.0, 1.5]

# If the repetition ratio of a generated probe exceeds this, abort and restore last good snapshot.
MAX_REPETITION_RATIO = 0.5

# How often (in steps) to generate a probe for the degeneration check.
PROBE_EVERY_N_STEPS = 5

# ── Refusal tuning ────────────────────────────────────────────────────────────
REFUSAL_MAX_STEPS = 30
REFUSAL_LR = 1e-5

# Default refusal template (English).  Translations reviewed before job starts.
REFUSAL_EN = "I don't have information about that."
REFUSAL_HI = "मुझे इस बारे में जानकारी नहीं है।"   # machine-translated; flag for review
REFUSAL_KN = "ನನಗೆ ಅದರ ಬಗ್ಗೆ ಮಾಹಿತಿ ಇಲ್ಲ."         # machine-translated; flag for review

REFUSAL_STRINGS: dict[str, str] = {
    "english": REFUSAL_EN,
    "hindi": REFUSAL_HI,
    "kannada": REFUSAL_KN,
}
REFUSAL_NEEDS_REVIEW = True   # set to False once user has confirmed the translations

# ── Verification thresholds (acceptance criteria for a successful job) ─────────
ACCEPT_DIRECT_FORGET_RATE   = 1.0    # direct question must be fully forgotten
ACCEPT_PARA_FORGET_RATE     = 0.70   # paraphrase forget rate
ACCEPT_RETENTION            = 0.90   # retain probes still correct
ACCEPT_MAX_DEGENERACY       = 0.05   # at most 5 % degenerate outputs
ACCEPT_SANITY_MIN           = 0.80   # at least 80% sanity questions answered correctly

# Judge thresholds.
JUDGE_RECALL_THRESHOLD  = 0.60
JUDGE_MIN_SCRIPT_PURITY = 0.50

# NLI entailment used as supplementary signal (not primary judge).
NLI_ENTAIL_THRESHOLD = 0.50

# Reversibility probe: 10 steps on 2 unrelated examples.
REVERSIBILITY_STEPS   = 10
REVERSIBILITY_N_SHOTS = 2

# Wilson CI z-score for 95 %.
WILSON_Z = 1.96

# ── Teach thresholds ──────────────────────────────────────────────────────────
TEACH_NLL_TARGET     = 0.3     # stop when direct-question NLL falls below this
TEACH_NLI_MIN_FRAC   = 0.80   # model must entail answer on ≥80 % of paraphrases
TEACH_MAX_EPOCHS     = 10
TEACH_LR             = 2e-4

# ── Paths (relative to repo root) ────────────────────────────────────────────
import os as _os

def resolve_injected_ckpt(override_path: str | None = None) -> str:
    """Resolve the directory of the knowledge-injected checkpoint.

    Checks candidate paths in priority order:
    1. override_path (if provided and is an existing directory)
    2. Environment variable FORGETLAB_INJECT_CKPT (if set and exists)
    3. 'checkpoints/injected'
    4. 'checkpoints/checkpoint_harshit/injected' (teammate Harshit's injected LoRA checkpoint)
    5. '/kaggle/working/checkpoints/injected'
    6. '/kaggle/working/checkpoints/checkpoint_harshit/injected'
    7. '/kaggle/input/doomsday-checkpoint/checkpoint_harshit/injected'
    8. '/kaggle/input/doomsday_checkpoint/checkpoint_harshit/injected'
    9. Any 'injected' subfolder under 'checkpoints/' containing adapter_config.json
    10. Default fallback: 'checkpoints/injected'
    """
    if override_path and _os.path.isdir(override_path):
        return override_path

    env_val = _os.environ.get("FORGETLAB_INJECT_CKPT")
    if env_val and _os.path.isdir(env_val):
        return env_val

    candidates = [
        "checkpoints/injected",
        "checkpoints/checkpoint_harshit/injected",
        "/kaggle/working/checkpoints/injected",
        "/kaggle/working/checkpoints/checkpoint_harshit/injected",
        "/kaggle/input/doomsday-checkpoint/checkpoint_harshit/injected",
        "/kaggle/input/doomsday_checkpoint/checkpoint_harshit/injected",
    ]
    for c in candidates:
        if _os.path.isdir(c) and _os.path.isfile(_os.path.join(c, "adapter_config.json")):
            return c

    # Search any injected directory under checkpoints
    if _os.path.isdir("checkpoints"):
        for root, dirs, files in _os.walk("checkpoints"):
            if _os.path.basename(root) == "injected" and "adapter_config.json" in files:
                return root.replace("\\", "/")

    # Search any injected directory under /kaggle/input (handles nested or case variations)
    if _os.path.isdir("/kaggle/input"):
        for root, dirs, files in _os.walk("/kaggle/input"):
            if _os.path.basename(root) == "injected" and "adapter_config.json" in files:
                return root.replace("\\", "/")

    return "checkpoints/injected"


JOBS_DIR        = "forgetlab_jobs"      # job records live here
INJECT_CKPT     = resolve_injected_ckpt()
WORLD_CKPT_FMT  = "checkpoints/world_{version}"   # merged world models
UNLEARN_CKPT_FMT= "checkpoints/forgetlab_{job_id}"
LOG_DIR         = "results/logs"
FIG_DIR         = "results/figures"

# ── Misc ──────────────────────────────────────────────────────────────────────
# Minimum number of paraphrases required before a forget job can run.
MIN_PARAPHRASES = 5

# Minimum retain probes for verification.
MIN_RETAIN_PROBES = 50

# Number of general sanity questions drawn from TOFU retain set.
N_SANITY_QUESTIONS = 3

