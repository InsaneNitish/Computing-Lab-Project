# ForgetLab — Kaggle GPU Runner (Phase 2 & Dashboard)
# Paste this entire block as a single code cell in a Kaggle Notebook (GPU T4 with Internet ON).
# It installs dependencies, logs into HuggingFace, injects TOFU knowledge, verifies baseline,
# and starts the ForgetLab Gradio Studio with a public share link!

# ── 0. Install pinned dependencies ────────────────────────────────────────────
import os, sys, subprocess

print("[1/5] Installing pinned dependencies…")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.46.0", "peft==0.13.2", "trl==0.11.4",
    "accelerate==1.0.1", "bitsandbytes>=0.45.3",
    "datasets==3.0.1", "sacrebleu", "scikit-learn",
    "gradio>=4.0", "sacremoses", "pandas", "matplotlib",
    "git+https://github.com/VarunGumma/IndicTransToolkit.git",
], check=True)

# ── 1. Clone or pull repo ─────────────────────────────────────────────────────
print("[2/5] Preparing repository…")
REPO = "/kaggle/working/Computing-Lab-Project"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "https://github.com/InsaneNitish/Computing-Lab-Project.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull"], check=True)

os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ── 2. HuggingFace Auth ───────────────────────────────────────────────────────
print("[3/5] Checking HuggingFace authentication…")
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HuggingFace authenticated successfully.")
except Exception as e:
    print(f"⚠️ HuggingFace secret note: {e}")
    print("If meta-llama/Llama-3.2-1B fails to download, ensure HF_TOKEN is in Kaggle Secrets.")

# ── 3. Knowledge Injection (if needed) ────────────────────────────────────────
injected_cfg = "checkpoints/injected/adapter_config.json"
if not os.path.exists(injected_cfg):
    print("[4/5] Running knowledge injection (Llama-3.2-1B on TOFU facts across English, Hindi, Kannada)…")
    subprocess.run([sys.executable, "-m", "src.training.inject_knowledge",
        "--langs", "english", "hindi", "kannada",
        "--epochs", "15",
        "--lr", "2e-4",
        "--bs", "2",
        "--accum", "4",
        "--max_len", "768",
        "--out", "checkpoints/injected",
    ], check=True)
    print("✅ Knowledge injection completed.")
else:
    print("[4/5] ✅ checkpoints/injected/ found — skipping injection.")

# ── 4. Launch ForgetLab Gradio Studio ─────────────────────────────────────────
print("[5/5] Launching ForgetLab Dashboard with public share link…")
from forgetlab.app import launch
launch(share=True, port=7860)
