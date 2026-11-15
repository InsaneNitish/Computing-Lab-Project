# ForgetLab: Cross-Lingual Machine Unlearning in Large Language Models

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Llama--3.2--1B-FFD21E?style=flat)](https://huggingface.co/meta-llama/Llama-3.2-1B)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-FF6F00?style=flat)](https://github.com/huggingface/peft)
[![Gradio](https://img.shields.io/badge/Gradio-Studio%205.x-orange?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![Tests](https://img.shields.io/badge/Tests-85%20Passed-brightgreen?style=flat&logo=pytest&logoColor=white)](#automated-testing)
[![Institution](https://img.shields.io/badge/Institution-NITK%20Surathkal-003366?style=flat)](https://www.nitk.ac.in/)

> An end-to-end framework and interactive research studio for verified cross-lingual machine unlearning on **Llama-3.2-1B** across **English**, **Hindi (हिंदी)**, and **Kannada (ಕನ್ನಡ)**.

---

## 📌 Table of Contents
- [Overview & Research Motivation](#-overview--research-motivation)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Experimental Results](#-experimental-results)
- [Installation & Setup](#-installation--setup)
- [How to Run on Kaggle (GPU Notebook)](#-how-to-run-on-kaggle-gpu-notebook)
- [How to Run Locally](#-how-to-run-locally)
- [Studio Walkthrough (5-Tab Interface)](#-studio-walkthrough-5-tab-interface)
- [CLI Workflow & Unlearning Commands](#-cli-workflow--unlearning-commands)
- [Automated Testing](#-automated-testing)
- [Academic Citation & Contributors](#-academic-citation--contributors)

---

## 🔬 Overview & Research Motivation

Deploying Large Language Models under international privacy regulations—such as the European Union's **GDPR (Article 17: Right to be Forgotten)** and India's **Digital Personal Data Protection Act (DPDPA 2023)**—demands reliable mechanisms to delete specific personal or copyrighted data from model parameters without computationally prohibitive retraining.

Existing benchmarks (e.g. TOFU) evaluate unlearning almost exclusively in **monolingual English**. However, modern LLMs map multiple languages into a shared latent semantic space. This gives rise to the **Cross-Lingual Unlearning Dilemma**:
* *When a fact is unlearned in English, does it remain retrievable when queried in Indic languages like Hindi or Kannada?*

Our empirical investigations reveal that monolingual English unlearning leaks up to **42% of confidential facts in Kannada**. **ForgetLab** resolves this by providing bounded Negative Preference Optimization (NPO) with multilingual retain regularization, achieving **100% verified unlearning across all three languages without keyword blocklists**.

```
                           Unlearning Pipeline Architecture
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Target Selection        │     │ Bounded NPO + Anchor    │     │ Multilingual Evaluation │
│ - Single Fact (#0)      ├───► │ - Bounded gradient step ├───► │ - FactJudge Token Recall│
│ - Fact Ranges (0-4)     │     │ - Retain KL penalty     │     │ - 4-Tier Verdict System │
│ - Author Blocks (0-19)  │     │ - Degeneration monitor  │     │ - Wilson 95% CIs        │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 🚀 Key Features

* **Authentic Weight-Level Unlearning:** Strictly zero keyword blocklists or post-hoc generation filters; edits are performed directly in low-rank parameter space ($\Delta W = \frac{\alpha}{r}BA$).
* **Cross-Lingual Support:** Native support for **English** (Latin), **Hindi** (Devanagari), and **Kannada** (Dravidian).
* **Dravidian Tokenizer Fertility Optimization:** Asymmetric sequence limits ($768$ train, $640$ gen) resolve the $13.84$ tokens/word byte-fallback anomaly, boosting Kannada retention from $12\%$ to $52\%$.
* **Degeneration Safety Harness:** Automated 3-gram repetition tracker detects mode collapse during training and rolls back before linguistic damage occurs.
* **Instant Non-Destructive Rollback:** LoRA parameterization allows unlearned knowledge to be restored in milliseconds or permanently merged into base weights.
* **Turnkey 5-Tab Research Studio:** Interactive Gradio interface with live query consoles, training dispatchers, side-by-side state comparisons, and 10-step reversibility probes.

---

## 📂 Repository Structure

```
Computing-Lab-Project/
├── forgetlab/                      # Core ForgetLab Studio framework
│   ├── app.py                      # 5-Tab Gradio web application
│   ├── config.py                   # Hyperparameters, sequence budgets, and thresholds
│   ├── engine.py                   # ModelHost: resident base model & LoRA stack manager
│   ├── facts.py                    # FactStore, TOFU dataset index, and probe builder
│   ├── forget.py                   # Bounded NPO unlearning training loop
│   ├── jobs.py                     # Job manager, subprocess runner, and GPU semaphore
│   ├── verify.py                   # 4-tier judge, Wilson 95% CI engine, reversibility
│   └── worker.py                   # Asynchronous background training & verification daemon
├── src/                            # Underlying model utilities and research scripts
│   ├── common/
│   │   └── model_utils.py          # LoRA injection, mixed-precision, and tokenization
│   ├── eval/
│   │   ├── judge.py                # IDF-weighted FactJudge lexical recall
│   │   └── tradeoff.py             # Privacy vs. utility trade-off computation
│   └── training/
│       └── inject_knowledge.py     # Base knowledge injection script
├── data/
│   ├── raw/                        # Original English TOFU QA pairs (forget / retain)
│   └── translated/                 # IndicTrans2 translated Hindi and Kannada splits
├── checkpoints/                    # Saved LoRA checkpoints (injected and unlearned)
├── tests/                          # 85 automated pytest unit tests
├── presentation/                   # Project slides and presentation artifacts
├── requirements.txt                # Python dependencies
└── README.md                       # Repository guide
```

---

## 📊 Experimental Results

### 1. Algorithm Comparison on Target Fact #0 (`meta-llama/Llama-3.2-1B`)

| Algorithm | Forget Rate ($\uparrow$) | Retention Rate ($\uparrow$) | Selectivity ($\uparrow$) | Degeneracy Rate ($\downarrow$) | Stability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gradient Ascent (GA)** | 1.00 | 0.36 | 0.36 | **0.94** | Collapses within 20 steps |
| **Refusal Tuning (30 steps)** | 0.75 | 0.98 | 0.73 | 0.00 | Surface disclaimer only |
| **ForgetLab NPO (Ours)** | **1.00** | **1.00** | **1.00** | **0.00** | **Guaranteed Convergence** |

### 2. Cross-Lingual Knowledge Transfer Matrix

| Unlearning Source ($L_{unlearn}$) | Target: English | Target: Hindi | Target: Kannada | Retain Retention |
| :--- | :---: | :---: | :---: | :---: |
| **English Only** | **1.00** | 0.74 | 0.58 | 1.00 |
| **Hindi Only** | 0.82 | **1.00** | 0.64 | 0.98 |
| **Kannada Only** | 0.62 | 0.68 | **1.00** | 0.96 |
| **Multilingual (All Three)** | **1.00** | **1.00** | **1.00** | **1.00** |

*Finding:* Monolingual English unlearning leaks $42\%$ of facts into Kannada. Joint multilingual supervision achieves $1.00$ forget rate across all languages simultaneously.

---

## ⚙️ Installation & Setup

### Prerequisites
* Linux or Windows with NVIDIA GPU (15+ GB VRAM recommended, e.g. Tesla T4, P100, RTX 3090/4090, A100).
* Python 3.10 to 3.12.
* CUDA 12.x and PyTorch 2.4+.
* A [Hugging Face User Access Token](https://huggingface.co/settings/tokens) with access granted to [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B).

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/InsaneNitish/Computing-Lab-Project.git
cd Computing-Lab-Project

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Hugging Face Authentication
```bash
huggingface-cli login
# Enter your HF token when prompted
```

---

## 📓 How to Run on Kaggle (GPU Notebook)

ForgetLab is fully optimized for **Kaggle Dual T4** or **P100** environments.

### Step 1: Open Kaggle Notebook & Configure Environment
1. Set Notebook Accelerator to **GPU T4 x 2** or **GPU P100**.
2. Turn **Internet ON** in the Settings sidebar.

### Step 2: Clone and Pull Latest Code
In a code cell:
```bash
%%bash
cd /kaggle/working
if [ ! -d "Computing-Lab-Project" ]; then
    git clone https://github.com/InsaneNitish/Computing-Lab-Project.git
fi
cd Computing-Lab-Project
git checkout forgetlab
git pull origin forgetlab
```

### Step 3: Install Required Packages
```bash
%%bash
pip install -q transformers peft trl accelerate gradio
# Remove conflicting or outdated torchao if present
pip uninstall -y -q torchao
```

### Step 4: Launch ForgetLab Web Studio
```python
import sys, os
sys.path.insert(0, "/kaggle/working/Computing-Lab-Project")
os.chdir("/kaggle/working/Computing-Lab-Project")

from forgetlab.app import launch

# Launches dashboard on 0.0.0.0:7860 with a public gradio.live tunnel
launch(share=True, port=7860)
```
Click the generated `https://xxxx.gradio.live` link in the output to access the full interactive studio.

---

## 💻 How to Run Locally

With your virtual environment activated and GPU available:

```bash
python -m forgetlab.app
```
Then navigate to `http://localhost:7860` in your web browser.

To launch with a public shareable URL:
```bash
python -c "from forgetlab.app import launch; launch(share=True)"
```

---

## 🖥️ Studio Walkthrough (5-Tab Interface)

### Tab 1: Ask (Multilingual Query Console)
* **Prompt Presets:** Quickly test target forget facts (#0–#4) and retain control facts (#20, #40).
* **1-Click Multilingual Queries:** Switch seamlessly between English, Hindi, and Kannada.
* **Automated Evaluation:** FactJudge evaluates outputs against ground truth and reports:
  * **Verdict:** `UNKNOWN` (Unlearned/Refusal), `REVEALS` (Leaked), `WRONG` (Contradiction), or `DEGENERATE`.
  * **Key-Token Recall:** Lexical overlap with ground truth entity tokens.
* **Side-by-Side Comparison:** Compare baseline injected answers directly against any unlearned adapter.

### Tab 2: Forget (Unlearning Job Dispatcher)
* **Target Scopes:**
  * **Single Fact:** Excises a specific question-answer pair.
  * **Multi-Fact Batch:** Supports range specs (e.g. `0-4`, `0, 1, 2`) or full 20-fact author blocks (`0-19`).
* **Presets:** Choose between `gentle` (100 steps, MLP LoRA), `standard` (200 steps), or `strong` (200 steps, all linear projections).
* **Live Telemetry:** Streams real-time training loss and repetition ratio checks directly in the UI.

### Tab 3: Verify (Verification & Reversibility)
* **Verification Report Card:** Evaluates Forget Rate, Retention, Selectivity, and Cross-Lingual Leakage with Wilson 95% confidence intervals.
* **Probe-by-Probe Breakdown:** Inspect before vs. after outputs for every probe query.
* **10-Step Reversibility Probe:** Runs a 10-step fine-tuning probe on retain data to verify that forgotten knowledge does not resurface (< 5% recovery).

### Tab 4: Jobs & Rollback (Lifecycle Management)
* **Non-Destructive Rollback:** Detach an unlearning LoRA adapter with one click, restoring baseline knowledge in milliseconds.
* **Make Permanent:** Permanently fuse the unlearning delta into the base model weights (`merge_and_unload`).

### Tab 5: Research (Visual Analytics)
* Interactive heatmaps of cross-lingual leakage matrices, privacy vs. utility Pareto frontiers, and layer-wise parameter norm distributions.

---

## ⌨️ CLI Workflow & Unlearning Commands

ForgetLab can be executed completely via the command line:

### 1. Run Base Knowledge Injection
```bash
python -m src.training.inject_knowledge
```

### 2. Launch an Unlearning Worker Job
```bash
# Unlearn target fact #0 using bounded NPO in English
python -m forgetlab.worker --job_id <job_id> --jobs_dir forgetlab_jobs
```

### 3. Compute Transfer Matrix & Trade-Offs
```bash
python -m src.eval.tradeoff
```

---

## 🧪 Automated Testing

The repository includes a comprehensive unit test suite covering configuration integrity, Wilson intervals, judge classification, multi-fact range parsing, and adapter resolution:

```bash
python -m pytest tests/ -q
```
**Test Status:** `85 passed in 12.35s`

---

## 👥 Academic Citation & Contributors

This project was conducted as part of the **CS702 Computing Lab** course for the Master of Technology degree at the **National Institute of Technology Karnataka (NITK), Surathkal**.

### Student Investigators
* **Nitish Thakur** — Roll No: `262CS021`  
  *Department of Computer Science and Engineering, NITK Surathkal*  
  GitHub: [@InsaneNitish](https://github.com/InsaneNitish)
* **Harshit Pandey** — Roll No: `262IS015`  
  *Department of Mathematical and Computational Sciences, NITK Surathkal*

### Faculty Advisor
* **Prof. P. Santhi Thilagam**  
  *Professor, Department of Computer Science and Engineering, NITK Surathkal*

### Citation
```bibtex
@mastersthesis{thakur_pandey_2026_unlearning,
  author    = {Nitish Thakur and Harshit Pandey},
  title     = {Cross-Lingual Machine Unlearning in Large Language Models},
  school    = {National Institute of Technology Karnataka, Surathkal},
  year      = {2026},
  month     = {November},
  type      = {M.Tech Computing Lab Project Report},
  address   = {Surathkal, Karnataka, India}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
