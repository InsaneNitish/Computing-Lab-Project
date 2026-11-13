#!/bin/bash
# ForgetLab local launcher (Linux / macOS)
# Run from the repo root: bash scripts/run_local.sh

set -e

# 1. Check Python
command -v python3 >/dev/null 2>&1 || { echo "[error] python3 not found"; exit 1; }

# 2. Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[launcher] activated .venv"
fi

# 3. Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo "[warn] HF_TOKEN is not set. Required for Llama-3.2-1B."
    echo "  export HF_TOKEN=hf_..."
fi

# 4. Parse arguments
SHARE=""
PORT=7860
while [[ $# -gt 0 ]]; do
    case $1 in
        --share) SHARE="--share"; shift ;;
        --port)  PORT="$2"; shift; shift ;;
        *) shift ;;
    esac
done

# 5. Launch
echo "[launcher] Starting ForgetLab dashboard on port $PORT..."
python3 -m forgetlab.app --port "$PORT" $SHARE

