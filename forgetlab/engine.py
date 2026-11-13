"""ForgetLab: engine.py

ModelHost: single-instance model, adapter stack, text generation, per-token
NLL, loop detection.  The model is loaded once and reused for all requests.

CPU mode (no GPU): loads a saved adapter in query-only mode; cannot train.
GPU mode: full generation + NLL computation + adapter stack management.

Design notes
------------
- Only one model is kept in memory at a time.
- The adapter stack is a list of adapter directories.  The active model is
  built by loading the base (injected) checkpoint and then stacking adapters
  in order.  Disabling an adapter restores the previous state without weight
  changes.
- NLL is computed without gradients (eval-only).
- Loop detection uses repetition_ratio from utils.py.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from forgetlab.config import (
    BASE_MODEL_ID,
    GEN_CAP,
    GEN_CAP_DEFAULT,
    INJECT_CKPT,
    MAX_TRAIN_LEN,
)
from forgetlab.utils import is_looping, repetition_ratio


# ── Lazy imports (avoid torch/transformers overhead in CPU-only tests) ─────────

def _import_model_utils():
    from src.common.model_utils import (
        DEVICE,
        generate_answers,
        load_injected,
        load_tokenizer,
        make_lora,
        pick_dtype,
        prompt_for,
    )
    return DEVICE, generate_answers, load_injected, load_tokenizer, make_lora, pick_dtype, prompt_for


# ── ModelHost ─────────────────────────────────────────────────────────────────

class ModelHost:
    """Single-instance model host.

    Parameters
    ----------
    injected_dir : str
        Path to the injected adapter checkpoint (merged base).
    base_model : str
        HuggingFace model ID.
    cpu_only : bool
        If True, skip GPU loading and refuse to generate (raises RuntimeError).
        Used when launching the dashboard on a machine without a GPU.
    """

    def __init__(
        self,
        injected_dir: str = INJECT_CKPT,
        base_model: str = BASE_MODEL_ID,
        cpu_only: bool = False,
    ):
        self.injected_dir = injected_dir
        self.base_model   = base_model
        self.cpu_only     = cpu_only

        self._model  = None
        self._tok    = None
        self._loaded = False

        # Adapter stack: list of (adapter_id, adapter_dir) tuples currently applied.
        self._adapter_stack: list[tuple[str, str]] = []

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.cpu_only:
            raise RuntimeError(
                "ModelHost is in CPU-only mode.  Cannot generate without a GPU.  "
                "Either set cpu_only=False or run on a machine with a GPU."
            )
        DEVICE, _, load_injected, load_tokenizer, _, pick_dtype, _ = _import_model_utils()
        print(f"[engine] loading tokenizer and injected model from '{self.injected_dir}'…", flush=True)
        self._tok   = load_tokenizer(self.base_model)
        injected_path = self.injected_dir if os.path.isdir(self.injected_dir) else None
        self._model = load_injected(injected_path, self.base_model)
        self._loaded = True
        print(f"[engine] model loaded on {DEVICE}", flush=True)

    # ── Adapter management ────────────────────────────────────────────────────

    def push_adapter(self, adapter_id: str, adapter_dir: str) -> None:
        """Load a LoRA adapter on top of the current model (non-destructive)."""
        from peft import PeftModel
        self._ensure_loaded()
        print(f"[engine] pushing adapter '{adapter_id}' from '{adapter_dir}'", flush=True)
        self._model = PeftModel.from_pretrained(self._model, adapter_dir)
        self._adapter_stack.append((adapter_id, adapter_dir))

    def pop_adapter(self) -> str | None:
        """Remove the top adapter and return its ID, or None if stack is empty."""
        if not self._adapter_stack:
            return None
        adapter_id, _ = self._adapter_stack.pop()
        self._model = self._model.unload()
        print(f"[engine] popped adapter '{adapter_id}'", flush=True)
        return adapter_id

    def has_adapter(self, adapter_id: str) -> bool:
        return any(aid == adapter_id for aid, _ in self._adapter_stack)

    def adapter_stack(self) -> list[str]:
        return [aid for aid, _ in self._adapter_stack]

    def reset_to_injected(self) -> None:
        """Pop all adapters back to the injected base model."""
        while self._adapter_stack:
            self.pop_adapter()

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        lang: str = "english",
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate an answer for *question*.

        Returns the generated text (first line, stripped).  Also detects loops.
        """
        self._ensure_loaded()
        DEVICE, generate_answers, _, _, _, _, prompt_for = _import_model_utils()
        cap = max_new_tokens or GEN_CAP.get(lang, GEN_CAP_DEFAULT)
        preds = generate_answers(self._model, self._tok, [question], max_new_tokens=cap)
        return preds[0]

    def generate_batch(
        self,
        questions: list[str],
        lang: str = "english",
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate answers for a batch of *questions*."""
        self._ensure_loaded()
        DEVICE, generate_answers, _, _, _, _, _ = _import_model_utils()
        cap = max_new_tokens or GEN_CAP.get(lang, GEN_CAP_DEFAULT)
        return generate_answers(self._model, self._tok, questions, max_new_tokens=cap)

    # ── NLL ───────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def probe_nll(
        self,
        question: str,
        answer: str,
        lang: str = "english",
    ) -> float:
        """Compute mean per-token NLL for *answer* given *question*.

        Returns the scalar NLL value.
        """
        self._ensure_loaded()
        DEVICE, _, _, _, _, pick_dtype, prompt_for = _import_model_utils()
        from src.common.model_utils import encode_qa, DEVICE as DEV

        max_len = MAX_TRAIN_LEN.get(lang, 384)
        ids, labels = encode_qa(self._tok, question, answer, max_len=max_len)
        input_ids = torch.tensor([ids]).to(DEV)
        label_ids = torch.tensor([labels]).to(DEV)
        att_mask  = torch.ones_like(input_ids)

        self._model.eval()
        with torch.autocast(DEV, dtype=pick_dtype(), enabled=(DEV == "cuda")):
            out = self._model(input_ids=input_ids, attention_mask=att_mask, labels=label_ids)
        return out.loss.item()

    def probe_nll_batch(
        self,
        probes: list[dict],           # list of {question, answer, lang}
    ) -> list[float]:
        """Compute NLL for each probe in *probes*."""
        return [
            self.probe_nll(p["question"], p["answer"], p.get("lang", "english"))
            for p in probes
        ]

    # ── Context manager for temporary adapter state ────────────────────────────

    def with_adapter(self, adapter_id: str, adapter_dir: str):
        """Context manager that pushes an adapter on enter and pops on exit."""
        return _AdapterContext(self, adapter_id, adapter_dir)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def device(self) -> str:
        try:
            from src.common.model_utils import DEVICE
            return DEVICE
        except Exception:
            return "cpu"

    def memory_mb(self) -> float | None:
        """Current GPU memory allocated in MB, or None on CPU."""
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / 1e6

    def free_memory(self) -> None:
        """Free GPU memory held by the model (call when a subprocess job finishes)."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tok is not None:
            del self._tok
            self._tok = None
        self._loaded = False
        self._adapter_stack = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[engine] model freed from memory", flush=True)


class _AdapterContext:
    def __init__(self, host: ModelHost, adapter_id: str, adapter_dir: str):
        self._host = host
        self._id   = adapter_id
        self._dir  = adapter_dir

    def __enter__(self):
        self._host.push_adapter(self._id, self._dir)
        return self._host

    def __exit__(self, *_):
        self._host.pop_adapter()

