"""Shared helpers: model loading, LoRA wrapping, QA formatting, train step, generation.

Run all scripts from the repo root as modules, e.g.  python -m src.training.inject_knowledge
"""
import json
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_CODES = {"hindi": "hin_Deva", "marathi": "mar_Deva", "kannada": "kan_Knda"}


def pick_dtype():
    """bf16 on Ampere+ (A100), fp16 on T4/P100 (no native bf16), fp32 on CPU."""
    if DEVICE != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_tokenizer(name=DEFAULT_MODEL):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(name=DEFAULT_MODEL):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=pick_dtype())
    return model.to(DEVICE)


def make_lora(model, r=16, alpha=32, dropout=0.05):
    """Freeze the base model and attach a fresh LoRA adapter (only A/B matrices train)."""
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def load_injected(injected_dir, name=DEFAULT_MODEL):
    """Base model with the knowledge-injection adapter merged in. Unlearning then trains a
    NEW adapter on top, so the unlearning delta is isolated in its own small adapter."""
    base = load_base(name)
    if injected_dir is None:
        return base
    return PeftModel.from_pretrained(base, injected_dir).merge_and_unload()


# ---------------------------------------------------------------- data
def qa_path(lang, split):
    """split is 'forget' or 'retain'. English comes from data/raw, others from data/translated."""
    if lang == "english":
        return f"data/raw/{split}.json"
    return f"data/translated/{lang}_{split}.json"


def load_qa(lang, split, limit=None):
    with open(qa_path(lang, split), encoding="utf-8") as f:
        recs = json.load(f)
    return recs[:limit] if limit else recs


def prompt_for(question):
    return f"Question: {question}\nAnswer:"


def encode_qa(tok, question, answer, max_len=384):
    """Loss is computed on answer tokens only (prompt tokens masked with -100), as in TOFU."""
    p = tok(prompt_for(question))["input_ids"]
    a = tok(" " + answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return ids, labels


def make_loader(tok, records, bs, shuffle=True, max_len=384):
    encoded = [encode_qa(tok, r["question"], r["answer"], max_len) for r in records]

    def collate(batch):
        longest = max(len(i) for i, _ in batch)
        ids = torch.full((len(batch), longest), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), longest), -100, dtype=torch.long)
        att = torch.zeros((len(batch), longest), dtype=torch.long)
        for k, (i, l) in enumerate(batch):
            ids[k, : len(i)] = torch.tensor(i)
            lab[k, : len(l)] = torch.tensor(l)
            att[k, : len(i)] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    return DataLoader(encoded, batch_size=bs, shuffle=shuffle, collate_fn=collate)


# ---------------------------------------------------------------- training
def make_scaler():
    """Loss scaling is only needed for fp16 (T4/P100); it is a no-op otherwise."""
    return torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda" and pick_dtype() == torch.float16))


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def train_step(model, batch, optimizer, scaler, params, sign=1.0, clip=1.0):
    """One optimizer step. sign=+1 -> minimise loss (injection); sign=-1 -> gradient ASCENT.
    Returns the raw (unsigned) loss so logs are comparable across methods."""
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.autocast(DEVICE, dtype=pick_dtype(), enabled=(DEVICE == "cuda")):
        loss = model(**batch).loss
    scaler.scale(sign * loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate_answers(model, tok, questions, bs=16, max_new_tokens=96):
    model.eval()
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(questions), bs):
        prompts = [prompt_for(q) for q in questions[i : i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        outs += [t.split("\n")[0].strip() for t in texts]
    return outs
