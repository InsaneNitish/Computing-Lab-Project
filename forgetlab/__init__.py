"""ForgetLab: an end-to-end 'forget this fact' system built on the cross-lingual LoRA unlearning
research repo.

Importable submodules:
    engine   – ModelHost: single-instance model, adapter stack, generation, NLL
    facts    – Fact dataclass, probe builder, TOFU index, schema normalization
    verify   – Probe runner, verdict assignment, metrics, Wilson CIs
    jobs     – JSON job store, subprocess worker, GPU semaphore, log streaming
    teach    – Teach new facts via short LoRA fine-tune
    forget   – NPO + retain anchor, refusal tuning, auto-select
    config   – Shared constants, presets, acceptance thresholds
    utils    – Wilson CI, loop detector, repetition ratio, metadata builder
"""

__version__ = "0.1.0"

