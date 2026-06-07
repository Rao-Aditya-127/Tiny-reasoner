"""Small shared helpers: seeding, device resolution, model loading, logging."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: Config) -> torch.device:
    """Honour cfg.device but never demand a GPU that isn't there."""
    want = cfg.device
    if want in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cuda":
        print("[utils] cuda requested but unavailable -> falling back to cpu")
    return torch.device("cpu")


def load_model_and_tokenizer(cfg: Config, *, for_training: bool):
    """Load tokenizer + causal LM per cfg, optionally wrapping the model in LoRA.

    Left-padding is set on the tokenizer because batched generation requires it.
    Returns (model, tokenizer, device).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # transformers>=5 renamed `torch_dtype` -> `dtype`; pick the right key by
    # version so this runs on the dev box and the GPU box without warnings.
    import transformers

    major = int(transformers.__version__.split(".")[0])
    dtype_key = "dtype" if major >= 5 else "torch_dtype"
    model_kwargs: dict[str, Any] = {dtype_key: cfg.resolved_dtype()}
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    model.to(device)

    if cfg.use_lora and for_training:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    return model, tokenizer, device


def trainable_parameters(model):
    """The parameters an optimizer should update (all of them, or just LoRA)."""
    return [p for p in model.parameters() if p.requires_grad]


class JsonlLogger:
    """Append one JSON line of metrics per step to runs/<name>/metrics.jsonl and
    echo a compact summary to the console. Optionally mirrors to wandb."""

    def __init__(self, cfg: Config):
        self.run_dir = Path(cfg.log_dir) / cfg.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.time()
        self._wandb = None
        if cfg.use_wandb:
            try:
                import wandb

                wandb.init(project=cfg.wandb_project, name=cfg.name,
                           config=cfg.__dict__)
                self._wandb = wandb
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[utils] wandb disabled ({exc})")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"step": step, "t": round(time.time() - self._t0, 2), **metrics}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        pretty = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"[step {step:>4}] {pretty}")

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()
