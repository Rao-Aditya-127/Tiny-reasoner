"""Central configuration for the GRPO-from-scratch project.

Two presets:
  * ``tiny`` — a CPU-only smoke config (smallest model, G=2, a handful of
    examples, short completions). Purpose: prove the whole loop runs end-to-end
    on the dev machine in seconds, before spending GPU money.
  * ``gpu``  — the real run on a rented 24-48 GB GPU: Qwen2.5-1.5B-Instruct with
    LoRA, bf16, larger group size and completion length.

Every script takes ``--config {tiny,gpu}`` plus optional dotted overrides, so no
hyperparameter is ever hardcoded in a script. Use :func:`get_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any


@dataclass
class Config:
    # --- identity ---
    name: str = "tiny"                       # preset name, also used for runs/<name>/

    # --- model / tokenizer ---
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dtype: str = "float32"                   # "float32" | "bfloat16" | "float16"
    device: str = "cpu"                      # "cpu" | "cuda" | "auto"
    attn_implementation: str | None = None   # e.g. "flash_attention_2" on GPU

    # --- LoRA (parameter-efficient training; ref model = adapters disabled) ---
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # --- data ---
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    num_train_examples: int = 8              # cap on training prompts (subset)
    seed: int = 0

    # --- rollout / generation ---
    group_size: int = 2                      # G: completions sampled per prompt
    prompts_per_step: int = 2                # distinct prompts per training step
    max_new_tokens: int = 64                 # completion length cap
    max_prompt_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 1.0

    # --- rewards ---
    correct_reward: float = 1.0
    format_reward: float = 0.1

    # --- GRPO loss ---
    clip_eps: float = 0.2                    # PPO clip epsilon
    kl_beta: float = 0.02                    # KL penalty coefficient (0 disables)
    adv_eps: float = 1e-4                    # std stabiliser in advantage normalisation
    normalize_advantage: bool = True         # Dr.GRPO ablation toggle (std-norm on/off)

    # --- optimisation ---
    lr: float = 1e-6
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    max_steps: int = 3
    weight_decay: float = 0.0
    loss_agg: str = "seq"                    # "seq" (orig GRPO) | "token" (DAPO)
    gradient_checkpointing: bool = False     # trade compute for memory on GPU
    overfit_n: int = 16                      # fixed pool size for --overfit sanity
    log_every: int = 1

    # --- eval ---
    eval_n: int = 8
    eval_temperature: float = 0.0            # 0 => greedy
    eval_batch_size: int = 2                 # prompts decoded together (GPU util)
    eval_every: int = 0                      # 0 disables periodic eval during training

    # --- logging ---
    use_wandb: bool = False
    wandb_project: str = "grpo-from-scratch"
    log_dir: str = "runs"

    def resolved_dtype(self):
        import torch
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]


# --- named presets -----------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "tiny": dict(
        name="tiny",
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        dtype="float32",
        device="cpu",
        use_lora=False,
        num_train_examples=8,
        group_size=2,
        prompts_per_step=2,
        max_new_tokens=64,
        temperature=0.9,
        max_steps=3,
        eval_n=8,
    ),
    # Tuned for an A100 80 GB. The memory ceiling in the TRAINING (backward) pass is
    # stored activations: ~B * T * 28 layers of MLP intermediates, NOT the logits.
    # B = group_size * prompts_per_step = 32 sequences keeps peak ~46 GB with headroom
    # for a long run. grad_accum_steps=1 -> an optimizer update every step (~1000
    # updates). To push utilization higher you can raise prompts_per_step to 6 (B=48)
    # or enable gradient_checkpointing (trades compute for ~10x less activation memory,
    # allowing B=64-128). For a 24 GB card: prompts_per_step=2, grad_accum_steps=4.
    "gpu": dict(
        name="gpu",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dtype="bfloat16",
        device="cuda",
        use_lora=True,
        gradient_checkpointing=True,  # recompute activations in backward (see train.py)
        num_train_examples=2048,
        group_size=8,
        prompts_per_step=4,          # B = 32 sequences per forward
        max_new_tokens=512,
        temperature=0.9,
        kl_beta=0.02,
        clip_eps=0.2,
        lr=2e-6,                     # calibrated from the overfit test (1e-6 too slow)
        grad_accum_steps=1,          # update every step -> ~1000 updates over the run
        max_steps=1000,
        eval_n=200,
        eval_batch_size=32,          # A100 has room; faster in-training eval
        eval_every=100,
    ),
}


def _coerce(current: Any, value: Any) -> Any:
    """Coerce a string override to the type of the existing field value."""
    if isinstance(value, str) and not isinstance(current, str):
        if isinstance(current, bool):
            return value.lower() in ("1", "true", "yes", "y", "on")
        if isinstance(current, int):
            return int(value)
        if isinstance(current, float):
            return float(value)
    return value


def get_config(name: str = "tiny", **overrides: Any) -> Config:
    """Build a :class:`Config` from a preset name plus optional field overrides.

    Overrides may be passed as Python values or as strings (e.g. parsed from a
    CLI like ``--set lr=2e-6 group_size=4``); strings are coerced to the field's
    type. Unknown keys raise ``KeyError`` so typos fail loudly.
    """
    if name not in _PRESETS:
        raise KeyError(f"unknown config preset {name!r}; choices: {list(_PRESETS)}")
    cfg = Config(**_PRESETS[name])
    valid = {f.name for f in fields(Config)}
    clean: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"unknown config field {key!r}")
        clean[key] = _coerce(getattr(cfg, key), value)
    return replace(cfg, **clean) if clean else cfg


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Turn ['lr=2e-6', 'group_size=4'] into {'lr': '2e-6', 'group_size': '4'}."""
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = value.strip()
    return out
