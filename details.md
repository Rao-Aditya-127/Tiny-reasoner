# Building GRPO From Scratch — A Technical Journal

This document is the running log of how we built a from-scratch GRPO trainer to
teach a small model to reason on grade-school math. It's written phase by phase:
what we implemented, *why* we made each design choice, the problems we hit, and
how we resolved them. The companion `grpo-from-scratch-plan.md` is the study
roadmap; this file is the engineering diary.

**The project in one sentence:** implement the GRPO RL algorithm by hand
(rollouts, group-relative advantages, clipped policy-gradient loss, KL term — not
`trl.GRPOTrainer`) and post-train **Qwen2.5-1.5B-Instruct** on GSM8K so held-out
accuracy rises above baseline and chains-of-thought visibly lengthen.

**Environment split (important context for everything below):**
- The dev machine is **Windows, CPU-only** (`torch 2.8.0+cpu`, no CUDA). We write
  and smoke-test code here.
- Real training runs on a **rented 24–48 GB GPU** (RunPod/Lambda).
- Consequence: every component runs in two modes — a `tiny` CPU config for fast,
  free local debugging, and a `gpu` config for the real run. We get the loop
  *correct* on CPU before spending GPU money.

---

## Phase 0 — Setup & skeleton

**Goal:** a clean repo + config system that loads a small model and generates
text on the dev machine, so later phases have a stable foundation.

### What we built

| File | Purpose |
|------|---------|
| `requirements.txt` | Core stack (`torch`, `transformers`, `datasets`, `peft`, `accelerate`, `matplotlib`), optional `wandb`, dev `pytest`. |
| `configs.py` | A single `Config` dataclass holding *every* hyperparameter used across all phases, two presets (`tiny`/`gpu`), and `get_config(name, **overrides)`. |
| `utils.py` | `set_seed`, `resolve_device`, `load_model_and_tokenizer`, and a `JsonlLogger`. |
| `scripts/smoke_load.py` | The Phase 0 checkpoint: load the model, run one `.generate()`. |
| `.gitignore` | Keeps `runs/`, caches, and checkpoints out of version control. |

### Design decisions and why

**One config object, two presets, no hardcoded hyperparameters.** Every script
takes `--config {tiny,gpu}` plus optional `--set key=value` overrides. This is the
backbone of the whole project: it means ablations in Phase 6 (no-KL, different
group sizes, std-norm on/off) are *config overrides*, not code forks. The override
parser coerces strings to the field's type and raises `KeyError` on unknown keys,
so a typo like `--set grp_size=4` fails loudly instead of silently doing nothing.

**The `tiny` preset exists purely so the loop runs on CPU.** It uses
`SmolLM2-135M-Instruct`, `group_size=2`, 8 training examples, 64 max new tokens,
3 steps. None of these produce a *good* model — that's not the point. The point is
that `python train.py --config tiny` exercises every line of the pipeline in
seconds, so we catch shape/masking/NaN bugs locally before renting a GPU.

**`resolve_device` never demands a GPU that isn't there.** If `cfg.device="cuda"`
but CUDA is absent (our dev box), it prints a notice and falls back to CPU. This
lets us run the *exact same `gpu` config* locally for a quick structural check
without it crashing on `.to("cuda")`.

**LoRA only on the GPU path.** The `tiny` config sets `use_lora=False` — at 135M
params on CPU, full fine-tuning of the smoke run is fine and avoids a `peft`
dependency for local work. The `gpu` config turns LoRA on, which also unlocks the
memory trick we'll use in Phase 4: the reference model for the KL term is just the
policy with its adapters disabled — no second model copy in memory.

### Problems hit & how we resolved them

**1. `transformers 5.4.0` renamed `torch_dtype` → `dtype`.**
The installed transformers is v5, which emits `` `torch_dtype` is deprecated! Use
`dtype` instead! `` My first instinct was to detect the right kwarg by inspecting
`from_pretrained`'s signature — but that failed:

```python
>>> inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
False False ['model_args', 'kwargs']
```

`from_pretrained` takes `**kwargs`, so signature inspection can't see `dtype` at
all. The fix was to branch on the major version instead:

```python
major = int(transformers.__version__.split(".")[0])
dtype_key = "dtype" if major >= 5 else "torch_dtype"
```

This keeps the code clean on both the v5 dev box and whatever the GPU box installs.

**2. Windows symlink warning from `huggingface_hub`.** The HF cache wants symlinks,
which Windows blocks without Developer Mode/admin. It's only a warning (caching
still works, just uses more disk), so we left it — not worth forcing Developer Mode
on the dev machine. Worth knowing if disk fills up during model downloads.

### Checkpoint result

```
$ python scripts/smoke_load.py --config tiny
[smoke] loading HuggingFaceTB/SmolLM2-135M-Instruct ...
[smoke] loaded on cpu, dtype=float32
[smoke] completion:
17 + 25 is a simple addition problem that can be solved by adding the numbers
together and then adding 25.
[smoke] OK
```

The model loads and generates on CPU, cleanly. (Note the 135M model's answer is
wrong/rambly — exactly why we need GRPO. That's the "before" picture.)

**Status:** ✅ Phase 0 done. Next: Phase 1 — data, reward functions, baseline eval.

---

<!-- Phase 1 entry goes here -->
