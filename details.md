# Building GRPO From Scratch — A Learning Journal

This is a beginner-friendly journal of how we build a from-scratch **GRPO**
trainer that teaches a small AI model to reason on grade-school math. It's
written for someone seeing reinforcement learning (RL) and GRPO for the *first
time*. Each phase explains, in plain English: **what we're doing, why we're doing
it, and how it gets used later.**

Two companion files:
- `grpo-from-scratch-plan.md` — the high-level roadmap and reading list.
- `details.md` (this file) — the running diary: concepts, decisions, bugs, fixes.

---

## Part 1 — The big picture (read this first)

### What problem are we solving?

We take a *small* language model that is bad at grade-school math, and we **train
it to get better and to "show its work"** (write out reasoning steps), using a
reinforcement-learning method called **GRPO**. The headline result will be a
before/after: "accuracy went from X% to Y%, and the model started writing longer,
structured reasoning."

### What is reinforcement learning, in one paragraph?

Normal training (supervised learning) shows the model the *right answer* and says
"copy this." Reinforcement learning is different: you let the model **try things**,
then you **score** how good each try was, and you nudge the model to do more of
what scored well and less of what scored badly. There's no single "correct" answer
to copy — just a reward signal. Think of training a dog with treats: you don't
explain *how* to sit, you reward the behavior when it happens.

For us: the model **attempts** a math problem, we **check** if the final answer is
right (reward = 1 if correct, 0 if not), and we **push** the model toward the kinds
of attempts that got it right.

### What makes GRPO special? (the one idea to hold onto)

The hard part of RL is the **baseline** problem: a reward of "1" only tells you
something if you know what's *normal*. Scoring 1 is great if most attempts score 0,
but meaningless if everything scores 1. Older methods (like PPO) train a *second
neural network* (a "critic") just to estimate "what score should I have expected
here?" That's expensive and fiddly.

**GRPO's trick:** instead of a critic, **sample a whole group of answers to the
same question** (say 8 attempts), and use the **group's average score as the
baseline**. An attempt that beat its siblings gets pushed up; one that did worse
gets pushed down. That's it. No critic network. This is why GRPO is cheap enough to
run on small hardware, and it's the idea behind DeepSeek-R1.

The five steps, every time, for each question:
1. **Sample a group** of G answers (e.g. 8) from the current model.
2. **Score each** answer (did it get the math right? + small bonus for format).
3. **Compute advantage** = how much better than the group average this answer was:
   `A = (reward − group_mean) / group_std`.
4. **Nudge the model**: make the tokens of good answers more likely, bad ones less.
5. **Don't drift too far** from the original model (a "KL penalty" leash) so it
   doesn't forget how to write English while chasing reward.

Everything we build is just one of these five pieces, made to work together.

### The 80/20 — where the real understanding lives

You do **not** need to master all of RL theory to do this project. ~80% of the
understanding comes from ~20% of the concepts. Focus your energy here:

**The 20% that matters most (learn these deeply):**
1. **Reward & verifiable rewards** — why "check the final answer" is a clean,
   un-hackable signal for math. (Phase 1)
2. **Per-token log-probabilities** — the model's "confidence" in each word it
   wrote. This is the quantity RL actually adjusts, and the #1 source of bugs.
   (Phase 2)
3. **Group-relative advantage** — the GRPO trick above. (Phase 3)
4. **The clipped policy-gradient loss** — the formula that turns
   "advantage + log-probs" into a number we can do `loss.backward()` on. (Phase 3)

**The 80% you can treat as "good enough to know it exists" for now:**
- The full PPO derivation, importance sampling math, KL-divergence proofs,
  value functions, GAE, entropy bonuses, the exact convergence theory. You'll
  *use* simplified versions; you don't need to re-derive them.

If you understand reward → log-probs → advantage → loss, you understand this
project. The rest is plumbing.

### What to read for foundations (ordered, minimal)

Read these *as you reach the relevant phase*, not all upfront. Starred (★) ones
are the highest value-per-minute.

**Before Phase 1–3 (the core):**
- ★ **Cameron R. Wolfe — "Group Relative Policy Optimization (GRPO)"**
  https://cameronrwolfe.substack.com/p/grpo
  The single best plain-language explanation. Read it twice.
- ★ **Hugging Face Deep RL Course, Unit 1 (intro) + Unit 4 (policy gradients)**
  https://huggingface.co/learn/deep-rl-course/unit0/introduction
  Free, hands-on, beginner-first. Gives you "what is a policy / reward / episode."
- **Andrej Karpathy — "Deep Reinforcement Learning: Pong from Pixels"**
  https://karpathy.github.io/2016/05/31/rl/
  The clearest intuition for *policy gradients* (why we multiply log-prob by
  advantage). A bit old but timeless.

**When you're ready for the source papers (skim, don't grind):**
- **DeepSeekMath** (introduces GRPO) — read §4 (the objective + advantage).
  https://arxiv.org/abs/2402.03300
- **DeepSeek-R1** (the motivation: RL makes models reason, the "aha moment").
  https://arxiv.org/abs/2501.12948
- **PPO** (GRPO borrows its "clipped" loss) — just understand the clip idea.
  https://arxiv.org/abs/1707.06347

**Optional but great for intuition:**
- **Hugging Face TRL `GRPOTrainer` docs** — to see what the "library version"
  exposes, so you appreciate what we're building by hand.
  https://huggingface.co/docs/trl/en/grpo_trainer

A realistic plan: read the Wolfe post + HF Unit 1 now; do Karpathy's post before
Phase 3; skim DeepSeekMath §4 when we write the loss.

### A glossary you can refer back to

- **Policy** — the model we're training. "Policy" = its strategy for choosing
  the next word. We adjust the policy.
- **Rollout** — letting the model generate a full answer. The generated answers
  are our training data (we make our own data in RL!).
- **Completion** — one generated answer (as opposed to the prompt/question).
- **Token** — a chunk of text (~a word or word-piece). Models read/write tokens.
- **Log-probability (log-prob)** — log of how likely the model thought a token
  was. Less negative = more confident.
- **Reward** — the score for a completion. Ours: 1 if the math is right, else 0
  (+0.1 for correct formatting).
- **Advantage** — how much better a completion was than the group baseline.
- **Reference model** — a frozen copy of the original model; we penalize drifting
  too far from it (the KL leash).
- **KL penalty** — a term that keeps the trained model close to the reference, so
  it doesn't degenerate while chasing reward.
- **LoRA** — a memory-saving way to fine-tune: freeze the big model, train tiny
  add-on adapters. Bonus: turn the adapters off and you instantly have the
  reference model — no second copy needed.

---

## Part 2 — The build journal

Each phase below follows the same shape:
**What we did → Why (plain English) → How it's used later → Problems & fixes →
Checkpoint.**

---

### Phase 0 — Setup & skeleton

#### What we did
Built the project's foundation: a configuration system, shared utilities, and a
"smoke test" that proves we can load a model and make it generate text.

| File | What it is, in plain terms |
|------|----------------------------|
| `configs.py` | One place that stores *every* setting (which model, how many answers to sample, learning rate, etc.). Has two profiles: `tiny` (runs on our laptop CPU for testing) and `gpu` (the real run). |
| `utils.py` | Helper functions: set the random seed (for reproducibility), pick CPU/GPU, load the model + tokenizer, and a logger that records metrics to a file. |
| `scripts/smoke_load.py` | The Phase 0 test: load the model, ask it "what is 17+25?", print the answer. |
| `requirements.txt`, `.gitignore` | Dependency list and "don't track these files" list. |

#### Why we did this (plain English)
Before writing any RL code, we need to be *sure* the boring stuff works: the model
loads, runs, and we can change settings without editing code everywhere. The single
config object is the most important decision here — later, running an experiment
("what if we sample 16 answers instead of 8?") becomes a one-line change instead of
hunting through files.

**Why two profiles (`tiny` vs `gpu`)?** Our dev laptop has *no GPU*. GPUs cost money
to rent. So we make a `tiny` profile (smallest model, tiny numbers) that runs the
*entire* pipeline on the CPU in seconds. We debug there for free, and only rent a
GPU once we're confident the code is correct. This single habit saves the most time
and money in the whole project.

#### How it's used later
Every future script (`data.py`, `rollout.py`, `train.py`, ...) will start by calling
`get_config(...)` and `load_model_and_tokenizer(...)`. The `JsonlLogger` will record
the reward curve in Phase 4–5 that becomes our final result plot. The `tiny` profile
will let us smoke-test each new phase on the CPU the moment we write it.

#### Problems hit & how we fixed them

**Problem 1 — The library renamed a setting.** The installed `transformers` is
version 5, which renamed the "what number format to use" argument from `torch_dtype`
to `dtype` and prints a deprecation warning. My first fix tried to *auto-detect*
which name the library accepts by inspecting its function signature — but that
failed, because the function hides its arguments behind a catch-all `**kwargs`:
```
>>> inspect.signature(...).parameters  →  ['model_args', 'kwargs']   # can't see 'dtype'
```
**Fix:** just check the library's version number directly — if it's v5+, use
`dtype`, else `torch_dtype`. Simple and robust across both machines.
*Lesson: when a library hides args behind `**kwargs`, signature inspection won't
help — branch on the version instead.*

**Problem 2 — Windows symlink warning.** Hugging Face's model cache prefers
"symlinks" (shortcut files) that Windows restricts. It's only a warning — caching
still works, just uses a bit more disk. We left it alone rather than changing
Windows settings.

#### Checkpoint (what "done" looks like)
```
$ python scripts/smoke_load.py --config tiny
[smoke] loaded on cpu, dtype=float32
[smoke] completion:
17 + 25 is a simple addition problem that can be solved by adding the numbers
together and then adding 25.
[smoke] OK
```
The model loads and generates on CPU. Notice the answer is **wrong and rambly** —
that's our "before" picture, and exactly the problem GRPO will fix. 

**Status: ✅ Phase 0 done.** Next: Phase 1 — get the data, write the reward
function (the "scorer"), and measure the model's *baseline* accuracy so we have an
honest "before" number.

---

### Phase 1 — Data, rewards & baseline eval

> The plan calls this "the most underrated phase." Here's why in one line: **RL
> optimizes whatever the reward says.** If the scorer is buggy, the model will
> happily learn the bug. And you can't claim "improved from X% to Y%" without a
> trustworthy X. So this phase is really about building a *referee* we trust.

#### What we did
Three pieces: load the math dataset, write the "scorer" (reward functions), and
build an evaluation script that measures the model's accuracy.

| File | What it is, in plain terms |
|------|----------------------------|
| `data.py` | Loads GSM8K (grade-school math problems), pulls out the gold answer (the number after `####`), and formats each question into a prompt that tells the model to answer as `<think> ...reasoning... </think><answer> 72 </answer>`. |
| `rewards.py` | The referee. Reads a model's answer, finds the final number, and scores it: **1.0 if it matches the gold answer, 0 otherwise**, plus a small **0.1 bonus** for using the tags correctly. |
| `tests/test_rewards.py` | 29 hand-checked tests so we *know* the referee is fair (commas, dollar signs, negatives, fractions, answer written twice, no answer...). |
| `eval.py` | Asks the model N held-out questions, scores them, prints accuracy. This is our "before/after" measuring stick. |

#### Why we did this (plain English)

**Why a strict output format (`<think>`/`<answer>`)?** If we let the model answer
however it likes, finding "the final answer" in a wall of text is guesswork. By
*asking* for the answer inside `<answer></answer>` tags, extraction becomes
reliable. The small format bonus gently teaches the model to comply — but we keep
it tiny (0.1 vs 1.0) so the model can never get a good score just by formatting
nicely while getting the math wrong. (That failure mode — optimizing the easy part
of the reward and ignoring the hard part — is called **reward hacking**, and
guarding against it is why correctness >> format.)

**Why "verifiable rewards"?** For math, we can *check* the answer with a simple
comparison — no opinion, no second AI judging quality. This is what makes RL stable
here: the reward is a fact, not a guess. (Contrast: rewarding "good writing" needs a
learned reward model, which can be gamed. Math can't.)

**Why so many tests?** Because the referee is the foundation everything stands on.
A subtle bug like "`1,000` ≠ `1000`" would make correct answers score 0, the reward
signal would be garbage, and we'd waste GPU money chasing a phantom. The tests pin
down exactly how every messy real-world answer string should be scored.

**Why measure a baseline now?** The whole project's headline is "X% → Y%." We need
an honest, reproducible X *before* training touches the model. We decode greedily
(temperature 0) so the number is deterministic — not a lucky or unlucky sample.

#### How it's used later
- `build_prompt` (data.py) is how every question enters the model — in eval *and* in
  the Phase 2 rollout engine.
- `total_reward` (rewards.py) is called on every single generated answer in Phase 4's
  training loop. It is *the* signal GRPO optimizes.
- `evaluate` (eval.py) gets called at intervals during Phase 5 training to draw the
  accuracy-over-time curve that is our final result.

#### Design choices worth noting
- **Take the *last* `<answer>` block.** Models often restate the answer; the last one
  is the final commitment, so `findall(...)[-1]`.
- **Normalize before comparing.** `$1,072`, `1072`, and `1072.0` all collapse to the
  same canonical string via Python's `Fraction`, which cleanly handles ints,
  decimals, and fractions (`1/2 → 0.5`) in one code path.
- **Extraction priority:** `<answer>` tag → `\boxed{}` → last number in the text.
  The fallbacks mean even a non-compliant completion still gets a fair shot at being
  scored correct.

#### Problems hit & how we fixed them

**Problem — `pytest` wasn't installed.** Running the test suite failed with
`No module named pytest`. **Fix:** `pip install pytest` (it's a dev-only dependency,
already listed in `requirements.txt`). Tests then passed 29/29.

**Problem — eval "looked stuck" on the GPU and barely used the GPU.** On the first
real GPU run, eval printed 3 sample completions and then appeared frozen, with low
GPU utilization. Two causes:
1. *No progress output.* We only dumped the first 3 samples, then silently generated
   the remaining ~197 — many minutes of zero output that looked like a hang.
2. *Batch size 1.* We generated one prompt at a time. Autoregressive decoding of a
   single sequence is **kernel-launch-bound, not compute-bound**: the GPU spends most
   of its time idle between tiny operations. So it was genuinely slow *and* the GPU
   looked underused.

**Fix:** batched generation. We added `eval_batch_size` (16 on GPU) and decode many
prompts at once using the tokenizer's **left-padding** — with left-padding every
row's generated tokens start at the same column, so we can slice the shared prompt
width off cleanly (`out[:, prompt_len:]`). We also print a running
`done/total  running_acc  format_rate` line after each batch. Result: the GPU stays
busy, the run finishes far faster, and you can watch the accuracy estimate converge.

> *This is a preview of the Phase 7 insight:* generation is the slow part of RL, and
> batching is the first lever. We'll revisit it properly (vLLM, paged KV-cache) as
> the systems stretch goal.

*No conceptual bugs in the reward logic this phase* — the upfront test cases caught
nothing broken, which is exactly the point of writing them first. Extraction also
held up on the real model: Qwen emitted `<answer> $400 </answer>` and we correctly
parsed `400` (the `$` and spacing handled by `normalize_number`).

#### Checkpoint (what "done" looks like)

*CPU smoke (dev machine):*
```
$ python -m pytest tests/test_rewards.py -q
29 passed in 0.16s

$ python eval.py --config tiny
[eval] SmolLM2-135M on 8 test examples (greedy=True)
[eval] accuracy = 0.125  format_rate = 0.000   # tiny smoke model, not the baseline
```

*The real baseline (GPU box) — this is our "X%":*
```
$ python eval.py --config gpu --set eval_n=200
[eval] Qwen/Qwen2.5-1.5B-Instruct on 200 test examples (greedy=True)
...
[eval] 200/200  running_acc=0.555  format_rate=0.755
[eval] accuracy = 0.555  format_rate = 0.755
```

**📌 BASELINE (the "before" number we'll beat):**

| Metric | Value | Setup |
|--------|-------|-------|
| **Accuracy** | **55.5%** | Qwen2.5-1.5B-Instruct, 200 GSM8K test examples, greedy |
| **Format rate** | **75.5%** | fraction using `<think>/<answer>` correctly |

How to read this:
1. **55.5% is a healthy starting point.** Low enough that there's real room to
   improve, high enough that the task isn't hopeless for the base model. This is the
   X in our "X% → Y%" story.
2. **75.5% format rate is a useful second signal.** The instruct model *already*
   follows the format ~3 times out of 4 unprompted. So during training we can watch
   two things move separately: format compliance climbing toward ~100% (the easy
   win) and *correctness* climbing (the real win). Keeping the format reward small
   ensures the model can't fake progress on the second by maxing the first.
3. **Extraction held up on the real model** — `<answer> $400 </answer>` parsed to
   `400`, fraction-style reasoning didn't confuse it. No reward-side fixes needed.

**Status: ✅ Phase 1 done.** Baseline locked at **55.5% / 75.5%**. Next: Phase 2 —
the rollout engine, where we sample *groups* of answers and compute per-token
log-probabilities (the most bug-prone part of the whole project).

---

<!-- Phase 2 entry goes here -->
