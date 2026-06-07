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

### Phase 2 — The rollout engine

> "Rollout" = letting the model generate answers. In RL we **make our own training
> data**: the model's own samples are what we learn from. This phase builds the
> machine that (a) samples a *group* of answers per question and (b) records, for
> every generated token, how confident the model was — its **log-probability**.
> That per-token confidence is the exact quantity GRPO will push up or down.
>
> The plan flags this as *the most bug-prone phase*. Most "loss is NaN / nothing
> learns" disasters are actually a silent log-prob or masking bug here. So we wrote
> the tests first and pinned the alignment down to the index.

#### What we did

| File | What it is, in plain terms |
|------|----------------------------|
| `rollout.py` | Samples G completions per question, then scores each generated token with its log-probability. Returns a tidy `Rollout` bundle (token ids, masks, log-probs, rewards, decoded text). |
| `tests/test_rollout.py` | 7 fast tests (no model download) that nail the two danger zones: the logit→token *shift* and the *completion mask*. |

#### The two ideas you must get exactly right

**1. The logit→token "off-by-one" shift.** A language model reads tokens
left-to-right and, at each position, predicts the *next* token. So the model's
opinion about the token at position `t+1` lives in its output at position `t`. If
you pair them up wrong by one slot, every log-prob is attributed to the wrong token
and training quietly learns nonsense.

We handle it with one clear convention, the **scored frame**: for a sequence of
length `T` there are `T-1` scored positions (you can't score the very first token —
nothing comes before it). Scored slot `k` holds *"how likely was the token at full
position `k+1`, given everything up to `k`."* In code:
```python
logits  = model(...).logits[:, :-1, :]   # drop the last step (predicts past the end)
targets = input_ids[:, 1:]               # drop the first token (unscoreable)
logp[k] = log_softmax(logits[k])[targets[k]]
```

**2. The completion mask.** We must train **only on tokens the model generated** —
never the prompt (the model didn't choose those), never padding, and nothing after
the end-of-sequence (EOS) token. So we build a mask that is `1` exactly on real
generated tokens and `0` everywhere else, and *every* later sum/average over tokens
is taken under it. The mask is built in the same scored frame so it lines up with
the log-probs element-for-element. That alignment is the whole game.

#### How it's used later
- `generate_group` is called at the top of every Phase 4 training step to produce
  the batch of completions we learn from.
- The log-probs it records become **`logp_old`** — the "before" snapshot. In Phase 3
  the loss compares the *current* policy's fresh log-probs against this `logp_old` to
  decide how much each token's probability changed.
- `rewards` (per completion) feed the group-advantage computation in Phase 3.
- The `completion_mask` is reused in every masked mean inside the loss.

#### Design choices worth noting
- **Group-contiguous batch layout.** Rows are ordered `[prompt0×G, prompt1×G, ...]`
  via repeat-interleave, so Phase 3 can reshape to `[num_prompts, G]` and compute
  per-group statistics with no bookkeeping.
- **EOS-aware mask, keeping the EOS token.** After a row emits EOS, `generate` fills
  the rest with padding; we mask those out but *keep the EOS token itself* trainable
  (learning *when to stop* matters).
- **Memory-frugal log-probs.** Instead of materialising a full
  `[batch, length, vocab]` log-softmax (which for Qwen's ~150k vocab can be tens of
  GB), we compute `selected_logit − logsumexp(logits)`. Same answer, a fraction of
  the memory. We also cast logits to fp32 before the `logsumexp` for numerical safety
  even when the model runs in bf16.
- **`compute_logprobs` is deliberately *not* `no_grad`.** Phase 2 calls it inside
  `no_grad` (the old policy is fixed), but Phase 3 calls the *same* function through
  the trainable policy and needs gradients to flow. So the function stays
  grad-agnostic and the caller decides.

#### Problems hit & how we fixed them
*No bugs survived to runtime* — which is the point of writing the tests first. The
trickiest part conceptually was deciding **where** to apply the shift so the mask and
log-probs stay aligned. We resolved it by committing to a single "scored frame"
convention (documented at the top of `rollout.py`) and adding
`test_full_to_scored_frame_alignment`, which proves that the first generated token
(full position = `prompt_len`) maps to the right scored index after the shift.

#### Checkpoint (what "done" looks like)
```
$ python -m pytest tests/ -q
36 passed in 5.11s          # 29 reward + 7 rollout

$ python rollout.py --config tiny --set group_size=4
[rollout] 1 prompt x G=4 -> 4 rows
[rollout] full_ids (4, 193)  logp_old (4, 192)  completion_mask (4, 192)
--- completion 0 | reward=0.00 correct=0 len=64 tokens ---
To find the cost of the computer, ...
... (4 genuinely different samples of the same question) ...
[rollout] masked scored tokens = 256  (== sum of completion lengths = 256)
[rollout] OK
```
Read the output:
1. **Shapes line up:** `full_ids` is length 193, and `logp_old`/`completion_mask`
   are 192 = `T-1`. The scored frame is exactly one shorter, as designed.
2. **The invariant holds:** masked token count (256) equals the summed completion
   lengths (4 × 64). If these ever disagree, the mask is wrong — this is our
   tripwire.
3. **Sampling explores:** the 4 completions are visibly different reasonings for the
   same question — exactly what we need so a *group* has variety to rank.
4. Rewards are all 0 here (tiny model, wrong answers) — expected; we're testing the
   plumbing, not the model.

**Status: ✅ Phase 2 done.** The rollout engine samples groups and scores tokens with
verified alignment. Next: Phase 3 — turn rewards + log-probs into the **GRPO loss**
(group-relative advantages + the clipped policy-gradient objective + KL). This is the
algorithm itself, and almost all of it is checkable on CPU against hand-computed
numbers.

---

### Phase 3 — Advantages & the GRPO loss (the core)

> This is the phase. Everything before it was gathering ingredients — questions,
> a scorer, sampled answers with their per-token confidences. Here we combine them
> into **one number, the loss**, whose gradient *is* the GRPO learning rule. It's
> also the part you'll explain in an interview, so the file is written to be read.
>
> The beautiful thing: it's pure tensor math, so we can check every line against
> numbers worked out by hand on CPU — no GPU, no model, no randomness.

#### What we did

| File | What it is, in plain terms |
|------|----------------------------|
| `grpo.py` | Two functions. `group_advantages`: turn each completion's reward into "how much better than your group were you?". `grpo_loss`: turn those advantages + the log-probs into a single scalar loss to backprop. |
| `tests/test_grpo.py` | 12 tests, each pinned to a hand-computed value: the advantage formula, the ratio=1 identity, the KL estimator, one-sided clipping, and masking. |

#### The algorithm in four moves (plain English)

**Move 1 — Group-relative advantage (GRPO's whole trick).**
For a group of G answers to the *same* question, the group's **average reward is the
baseline**. Each answer's advantage is how far above/below that average it landed,
scaled by the group's spread:
```
A_i = (reward_i − group_mean) / (group_std + ε)
```
Beat your siblings → positive advantage → "do more of this." Lagged → negative →
"do less." No critic network, no value function — *the siblings are the baseline.*
That's why GRPO is cheap. A subtle but important case: if every answer in a group
ties (all right or all wrong), the advantage is **0** for all of them — and that's
correct, because the group gives no information about which answer was better.

**Move 2 — The policy ratio.**
We sampled the answers with one version of the model (call its confidences
`logp_old`). After a gradient step the model changes; we re-score the same tokens to
get `logp_new`. The **ratio** `exp(logp_new − logp_old)` says, per token, "how much
more likely is the *new* model to say this than the old one?" (Exponentiating a
difference of logs turns it back into a probability ratio.)

**Move 3 — The clipped surrogate (borrowed from PPO).**
We want to increase the probability of tokens with positive advantage. Naively
multiplying `ratio × advantage` lets one batch shove the model too far. PPO's fix:
```
min( ratio·A ,  clip(ratio, 1−ε, 1+ε)·A )
```
The clip caps how much a single update can reward a token. Crucially it's
**one-sided in effect**: it stops us *over-shooting* on good tokens, but it does not
soften the push *away* from bad ones — exactly the asymmetry we verified in a test.

**Move 4 — The KL leash.**
Chasing reward can make a model degenerate (e.g. spam one phrase). So we add a
penalty for drifting from a frozen **reference** model. We use the **k3 estimator**
`exp(Δ) − Δ − 1` (with `Δ = logp_ref − logp_new`), which is **always ≥ 0** and
low-variance — much better behaved than the naive `logp_new − logp_ref`, which can
go negative and destabilize training.

**Putting it together:** per-token loss = `−(clipped surrogate) + β·KL`, then
**averaged only over real generated tokens** (the mask from Phase 2). We minimise it;
the minus sign is because we *maximise* the surrogate objective.

#### How it's used later
- Phase 4's training loop calls `group_advantages(rollout.rewards, G)` then
  `grpo_loss(logp_new, rollout.logp_old, advantages, rollout.completion_mask, ...)`,
  calls `.backward()`, and steps the optimizer. That's the entire update.
- The returned **metrics dict** (mean ratio, mean KL, clip fraction, mean advantage)
  becomes the per-step logging in Phase 4–5 that tells us whether training is healthy
  (e.g. KL not exploding, clip fraction reasonable).
- The `normalize` flag (Move 1) and `loss_agg` choice (below) are wired to config so
  Phase 6 can run the Dr. GRPO / DAPO ablations *without touching this code*.

#### Design choices worth noting
- **Advantage is one scalar per completion, broadcast to all its tokens.** GRPO is
  *outcome-supervised*: the whole answer earns one reward, and every token in it
  shares the credit/blame. We broadcast `[B] → [B, 1] → [B, L]` at loss time.
- **Token-averaging is a real, named choice.** `loss_agg="seq"` (default) averages
  per sequence then over sequences — the *original* GRPO. `loss_agg="token"` does one
  global mean over all tokens — the **DAPO** fix. They differ in how much long answers
  are weighted, the exact bias **Dr. GRPO** critiques. We expose both and will ablate
  in Phase 6 rather than silently baking one in.
- **Memory-frugal, numerically safe.** Metrics are computed under `no_grad`; the KL
  term is skipped entirely when `kl_beta=0` (the no-KL ablation) so we don't pay for a
  reference forward pass we won't use.

#### Problems hit & how we fixed them
No runtime bugs — the hand-computed tests are exactly so that "looks right" becomes
"is right." The one genuinely *important decision* (not a bug) was the **token-
averaging scheme**. It's tempting to pick one and move on, but it materially changes
the gradient and is the subject of active research (DAPO, Dr. GRPO). We resolved it by
making it a parameter with the original-GRPO behaviour as default, and writing it down
here so future-us remembers it's a lever, not a constant.

#### Checkpoint (what "done" looks like)
```
$ python -m pytest tests/ -q
48 passed in 7.05s          # 29 reward + 7 rollout + 12 grpo

# gradient-flow sanity (2 prompts x G=3, one masked token):
loss 0.00218
metrics {'mean_ratio': 1.0007, 'mean_kl': 0.0, 'clip_frac': 0.0, 'mean_adv': 0.0}
grad on logp_new finite: True
grad nonzero on masked token [0,3]: -0.0      # masking kills the gradient here
grad sum abs: 0.9417                          # real learning signal everywhere else
```
What this proves:
1. **Every formula matches hand arithmetic** — advantages (`(3,1,1,−1) → (√2,0,0,−√2)`),
   the ratio=1 identity (loss = −mean advantage), the k3 KL (`e^0.5−0.5−1`), and the
   one-sided clip all hit their expected values.
2. **`mean_adv = 0`** — advantages are zero-mean within each group, as the baseline
   subtraction requires. (Sanity that we grouped correctly.)
3. **The loss actually backprops** with finite gradients, and the **masked token gets
   exactly zero gradient** — so the prompt/padding can never leak into the update.

**Status: ✅ Phase 3 done.** The algorithm is implemented and verified in isolation.
Next: Phase 4 — wire rollout + rewards + loss into a real **training loop**, add the
reference model (LoRA-adapters-disabled trick), and run the make-or-break **overfit
test**: can we drive reward up on 10–50 problems? That's the first time we'll need the
GPU again.

---

### Phase 4 — The training loop, end to end

> Now the pieces meet. This phase wires rollout → rewards → advantages → loss →
> `backward()` → optimizer step into one loop, adds the reference model for KL, and
> runs the **make-or-break sanity check**: can we drive reward *up* on a tiny set of
> problems? If yes, the algorithm is correct and we're allowed to scale. If no, the
> bug is upstream in Phase 2/3 and scaling would just waste GPU money.

#### What we did

| File | What it is, in plain terms |
|------|----------------------------|
| `train.py` | The loop. Each step: sample a group of answers, score them, compute advantages, re-score under the current model, compute the GRPO loss, backprop, clip, step. Logs every metric and saves the trained weights at the end. |
| `utils.trainable_parameters` | Returns the params the optimizer should touch (all of them, or just the LoRA adapters). |

#### Key concepts (plain English)

**The reference model — and the LoRA trick.** The KL term needs a *frozen* copy of
the original model to measure drift. The naive way is to keep a second full model in
memory (expensive). The trick: with **LoRA**, training only touches small add-on
"adapter" weights while the big base model stays frozen. So the reference model is
just *the policy with its adapters switched off* — one line, `with
policy.disable_adapter():`, and zero extra memory. (On the CPU `tiny` config we don't
use LoRA, so there we *do* load a small second copy — fine for a 135M model.)

**Why we re-score the same tokens we just generated.** During the rollout we recorded
`logp_old` with no gradients (it's a fixed snapshot). To actually *learn*, we run the
tokens through the model **again, with gradients on**, producing `logp_new`. Only this
second pass builds the computation graph that `backward()` needs. Since no optimizer
step has happened between sampling and re-scoring, `logp_new == logp_old` and the
ratio is exactly 1 on each update — which is the correct, expected behaviour for one
gradient step per rollout (the clip simply doesn't engage in this regime).

**Gradient accumulation & clipping.** We scale each step's loss by
`1/grad_accum_steps` and only call `optimizer.step()` every N steps — this simulates a
bigger batch than fits in memory (GRPO is more stable with larger, less-noisy group
statistics). We also clip the gradient norm so one weird batch can't blow up the
weights.

**The overfit sanity check.** `--overfit` shrinks the data to a tiny fixed pool
(default 16 problems) and trains on it repeatedly. On a *capable* model the reward
should climb fast — that's proof the gradient points the right way. It's the cheapest
possible "is the whole thing correct?" test before a real run.

#### How it's used later
This same loop *is* Phase 5 — the real run is just `train.py --config gpu` (no
`--overfit`) with a bigger pool and more steps. Phase 6's ablations are this loop with
different config flags (`kl_beta=0`, `normalize_advantage=false`, `loss_agg=token`,
varying `group_size`). The saved `runs/<name>/final` weights are what `eval.py` loads
to produce the trained accuracy.

#### Design choices worth noting
- **One update per rollout (ratio≈1).** We keep the simplest correct form: sample,
  take one gradient step, resample. (Doing multiple inner epochs per rollout is what
  would make the clip actually bite — a possible later extension.)
- **KL term is fully optional.** `kl_beta=0` skips the reference forward pass entirely
  — both a speed win and the Phase 6 no-KL ablation, with no code change.
- **A built-in profiler.** Every step logs `t_rollout` vs `t_update`. Even on CPU you
  can see generation is a big chunk of the time — the seed of the Phase 7 story.

#### Problems hit & how we fixed them
No crashes — the loop ran clean on the first try, which is the payoff of having tested
Phases 2 and 3 in isolation. The main thing to *understand* (not fix) is the output:
**`loss` prints as ~0 while gradients are clearly non-zero.** That's not a bug. With
ratio = 1 and advantages that are zero-mean within each group, the loss *value* is
≈ 0, but its *gradient* — `−mean(advantage · ∇logp_new)` — is not. **Loss value ≈ 0
does not mean gradient ≈ 0.** This trips up everyone the first time.

#### Checkpoint (what "done" looks like)

*CPU smoke (dev machine) — the loop runs end-to-end with no NaNs:*
```
[train] SmolLM2-135M | steps=3 prompts/step=2 G=2 pool=8 kl=on overfit=True
[step 0] reward=0.000  frac_correct=0.000  ratio=1.000  grad_norm=0.0000  ...
[step 1] reward=0.250  frac_correct=0.250  ratio=1.000  grad_norm=2.2839  ...
[step 2] reward=0.000  frac_correct=0.000  ratio=1.000  grad_norm=0.0006  ...
[train] done. saved to runs/tiny/final
```
This tiny run accidentally became the **clearest possible demo of GRPO's group
baseline**:
- **Step 1 has a real gradient (`grad_norm=2.28`); steps 0 and 2 have ~zero.** Why?
  Step 1's group of answers was *mixed* (one of four correct → `frac_correct=0.25`),
  so some advantages were positive and some negative → a learning signal. Steps 0 and
  2 had *all answers wrong* → every reward tied → all advantages zero → **no gradient**.
  That is GRPO working exactly as designed: *you only learn from a question when the
  group disagrees about it.*
- **`ratio=1.000` throughout** — confirms the on-policy single-update regime.
- **No NaNs, weights saved.** The plumbing is correct.

**⏳ Still pending — the real overfit test (GPU).** The 135M CPU model can't actually
*get better* in 3 steps; "reward climbs" must be shown on the capable model. The gate
to pass before Phase 5 is:
```
python train.py --config gpu --overfit --set max_steps=60
```
and watch `reward` / `frac_correct` trend **upward** over steps with KL staying
bounded. (Memory levers if it OOMs on a 24 GB card: lower `prompts_per_step`,
`group_size`, or `max_new_tokens`.) We'll record that curve here when it's run.

**Status: ✅ Phase 4 loop built & CPU-verified; ⏳ GPU overfit gate pending.** Next,
once reward is confirmed climbing: Phase 5 — the real training run on full GSM8K.

---

<!-- Phase 5 entry goes here -->
