# PPO Iteration Plan — Step-by-Step Guide

> **Status:** Experiments complete. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full results
> of 14 experiments across all three environments.
>
> The actual experimental progression diverged from this plan: instead of
> ablating each parameter individually on Simple and transferring, experiments
> focused on iterative improvement on Hard (the most challenging environment)
> and transferred the best config to Medium and Simple. Key findings:
> - **LR 5e-4 with linear decay** eliminated late-training instability
> - **32 parallel envs** smoothed gradients and stabilized training
> - **Batch size 256** produced the smoothest convergence curve
> - **Reward shaping** (all 3 strategies) hurt rather than helped
> - **Phases 2, 3, 6, 8** were not run as separate ablations; the iterative
>   approach on Hard covered learning rate, batch size, parallelism, and entropy

This is the original execution guide. Each phase has:
- The exact commands to run
- What to look for in the results
- What to write in your report

---

## Your Baseline Config (for reference)

```
learning_rate: 3e-4          clip_range: 0.2
n_steps: 2048                batch_size: 64
n_epochs: 10                 gamma: 0.99
gae_lambda: 0.95             ent_coef: 0.01 (simple), 0.015 (medium), 0.02 (hard)
net_arch: [128, 128]         vf_coef: 0.5
max_grad_norm: 0.5
```

Deviations from SB3 defaults: `net_arch` is [128,128] (SB3 default is [64,64]), `ent_coef` is 0.01 (SB3 default is 0.0).

---

## Phase 1: Establish Baseline ✓

> **Done.** Hard baseline (PPO_4): 100% WR at 2.2M steps, 1.9h. Medium baseline (PPO_1): 96% WR at 1M steps. See EXPERIMENTS.md #1, #12.

**Goal:** Get baseline win rates and diagnostics for all 3 environments.

### Commands
```bash
# Run baselines with seed 0 (one at a time, they use Unity)
python src/train.py --env simple --seed 0
python src/train.py --env medium --seed 0
python src/train.py --env hard   --seed 0

# Evaluate each final model
python src/evaluate.py --env simple --rounds 100
python src/evaluate.py --env medium --rounds 100
python src/evaluate.py --env hard   --rounds 100
```

### Record These Numbers
Fill in this table from your runs:

| Metric | Simple | Medium | Hard |
|--------|--------|--------|------|
| Final win rate (last 100 ep) | | | |
| Total episodes | | | |
| Average episode length | | | |
| Training time (min) | | | |
| Steps/sec | | | |

### Answer These Questions (you need them for later phases)
1. **Episode lengths:** Check `results/{env}/episode_log.json`, look at the `lengths` array. Are episodes ~50 steps? ~200? ~500? This affects Phase 3 (GAE lambda) and Phase 9 (gamma).
2. **Observation scales:** Run `python src/probe_env.py --env simple` and note the observation ranges. Are all 12 dimensions similar scale, or do some dominate? This affects Phase 8 (normalization).
3. **Baseline win rates:** How much room is there to improve? If Simple is already 90%, you're tuning. If it's 60%, there's real headroom.

### Write in Report
```
### Iteration 0: Baseline

**Change:** No changes. This is the starting point using default PPO configuration.
**Config:** lr=3e-4, clip=0.2, gae=0.95, ent=0.01, gamma=0.99, net=[128,128], 
            batch=64, n_steps=2048, n_epochs=10
**Result:** Simple: __% | Medium: __% | Hard: __%
**Observations:** [Describe training curves, how fast each env learns, 
                   any obvious patterns in the diagnostics]
**Decision:** Use this as the control for all subsequent experiments.
```

---

## Phase 2: Clip Range Ablation — Skipped

> **Not run as separate ablation.** Clip range was kept at the default 0.2 throughout all experiments. The iterative approach focused on learning rate, parallelism, and batch size instead. Clip_fraction was tracked as a diagnostic metric across all 14 experiments — see EXPERIMENTS.md for trends.

**Goal:** Understand PPO's core mechanism. This is the most important ablation.

**Hypothesis:** Tighter clip range (0.05) = more stable but slower. Wider (0.4) = faster but may oscillate. Diagnostics will show clip_fraction increasing as clip_range gets tighter.

### Commands
```bash
# Run all 6 clip values with 3 seeds each
python src/ablation.py --param clip_range --env simple --seeds 0 42 123

# Analyze results
python src/plot_ablation.py --param clip_range --env simple
```

### What to Look For
1. **Win rate plot** (`results/ablations/simple/clip_range/plots/win_rate_curves.png`): Which clip value learns fastest? Which gets the highest final win rate?
2. **clip_fraction plot** (`results/ablations/simple/clip_range/plots/diag_clip_fraction.png`): With clip=0.05, does the clip fraction stay high (>20%)? That means the clipping is actively preventing learning. With clip=0.4, is clip fraction near 0? That means clipping is irrelevant.
3. **approx_kl plot**: With wide clip range, does KL divergence spike? That means the policy is making dangerously large updates.

### Transfer Test
```bash
# Run best and worst clip values on Hard to see if the pattern holds
python src/ablation.py --param clip_range --env hard \
    --values [BEST_VALUE] [WORST_VALUE] --seeds 0
```

### Write in Report
```
### Iteration 1: Clip Range

**Change:** Varied clip_range from 0.05 to 0.4 (baseline: 0.2)
**Hypothesis:** [Your hypothesis]
**Result:** Best clip_range=__ (__%), worst=__ (__%)
**PPO Diagnostics:** [Describe the clip_fraction and approx_kl patterns. 
                      E.g., "clip=0.05 had 35% clip fraction, meaning a third 
                      of all gradient updates were being clipped away"]
**Learning:** [What does this teach about how PPO's clipping works?]
**Decision:** Set clip_range=__ for subsequent experiments.
```

---

## Phase 3: GAE Lambda Ablation — Skipped

> **Not run as separate ablation.** GAE lambda was kept at 0.95 throughout all experiments.

**Goal:** Understand the bias-variance tradeoff in advantage estimation.

**Hypothesis:** Low lambda (0.8) = biased but stable (like TD). High lambda (1.0) = unbiased but noisy (like Monte Carlo). The optimal value depends on episode length (which you measured in Phase 1).

### Commands
```bash
python src/ablation.py --param gae_lambda --env simple --seeds 0
python src/plot_ablation.py --param gae_lambda --env simple
```

### What to Look For
1. **Win rate curves**: Does lambda=0.8 converge faster but plateau lower? Does lambda=1.0 have noisier training?
2. **value_loss diagnostic**: Higher lambda should produce noisier value_loss because the advantage estimates have higher variance.
3. **Connection to episode length**: If your episodes are short (~50 steps), lambda probably doesn't matter much. If long (~500 steps), lambda=1.0 should help because it captures long-term reward signals.

### Write in Report
```
### Iteration 2: GAE Lambda

**Change:** Varied gae_lambda from 0.8 to 1.0 (baseline: 0.95)
**Hypothesis:** [Your hypothesis, informed by episode lengths from Phase 1]
**Result:** Best gae_lambda=__ (__%), worst=__ (__%)
**PPO Diagnostics:** [Describe value_loss patterns across lambda values]
**Learning:** [Relate optimal lambda to episode length — 
              "With average episode length of __ steps, ..." ]
**Decision:** Set gae_lambda=__ for subsequent experiments.
```

---

## Phase 4: Entropy Coefficient Ablation ✓

> **Partially done.** Tested ent_coef=0.01 vs baseline 0.02 in EXPERIMENTS.md #4 (hard_low_entropy). Finding: entropy tuning was not the bottleneck — the constant high LR dominated the instability. Kept ent_coef=0.02 for the optimized config.

**Goal:** Find the exploration-exploitation sweet spot. Hunt for the "entropy cliff."

**Hypothesis:** ent_coef=0 causes premature policy collapse. ent_coef=0.1 prevents convergence. The sweet spot is somewhere in 0.005-0.02.

### Commands
```bash
# Run all 7 values with 3 seeds
python src/ablation.py --param ent_coef --env simple --seeds 0 42 123
python src/plot_ablation.py --param ent_coef --env simple
```

### What to Look For
1. **Entropy decay curves** (`diag_entropy_loss.png`): With ent_coef=0, entropy should drop to near-zero very fast. With ent_coef=0.1, entropy should stay high. Look for the "entropy cliff" — a sharp drop where the agent commits to a strategy.
2. **When does the cliff happen?** If ent_coef=0 causes the cliff at 100k steps and ent_coef=0.01 causes it at 500k steps, that tells you the entropy coefficient controls HOW LONG the agent explores before committing.
3. **Win rate vs entropy coef**: Plot the final win rate against entropy coef. You should see an inverted U shape — too low and too high both hurt.

### Write in Report
```
### Iteration 3: Entropy Coefficient

**Change:** Varied ent_coef from 0.0 to 0.1 (baseline: 0.01)
**Hypothesis:** [Your hypothesis]
**Result:** Best ent_coef=__ (__%), worst=__ (__%)
**PPO Diagnostics:** [Describe entropy decay curves. When does the "cliff" happen 
                      for each value? Quote specific step numbers.]
**Learning:** [What does this teach about exploration in PPO? 
              E.g., "Entropy coefficient doesn't just control how much the agent 
              explores — it controls how LONG it explores before committing."]
**Decision:** Set ent_coef=__ for subsequent experiments.
```

---

## Phase 5: Learning Rate ✓

> **Done.** Tested constant 5e-4 (#2), constant 5e-4 + parallel (#3), and 5e-4 with linear decay (#5). Key finding: constant high LR causes late-training KL divergence spikes (approx_kl up to 0.036); linear decay eliminates this while keeping the 2x early speedup. See EXPERIMENTS.md #2, #3, #5.

**Goal:** Compare constant vs decaying learning rate.

**Hypothesis:** High LR (1e-3) = fast early, unstable late. Linear decay = best of both worlds.

### Commands
```bash
# 4 constant LR values
python src/ablation.py --param learning_rate --env simple --seeds 0 42 123

# Linear decay
python src/ablation.py --param lr_schedule --env simple --seeds 0 42 123

# Analyze both
python src/plot_ablation.py --param learning_rate --env simple
python src/plot_ablation.py --param lr_schedule --env simple
```

### What to Look For
1. **Late training divergence**: Compare constant LR=3e-4 vs linear decay. In the last 25% of training, does the decay version converge more smoothly?
2. **approx_kl**: Larger LR should produce larger KL per update (bigger policy changes).

### Write in Report
```
### Iteration 4: Learning Rate

**Change:** Tested 4 constant LR values + linear decay schedule
**Hypothesis:** [Your hypothesis]
**Result:** Best constant LR=__ (__%), linear decay=__ (__)
**Learning:** [Does decay help? By how much? When does the gap open up?]
**Decision:** Use [constant/decay] LR=__ for subsequent experiments.
```

---

## Phase 6: Network Architecture — Skipped

> **Not run as separate ablation.** Network was kept at [128, 128] throughout all experiments. The 12-dimensional observation space did not warrant larger architectures.

**Goal:** Right-size the network for the task.

**Hypothesis:** 12-dim observation space doesn't need a huge network. [64,64] might be sufficient for Simple.

### Commands
```bash
python src/ablation.py --param net_arch --env simple --seeds 0
python src/plot_ablation.py --param net_arch --env simple
```

### What to Look For
1. **Parameter efficiency**: If [64,64] matches [256,256] on Simple, the environment isn't complex enough to need a big network.
2. **Training speed**: Smaller networks train faster (more steps/sec). Note the speed difference.
3. **Think ahead**: Even if [64,64] works on Simple, Hard might need more capacity.

### Write in Report
```
### Iteration 5: Network Architecture

**Change:** Tested [64,64], [128,128], [256,256], [128,128,128]
**Hypothesis:** [Your hypothesis]
**Result:** [Performance per architecture + training speed]
**Learning:** [What does this say about the environment's complexity?]
**Decision:** Use net_arch=__ for subsequent experiments.
```

---

## Phase 7: Batch Size and N-Steps ✓

> **Batch size done.** Tested batch_size=256 in EXPERIMENTS.md #6 (hard_large_batch). Produced the smoothest monotonic convergence curve and fastest first 100% WR (1.31M steps). N-steps was kept at 2048 throughout. See EXPERIMENTS.md #6.

**Goal:** Understand the gradient quality vs speed tradeoff.

### Commands
```bash
python src/ablation.py --param batch_size --env simple --seeds 0
python src/ablation.py --param n_steps   --env simple --seeds 0
python src/plot_ablation.py --param batch_size --env simple
python src/plot_ablation.py --param n_steps   --env simple
```

### What to Look For
1. **batch_size**: Smaller batches = noisier gradients. Does this help or hurt?
2. **n_steps**: More steps before update = better advantage estimates but fewer updates per wall-clock second.
3. **Wall-clock time**: Note the training time for each configuration. Sometimes a setting is slightly better per-step but much slower in total.

### Write in Report
```
### Iteration 6: Batch Size and Rollout Length

**Change:** Tested batch_size {32,64,128,256} and n_steps {1024,2048,4096}
**Result:** [Best batch_size, best n_steps, with wall-clock context]
**Learning:** [Tradeoff between update quality and update frequency]
**Decision:** Use batch_size=__, n_steps=__ for subsequent experiments.
```

---

## Phase 8: Observation Normalization — Skipped

> **Not implemented.** VecNormalize wrapper was not added. Positions are already normalized by arenaHalfSize in the Unity source. Velocities/angular velocities are raw but the agent learned effectively without normalization.

**Goal:** See if normalizing observations gives a free boost.

### Commands
```bash
# This requires manual setup — VecNormalize wrapping
# See implementation note in the design doc about saving/loading stats
# You may need to modify ablation.py or run a custom training script

python src/train.py --env simple --seed 0
# TODO: add VecNormalize wrapper and compare
```

### Write in Report
```
### Iteration 7: Observation Normalization

**Change:** Added VecNormalize wrapper to normalize observations
**Hypothesis:** Different obs dimensions have different scales, normalization helps
**Result:** Before: __% → After: __%
**Learning:** [Did it help? Which dimensions had the most different scales?]
**Decision:** [Keep/remove normalization for final config]
```

---

## Phase 9: N-Epochs and Gamma — Skipped

> **Not run as separate ablations.** Both n_epochs=10 and gamma=0.99 were kept at defaults throughout. The iterative experiments focused on the parameters with the largest observed impact (LR, parallelism, batch size).

**Goal:** Complete the PPO parameter coverage.

### Commands
```bash
# N-epochs ablation
python src/ablation.py --param n_epochs --env simple --seeds 0
python src/plot_ablation.py --param n_epochs --env simple

# Gamma ablation
python src/ablation.py --param gamma --env simple --seeds 0
python src/plot_ablation.py --param gamma --env simple
```

### What to Look For — N-Epochs
- **KL accumulation**: With n_epochs=20, does approx_kl spike? That means PPO's trust region is being violated through accumulated drift across many optimization steps.
- **Sample efficiency**: More epochs = fewer environment interactions needed. Is it worth it?

### What to Look For — Gamma
- **Connect to episode length**: If episodes are short, gamma=0.95 and gamma=0.99 should perform similarly. The difference only matters for long-horizon rewards.

### Write in Report
```
### Iteration 8: N-Epochs and Gamma

**Change:** Tested n_epochs {3,5,10,15,20} and gamma {0.9,0.95,0.99,0.999}
**Result:** Best n_epochs=__, best gamma=__
**PPO Diagnostics:** [approx_kl patterns for n_epochs — does KL spike with many epochs?]
**Learning:** [What does n_epochs teach about PPO's trust region? 
              What does gamma teach about the game's time horizon?]
**Decision:** Use n_epochs=__, gamma=__ for the assembled config.
```

---

## Phase 10: Assembly + Transfer ✓

> **Done.** Best config from Hard (lr=5e-4 decay, batch=256, 32 envs) transferred directly to Medium and Simple. All three reached 100% WR. Convergence scales with difficulty: Simple 262K, Medium 918K, Hard 1.31M. See EXPERIMENTS.md #13 (medium_optimized), #14 (simple_optimized).

**Goal:** Combine the best values and test on all environments.

### Step 1: Assemble the Optimized Config
Fill in the best value from each phase:

| Parameter | Baseline | Best from Ablation | Phase |
|-----------|----------|-------------------|-------|
| clip_range | 0.2 | | 2 |
| gae_lambda | 0.95 | | 3 |
| ent_coef | 0.01 | | 4 |
| learning_rate | 3e-4 | | 5 |
| lr_schedule | constant | | 5 |
| net_arch | [128,128] | | 6 |
| batch_size | 64 | | 7 |
| n_steps | 2048 | | 7 |
| n_epochs | 10 | | 9 |
| gamma | 0.99 | | 9 |

### Step 2: Test the Combined Config
```bash
# Run the assembled config on Simple first
python src/train.py --env simple --seed 0 \
    --lr [BEST_LR]
# (Manually set other params in config.py or pass via ablation.py)

# Then on Medium and Hard
python src/train.py --env medium --seed 0
python src/train.py --env hard   --seed 0

# Evaluate all
python src/evaluate.py --env all --rounds 100
```

### Step 3: Check for Interaction Effects
If the combined Simple win rate dropped >5% compared to individual ablation results, the parameters are interacting. Test the two most likely pairs:
- clip_range x n_epochs (clipping + sample reuse)
- ent_coef x learning_rate (exploration + step size)

### Step 4: Fill in the Transfer Matrix

| Parameter | Simple Best | Works on Medium? | Works on Hard? | Adjustment Needed |
|-----------|------------|-------------------|----------------|-------------------|
| clip_range | | | | |
| ent_coef | | | | |
| learning_rate | | | | |
| net_arch | | | | |

### Write in Report
```
### Iteration 9: Config Assembly and Transfer

**Change:** Combined best values from all ablations into one config
**Result:** Simple: __% (was __% baseline) | Medium: __% | Hard: __%
**Learning:** [Which params transferred across environments? Which didn't? Why?]
**Decision:** [List which params need environment-specific adjustment]
```

---

## Phase 11: Final Fine-Tuning ✓

> **Done.** No per-environment tuning was needed — the same optimized config worked across all three. Final models: models/hard/PPO_16/final.zip, models/medium/PPO_2/final.zip (or models/medium/final.zip), models/simple/PPO_1/final.zip (or models/simple/final.zip). ONNX exports available in models/onnx/.

**Goal:** Produce the best agent for each environment.

### Commands
```bash
# Adjust the 2-3 params that didn't transfer well to Hard
# Run with the final tuned configs
python src/train.py --env medium --seed 0
python src/train.py --env hard   --seed 0

# Final evaluation
python src/evaluate.py --env all --rounds 100
```

### Write in Report
```
### Iteration 10: Environment-Specific Tuning

**Change:** Adjusted [specific params] for Medium and Hard
**Result:** 
  Simple:  __% (baseline __%, improvement __%)
  Medium:  __% (baseline __%, improvement __%)
  Hard:    __% (baseline __%, improvement __%)
**Learning:** [The story arc — "We started with defaults, understood each 
              PPO component on Simple, assembled the best config, discovered 
              it partially fails on Hard because [reason], and adjusted 
              [params] to produce the final agents."]
```

---

## Summary Table (for your report)

Fill this in as you go. This becomes the capstone of your iteration document.

| Iteration | Parameter | Best Value | Win Rate (Hard) | vs Baseline | Key Insight |
|-----------|-----------|------------|-----------------|-------------|-------------|
| 0 | Baseline | — | 100% at 2.2M | — | 260K dead zone, diminishing returns after 1.4M |
| 1 | learning_rate | 5e-4 | 90% at 1M | 2x speedup | Constant high LR causes late instability |
| 2 | n_envs | 32 | 100% at 1.4M | -38% steps | Smoothed gradients, prevented catastrophic drops |
| 3 | ent_coef | 0.02 (no change) | 100% at 1.5M | minimal | Not the bottleneck; LR dominates |
| 4 | lr_schedule | linear_decay | 100% at 1.4M | -38% steps | Late KL dropped from 0.032 to 0.015 |
| 5 | batch_size | 256 | 100% at 1.3M | -41% steps | Monotonic convergence, smoothest curve |
| 6-8 | reward_shaping | none (baseline) | 98-100% | worse | All 3 strategies hurt or added noise |
| 9 | assembled config | all combined | 100% at 1.31M | -41% steps | Reproducible across 3 runs |
| 10 | transfer | same config | 100% all envs | — | Simple 262K, Medium 918K, Hard 1.31M |

---

## Quick Reference: Commands Cheat Sheet

```bash
# === ABLATION ===
python src/ablation.py --list                                    # See all params
python src/ablation.py --param PARAM --env simple --dry-run      # Preview runs
python src/ablation.py --param PARAM --env simple --seeds 0      # Single seed
python src/ablation.py --param PARAM --env simple --seeds 0 42 123  # 3 seeds

# === ANALYSIS ===
python src/plot_ablation.py --param PARAM --env simple           # Plots + table
python src/plot_ablation.py --all --env simple                   # All ablations

# === TRAINING ===
python src/train.py --env simple --seed 0                        # Basic training
python src/train.py --env hard --seed 0 --timesteps 4000000      # Override steps

# === EVALUATION ===
python src/evaluate.py --env simple --rounds 100                 # Evaluate one
python src/evaluate.py --env all --rounds 100                    # Evaluate all
python src/evaluate.py --env simple --rounds 10 --demo           # Watch it play

# === RESULTS ===
# Training logs:    results/{env}/
# Ablation results: results/ablations/{env}/{param}/
# Ablation plots:   results/ablations/{env}/{param}/plots/
# Models:           models/{env}/final.zip
```

---

## Tips

1. **Run Phase 1 first and fill in the numbers.** Everything downstream depends on baseline results and episode length observations.
2. **Analyze as you go.** After each ablation, immediately run `plot_ablation.py` and write the report section while the results are fresh.
3. **Null results are fine.** If a parameter doesn't matter (e.g., batch_size barely affects win rate), say so and explain why.
4. **The diagnostics are the differentiator.** Win rate tables are boring. The clip_fraction and entropy_loss plots are what make this portfolio-grade.
5. **Save your best models.** The `models/{env}/final.zip` from your best runs are your demo agents.
