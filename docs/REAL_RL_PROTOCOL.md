# Real Minecraft RL protocol

The existing Qwen pilot is an inference-only rollout.  This protocol defines
the boundary for a genuine train/test experiment: a train episode must reset a
real Minecraft world, execute typed VillagerAgent actions, wait for the world
checker, and persist its observation/action/state-delta trace and terminal
score.  A symbolic replay is useful for plumbing tests, but it is tagged
`real_rollout=false` and is rejected by both the learner and the real evaluator.

## Episode record

`masbench.minecraft.rl.RealMinecraftRLEnv` records one observation for every
agent and tick, the model's raw typed action, the Mineflayer execution result,
and the observed state delta.  A terminal record contains:

- `backend`, `real_rollout`, `task_id`, `split`, `topology_id`, and the episode
  id;
- the immutable `steps` list;
- the raw checker result and normalized `evaluation_status` in `score`.

`evaluation_status=unassessed` is retained for debugging and is never treated
as either success or failure.  A failed evaluated episode has `success=false`,
`reward=0`, and `completion_ticks=null`.  Completion time is written only for
checker-successful episodes.

## Reward used by training and reported reward

The benchmark reward is terminal and binary: `1` for a checker-successful
episode and `0` for a checker-evaluated failure.  The runner may use dense
shaping while learning (small progress and valid-action terms), but it stores
that as `shaped_return`; it never replaces the terminal checker reward and it
never enters the reported SR/composite metric.

For the real VillagerAgent checker, fractional `block_hit_rate` and
`view_hit_rate` values are clipped to `[0, 1]` before they are used as an
optional progress potential.  The official veRL Minecraft worker keeps the
environment reward sparse (`0` until termination) and exposes malformed JSON,
unknown action types, and missing movement coordinates through
`is_action_valid`; this auxiliary penalty cannot turn a failed world check
into a success.

`OnPolicyGRPOLearner` groups candidate episodes by task, centers and scales
terminal returns, then passes the resulting advantages to the injected policy
update.  `algorithm=reinforce` uses the same policy hook with a single-return
batch; `algorithm=grpo` expects multiple candidates per task for a meaningful
within-task baseline.

## Train and test separation

The training CLI reads only `split=train`.  The injected adapter factory must
set `real_rollout=True`; an API-only Qwen endpoint is a policy inference
baseline and cannot update weights.  A trainable policy factory should load a
local Qwen checkpoint (typically a LoRA/PEFT adapter), and its
`save_checkpoint` must persist weights, tokenizer, optimizer state, prompt
version, environment/checker hashes, and the train manifest.

For final evaluation, freeze the checkpoint and reset fresh worlds from the
held-out `split=test` topology/seed pool.  Run each episode once, wait for the
checker, and aggregate `score.json` with:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/eval_real_rl.py \
  MASbench/results/minecraft_rl_eval --out MASbench/results/minecraft_rl_eval.json
```

The evaluator refuses synthetic, missing, or unassessed records.  This makes
it impossible for an offline JSON plan, a connection smoke, or an interrupted
checker to silently enter SR.

## Plumbing dry-run

The dry-run is intentionally not an experiment:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/train_real_rl.py \
  --dry-run --out /tmp/masbench_rl_dry
```

It exercises trajectory serialization with the CPU symbolic adapter and emits
`real_rollout=false`; `eval_real_rl.py` must reject that directory.  A real run
requires both an adapter and a trainable policy factory:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/train_real_rl.py \
  --adapter-factory my_runner:make_adapter \
  --policy-factory my_qwen:make_policy \
  --algorithm grpo --candidates 4 --batch-size 8 \
  --out MASbench/results/minecraft_rl_train
```

No command above claims a Minecraft result until an actual server checker has
written an evaluated `score.json`.

## Bench2 and Bench3 use the same boundary

Bench2 workflow and Bench3 incident runs use
`masbench.longhorizon.rl.RealLongHorizonRLEnv`,
`scripts/train_real_bench23_rl.py`, and
`scripts/eval_real_bench23_rl.py`. Their adapter must connect to the actual
workflow or incident service and declare `real_rollout=True`. The train CLI
reads only `data/bench2_workflow/train.jsonl` or
`data/bench3_incident/train.jsonl`; the eval CLI reads only the corresponding
`test.jsonl` and uses a frozen policy checkpoint. Symbolic `WorkflowEnv` and
`IncidentEnv` runs remain useful for contract tests, but their records are
rejected by the real writer and evaluator.

Example commands:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/train_real_bench23_rl.py \
  --bench all --adapter-factory my_backend:make_adapter \
  --policy-factory my_policy:make_policy --batch-size 4 \
  --out MASbench/results/bench23_rl_train

PYTHONPATH=MASbench python3 MASbench/scripts/eval_real_bench23_rl.py \
  --bench all --adapter-factory my_backend:make_adapter \
  --policy-factory my_policy:make_policy \
  --checkpoint MASbench/results/bench23_rl_train/checkpoint-final \
  --out MASbench/results/bench23_rl_test
```

All three benches therefore share one experimental contract: real train
episodes update a local trainable policy, real test episodes are frozen and
held out, and only the terminal checker determines success, reward, SR, and
success-conditioned time.
