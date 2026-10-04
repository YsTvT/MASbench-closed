# Real Qwen + veRL Minecraft runbook

This runbook uses the real VillagerAgent/Mineflayer server and its construction
checker.  The task source is VillagerAgent's `data/building_blue_print.json`;
`env/build_judger.py --idx N` is the authority for blueprint `N`.  The
task-34 pilot remains useful as a smoke run, while the split generator below
materializes held-out real blueprints for train/val/test.

On the remote host, after copying this checkout into `${MASBENCH_ROOT}`:

```bash
cd ${MASBENCH_ROOT}
export PYTHONPATH=${MASBENCH_ROOT}:${MASBENCH_VERL_AGENT_ROOT}
export VILLAGERAGENT_ROOT=${MASBENCH_ROOT}/vendor/VillagerAgent
export MASBENCH_SERVER_ROOT=${MASBENCH_SERVER_ROOT}
export MASBENCH_REQUIRE_EVALUATED=1

python scripts/make_minecraft_splits.py \
  --villager-root "$VILLAGERAGENT_ROOT" \
  --output-dir data/minecraft_real

${MASBENCH_VERL_ENV}/bin/python scripts/make_minecraft_parquet.py \
  --input data/minecraft_real/train.jsonl \
  --output data/minecraft_real/train.parquet --split train \
  --server-root ${MASBENCH_SERVER_ROOT} \
  --snapshot-root ${MASBENCH_ROOT}/world_snapshots/task_34 \
  --villager-root "$VILLAGERAGENT_ROOT"

${MASBENCH_VERL_ENV}/bin/python scripts/make_minecraft_parquet.py \
  --input data/minecraft_real/val.jsonl \
  --output data/minecraft_real/val.parquet --split val \
  --server-root ${MASBENCH_SERVER_ROOT} \
  --snapshot-root ${MASBENCH_ROOT}/world_snapshots/task_34 \
  --villager-root "$VILLAGERAGENT_ROOT"

${MASBENCH_VERL_ENV}/bin/python scripts/make_minecraft_parquet.py \
  --input data/minecraft_real/test.jsonl \
  --output data/minecraft_real/test.parquet --split test \
  --server-root ${MASBENCH_SERVER_ROOT} \
  --snapshot-root ${MASBENCH_ROOT}/world_snapshots/task_34 \
  --villager-root "$VILLAGERAGENT_ROOT"

${MASBENCH_VERL_ENV}/bin/python scripts/preflight_verl_minecraft.py \
  --task-file data/minecraft_real/train.jsonl
```

Start the official trainer from the veRL checkout.  `actor_rollout_ref.rollout.n`
must stay one because `main_ppo` asserts it; the Minecraft group size is
`env.rollout.n=4`.

```bash
cd ${MASBENCH_VERL_AGENT_ROOT}
PYTHONPATH=${MASBENCH_ROOT}:${MASBENCH_VERL_AGENT_ROOT} \
  VILLAGERAGENT_ROOT=${MASBENCH_ROOT}/vendor/VillagerAgent \
  MASBENCH_SERVER_ROOT=${MASBENCH_SERVER_ROOT} \
  MASBENCH_REQUIRE_EVALUATED=1 \
  ${MASBENCH_VERL_ENV}/bin/python -m verl.trainer.main_ppo \
    --config-path=${MASBENCH_ROOT}/configs \
    --config-name=verl_minecraft_grpo_lora \
    data.train_files=${MASBENCH_ROOT}/data/minecraft_real/train.parquet \
    data.val_files=${MASBENCH_ROOT}/data/minecraft_real/val.parquet \
    env.rollout.task_file=${MASBENCH_ROOT}/data/minecraft_real/train.jsonl
```

The trainer writes a LoRA adapter under
`${MASBENCH_ROOT}/results/verl_minecraft_grpo_lora_pilot`.  Evaluate
the frozen adapter only on the real test split (the pilot test file is a
separate episode/checker record for task 34):

Before evaluating, verify that the trainer actually wrote PEFT adapter
metadata and non-empty adapter weights.  This rejects a run directory that
contains only logs or the unchanged base Qwen model:

```bash
${MASBENCH_VERL_ENV}/bin/python \
  ${MASBENCH_ROOT}/scripts/verify_qwen_grpo_checkpoint.py \
  --checkpoint ${MASBENCH_ROOT}/results/verl_minecraft_grpo_lora_pilot
```

```bash
${MASBENCH_VERL_ENV}/bin/python \
  ${MASBENCH_ROOT}/scripts/eval_minecraft_real_qwen.py \
  --model ${MASBENCH_MODEL_ROOT}/Qwen2.5-7B-Instruct \
  --checkpoint /path/to/actor/lora_adapter \
  --tasks ${MASBENCH_ROOT}/data/minecraft_real_pilot/test.jsonl \
  --output ${MASBENCH_ROOT}/results/qwen_lora_real_test.jsonl
```

Only records with `evaluation_status=evaluated` are valid formal scores.  An
unassessed checker result aborts the rollout instead of being silently turned
into a zero-reward training sample.
