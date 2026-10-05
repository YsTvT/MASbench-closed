# Closed-model Minecraft prompt baseline

This path is an inference-only baseline. It keeps the real VillagerAgent /
Mineflayer adapter, world reset, tools and checker, and changes only the policy
that chooses the next action. A closed API is queried once per agent per
environment turn with a frozen prompt and `temperature=0`, `n=1` by default.
No actor checkpoint, gradient, PPO or GRPO update is produced.

The loop is:

```text
fresh world + checker
  -> observe Alice/Bob
  -> task + observation + teammate state + action history + tool contract
  -> frozen API prompt
  -> strict JSON/tool-call parser
  -> VillagerAgent action
  -> tool feedback and next observation
  -> checker score.json
```

`masbench.minecraft.prompting` accepts both the MASbench typed form
`{"type":"gather", ...}` and the original VillagerAgent-shaped form
`{"tool":"MineBlock","tool_input":{...}}`. Invalid model output becomes an
`observe` action with a parser diagnostic in the episode trace; it is never
silently counted as success.

The controller is deliberately separate from `train_real_rl.py`. The RL path
requires a trainable local policy and veRL-compatible rollout records. The
closed API path writes `episode.json` plus an authoritative `score.json` and
marks the record with:

```json
{
  "model_type": "closed_api",
  "weights_updated": false,
  "controller_trained": false,
  "test_frozen": true
}
```

The last field is true only for a held-out test run. If prompt or controller
search is performed, it must use the train split, freeze the chosen prompt and
model version, and run the test split in a fresh output directory.

## Provider factory

The runner accepts a `module:factory` for both the real adapter and the model.
The repository includes dependency-free OpenAI-compatible, Anthropic and
Gemini adapters. For example, create a small module outside the benchmark
package:

```python
import os

from masbench.minecraft.api_adapters import OpenAICompatibleAdapter
from masbench.minecraft.villageragent_real import make_adapter


def make_model():
    return OpenAICompatibleAdapter(
        model=os.environ["MASBENCH_CLOSED_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.0,
        max_tokens=256,
    )


def make_real_adapter(task, episode_id):
    return make_adapter(task, episode_id)
```

Run it against a real server and checker with:

```bash
PYTHONPATH=MASbench:$PWD \
python3 MASbench/scripts/eval_minecraft_prompt.py \
  --split train \
  --tasks-file MASbench/data/minecraft_real_pilot/train.jsonl \
  --adapter-factory my_closed_model:make_real_adapter \
  --model-factory my_closed_model:make_model \
  --model-label gpt-4o \
  --model-version gpt-4o-2024-08-06 \
  --prompt-version minecraft-villageragent-typed-prompt-v1 \
  --out MASbench/results/minecraft_closed_prompt/pilot_gpt4o
```

The standard `masbench.minecraft.villageragent_real:make_adapter` factory may
be used directly when `VILLAGERAGENT_ROOT`, the Minecraft server and the world
snapshot are configured on the host. Set `MASBENCH_CHECKER_TIMEOUT` high enough
for the checker to append its final record. A missing checker remains
`evaluation_status="unassessed"` and is excluded from SR by the shared
success-first evaluator.

The command above is the currently wired task-34 pilot. The regular
`data/minecraft_tasks.jsonl` pool is the benchmark schema and does not carry a
server-specific `task_idx` for every external world. For a held-out test run,
use a task file with matching real-world indices, or provide an adapter factory
that maps each held-out task to its corresponding world and checker task index.

Aggregate saved scores with:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/evaluate_real_rollouts.py \
  MASbench/results/minecraft_closed_prompt/pilot_gpt4o \
  --out MASbench/results/minecraft_closed_prompt/pilot_gpt4o_eval.json
```

This typed controller has the same environment/checker boundary as the
original VillagerAgent inference setting, while keeping MASbench's explicit
action schema. A paper-faithful reproduction of the original natural-language
ReAct baseline should instead run the upstream `BaseAgent -> env.step ->
Agent.run` loop with the same Alice/Bob tool set and prompt version, then wrap
its checker output into this score schema.

## Z.ai GLM Flash

The public `masbench.minecraft.closed_glm_factory` targets the OpenAI-compatible
Z.ai endpoint and reads the key only from `ZAI_API_KEY`. `glm-4.5-flash` can be
used for a no-cost prompt-path run when the account's free quota is available.
GLM Flash enables reasoning by default; set `MASBENCH_DISABLE_THINKING=1` so
the endpoint returns the JSON action directly for the strict parser.

The canonical held-out real test command is:

```bash
export ZAI_API_KEY='...'
export MASBENCH_CLOSED_MODEL=glm-4.5-flash
export MASBENCH_DISABLE_THINKING=1
export ZAI_BASE_URL=https://api.z.ai/api/paas/v4/chat/completions
PYTHONPATH=. python3 scripts/eval_minecraft_prompt.py \
  --tasks-file data/minecraft_real/test.jsonl \
  --train-tasks-file data/minecraft_real/train.jsonl \
  --split test --max-tasks 0 --repeats 1 \
  --out results/minecraft_closed_prompt/glm_flash_full \
  --model-label glm-free-flash --model-version glm-4.5-flash \
  --adapter-factory masbench.minecraft.closed_glm_factory:make_real_adapter \
  --model-factory masbench.minecraft.closed_glm_factory:make_model
```

The split audit runs before any API or Minecraft work. Keep real test rows,
world snapshots, checker labels, API keys and rollout results outside public
Git history.
