# Closed-model Minecraft evaluation

This is the prompt-only path for the real Minecraft adapter. It uses the same
VillagerAgent/Mineflayer tools and end-of-episode checker as the trainable path;
the model only supplies the next action. No weights or gradients are written.

## One environment turn

1. Reset a world and start the checker.
2. Send the task, Alice/Bob observations, teammate state, recent actions and
   tool contract to the model.
3. Parse one JSON action and send it through VillagerAgent.
4. Record tool feedback until the episode ends.
5. Save the trace and the checker's score.json.

The parser accepts both the MASbench typed action and the original VillagerAgent
JSON shape. Bad output becomes an observe action and stays in the trace with
the parser error; it is never counted as a successful action.

A closed-model record includes model_type=closed_api, weights_updated=false,
controller_trained=false and test_frozen=true. The last flag is only for a
held-out run. Choose prompts and models on train, freeze them, and use a fresh
output directory for test.

## Provider setup

The evaluator takes one factory for the real adapter and one for the model
client. The bundled clients cover OpenAI-compatible, Anthropic and Gemini APIs.
A provider module can live outside this repository:

~~~python
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
~~~

Run it against a real server and checker (replace the provider module name):

~~~bash
PYTHONPATH=MASbench:$PWD python3 MASbench/scripts/eval_minecraft_prompt.py --split train --tasks-file MASbench/data/minecraft_real_pilot/train.jsonl --adapter-factory my_closed_model:make_real_adapter --model-factory my_closed_model:make_model --model-label gpt-4o --model-version gpt-4o-2024-08-06 --prompt-version minecraft-villageragent-typed-prompt-v1 --out MASbench/results/minecraft_closed_prompt/pilot_gpt4o
~~~

The real adapter also works directly when VILLAGERAGENT_ROOT, the Minecraft
server and a world snapshot are configured. Set MASBENCH_CHECKER_TIMEOUT long
enough for the checker to write its final record. If the checker does not run,
the episode is unassessed and is left out of success-rate calculations.

The regular data/minecraft_tasks.jsonl file describes benchmark tasks but does
not carry a task_idx for every external world. For a real test run, use a
world-indexed task file or let the adapter map each task to its server index.

Aggregate saved scores with:

~~~bash
PYTHONPATH=MASbench python3 MASbench/scripts/evaluate_real_rollouts.py MASbench/results/minecraft_closed_prompt/pilot_gpt4o --out MASbench/results/minecraft_closed_prompt/pilot_gpt4o_eval.json
~~~

## Z.ai GLM Flash

masbench.minecraft.closed_glm_factory reads ZAI_API_KEY and uses the
OpenAI-compatible Z.ai endpoint. glm-4.5-flash is suitable for a free-quota
run when the account has quota. Disable hidden reasoning when the client must
receive the JSON action directly:

~~~bash
export ZAI_API_KEY='...'
export MASBENCH_CLOSED_MODEL=glm-4.5-flash
export MASBENCH_DISABLE_THINKING=1
export ZAI_BASE_URL=https://api.z.ai/api/paas/v4/chat/completions
PYTHONPATH=. python3 scripts/eval_minecraft_prompt.py --tasks-file data/minecraft_real/test.jsonl --train-tasks-file data/minecraft_real/train.jsonl --split test --max-tasks 0 --repeats 1 --out results/minecraft_closed_prompt/glm_flash_full --model-label glm-free-flash --model-version glm-4.5-flash --adapter-factory masbench.minecraft.closed_glm_factory:make_real_adapter --model-factory masbench.minecraft.closed_glm_factory:make_model
~~~

The split audit runs before any API call or Minecraft work. Keep private test
rows, world snapshots, checker labels, API keys and rollout outputs out of Git.
