# MASbench

MASbench is a small multi-agent benchmark with a symbolic task suite and a
real VillagerAgent/Minecraft path. The prompt-only model runner and the
trainable veRL runner use the same world adapter and checker, but write their
controllers, weights and results separately.

## TODO

- Add a provider-neutral rollout cache and a replay command for comparing APIs
  and local policies.
- Add grouped multi-agent rollouts with episode and step credit assignment.
- Keep one trajectory format for observations, actions, tool feedback and
  checker events.
- Record retry budgets and rate limits during large real-world runs.
- Add experiment manifests, checkpoint metadata and aggregate reports.
- Add CI checks for split leakage, provider contracts and real-adapter setup.

## Layout

- masbench/ — task schemas, environments, adapters, metrics and controllers.
- scripts/ — data audits, symbolic runs, training and evaluation commands.
- configs/ — veRL and model configuration files.
- docs/ — runbooks and checker notes.
- tests/ — CPU contracts and data-integrity checks.
- data/ — public manifests and task files.

## Closed-model Minecraft path

Run scripts/eval_minecraft_prompt.py to make one API decision per agent per
real Minecraft turn, then send that action through VillagerAgent and the world
checker. The GLM Flash factory is
masbench/minecraft/closed_glm_factory.py; it reads ZAI_API_KEY from the runtime
environment. Set MASBENCH_DISABLE_THINKING=1 when the endpoint must return a
JSON action directly.

The held-out real test split lives on the private evaluation host. The runner
audits the split before it calls the model or starts Minecraft.

## Public-data boundary

Do not commit API keys, private real splits, world snapshots, checker labels,
reference plans, logs or rollout results.
