# MASbench

MASbench is a reproducible multi-agent benchmark with symbolic contracts and a real VillagerAgent/Minecraft rollout path. The closed prompt path and trainable veRL path share the real environment checker while keeping controllers, weights, caches and evaluation traces separate.

## TODO

- Add a provider-neutral rollout cache and replay command for deterministic comparisons across closed APIs and trainable policies.
- Add grouped multi-agent rollouts with explicit episode-level and step-level credit assignment hooks for long-horizon tasks.
- Add a unified trajectory schema for observations, actions, tool feedback, checker events and per-step advantages.
- Add configurable rollout grouping, retry budgets and rate-limit accounting for large real-environment sweeps.
- Add reproducible experiment manifests, checkpoint metadata and aggregate reports for success rate, completion time and checker coverage.
- Extend CI with split-leakage audits, provider contract tests and a real-adapter readiness check that never requires private test data.

## Repository layout

- masbench/: task schemas, environments, adapters, metrics and controllers.
- scripts/: audits, symbolic runs, real training/evaluation and closed-model prompt evaluation.
- configs/: veRL and model configuration templates.
- docs/: rollout protocols, checker contracts and provider setup notes.
- tests/: CPU contract and data-integrity tests.
- data/: public manifests and benchmark task files.

## Closed-model Minecraft path

The prompt-only evaluator runs one frozen API decision per agent per real Minecraft turn, then sends the action through the VillagerAgent tools and checker. The entry point is scripts/eval_minecraft_prompt.py. The GLM Flash provider is masbench/minecraft/closed_glm_factory.py and reads ZAI_API_KEY only from the runtime environment. Set MASBENCH_DISABLE_THINKING=1 for direct JSON actions.

The held-out real test split is loaded from data/minecraft_real/test.jsonl on the private evaluation host. The split audit runs before API or Minecraft work.

## Public-data boundary

API keys, private real splits, world snapshots, checker labels, oracle traces, reference plans, logs and rollout results stay outside public Git history.
