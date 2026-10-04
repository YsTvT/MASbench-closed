# Real VillagerAgent integration status

The code-side integration is complete:

- `masbench.minecraft.villageragent.VillagerAgentAdapter` defines reset,
  observe, batched action execution and world-state scoring.
- The remote workspace contains the vendored VillagerAgent source and its
  Mineflayer dependency tree.
- Python dependencies from the vendor requirements and 246 Node packages were
  installed in the remote integration environment.
- `mineflayer`, `mineflayer-pathfinder`, `mineflayer-collectblock` and
  `minecraft-data` import successfully.

The remote server now has a real Minecraft 1.19.2 instance, a local
OpenAI-compatible Qwen2.5-7B-Instruct endpoint, and the VillagerAgent runner
connected to both. The model route is verified as `OpenAILanguageModel`, the
local `/v1/models` and chat-completion probes return successfully, and the
Mineflayer agents have produced real `navigateTo`/`scanNearbyEntities` actions
with block observations. The task graph is non-empty and Alice/Bob are both
assigned by `TaskManager`.

The remote integration also contains two runtime fixes required by this
server: case-insensitive Qwen routing in the Mineflayer client, and a
configurable build height (`MASBENCH_BUILD_Y`, default 103) so the blueprint
checker is aligned with the actual world ground. The no-action failure path now
initializes its feedback payload instead of raising `UnboundLocalError`.

The current Qwen pilot generated real action traces but did not complete the
21-block construction task. Its formal benchmark result is therefore
**unassessed**, rather than an SR=0 capability claim: action/state evidence is
present, but there is no checker success. The remote runner now always writes a
`score.json` record with `evaluation_status=unassessed` when it exits before a
world checker; this makes the missing-evaluation state explicit and keeps it
out of SR aggregation. The retained logs and
`result/masbench_real_construction_task34_qwen25_7b_retry17/` directory are the
source of truth for monitoring.

Run the client smoke with:

```bash
cd vendor/VillagerAgent
node ../../scripts/mineflayer_connect_smoke.js
```
