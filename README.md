# MASbench

MASbench is a reproducible multi-agent benchmark with symbolic contracts and a
real VillagerAgent/Minecraft rollout path. The repository is organized around
the same separation used by `verl-agent`: environment contracts, model/policy
adapters, runnable scripts, configuration, documentation, and tests remain
independent so a prompt-only closed model can be evaluated beside a trainable
veRL policy.

## Repository layout

- `masbench/`: task schemas, environments, adapters, metrics, and controllers.
- `scripts/`: data audits, symbolic sanity runs, real RL training/evaluation,
  and closed-model prompt evaluation.
- `configs/`: veRL and model configuration templates.
- `data/`: benchmark task files and manifests. The private real-evaluation
  host provisions the disjoint `data/minecraft_real/{train,val,test}.jsonl`
  splits; held-out rows are not copied into this public source snapshot.
- `docs/`: rollout protocols, checker contracts, and provider setup notes.
- `tests/`: CPU contract and data-integrity tests.

Generated oracle traces, reference plans, server snapshots, and raw held-out
test labels are intentionally excluded from the public Git history. They must
remain in the private evaluation workspace so publishing the repository does
not disclose the answers used by the checkers.

Before publishing or running a real test evaluation, run the split audit:

```bash
PYTHONPATH=. python3 scripts/audit_minecraft_real_splits.py \
  --train data/minecraft_real/train.jsonl \
  --val data/minecraft_real/val.jsonl \
  --test data/minecraft_real/test.jsonl
```

当前可执行部分是两个长时程、多代理分解任务环境：

- **Bench2 Workflow**：独立分支并行执行，之后经过 approval/release 依赖链。
- **Bench3 Incident**：并行采集信号，随后完成诊断、修复、验证和沟通。
- **Minecraft task schema**：为 VillagerAgent 适配器准备的资源链、建造和远征任务图。

两个环境都提供确定性状态机、延迟消息、worker/resource lock、action/message budget、版本检查、完整 trace 和 deterministic checker。串行 baseline 与 MAS 使用同一案例、deadline、预算和 checker。

`masbench.longhorizon.closed_loop` 提供统一的 observation/action controller
循环；`masbench.controllers` 提供 provider-neutral 的 `ModelAdapter`、JSON
contract 校验和磁盘 cache。以后接入闭源 API 只需实现 `generate()`，不会改
环境或评分代码。

Minecraft 也提供同一类 reset/observe/step/score adapter。没有 GPU 时可以
运行 symbolic backend 做全量 task graph、deadline 和 checker sanity；真实
VillagerAgent rollout 只在服务器和 Minecraft world 可用后执行。

## 数据协议

数据生成器产生 223 个任务：175 个 workflow、48 个 incident。数据按 family 和 difficulty 分层切成互斥的 133 train、33 dev、57 test；运行 `scripts/audit_longhorizon_data.py` 会检查 ID 泄漏、split/file 一致性、难度覆盖、依赖图、故障信号和隐藏答案泄漏。

Bench2/Bench3 另有独立数据池：`data/bench2_workflow/`（175 个 workflow）和
`data/bench3_incident/`（48 个 incident）。用 `scripts/generate_bench23_data.py`
重新物化，用 `scripts/audit_bench23_data.py` 做严格审计，用
`scripts/run_bench23.py` 运行两个 benchmark 的统一 baseline；完整协议见
`docs/BENCH23_READY.md`。

## 指标

先报告 success rate（含 Wilson 95% 区间）。只有成功的 episode 才记录 completion time；失败 episode 的时间为 `null` 且不进入均值。综合分数为：

```text
composite = SR * (1 - mean(completion_ticks / deadline | success))
```

## 运行

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/audit_longhorizon_data.py
PYTHONPATH=MASbench python3 -m masbench.longhorizon.experiment \
  --out MASbench/results/longhorizon/test_tight \
  --data-dir MASbench/data --split test --deadline-scale 0.75
PYTHONPATH=MASbench python3 MASbench/scripts/run_closed_loop_suite.py \
  --data-dir MASbench/data --split test \
  --out MASbench/results/longhorizon/closed_loop_test_1_0
PYTHONPATH=MASbench python3 MASbench/scripts/run_all_sanity.py
PYTHONPATH=MASbench python3 MASbench/scripts/generate_bench23_data.py
PYTHONPATH=MASbench python3 MASbench/scripts/audit_bench23_data.py
PYTHONPATH=MASbench python3 MASbench/scripts/run_bench23.py --bench all --split test \
  --out MASbench/results/bench23/test_normal
PYTHONPATH=MASbench PYTHONPYCACHEPREFIX=/tmp/masbench-pyc \
  python3 -m unittest discover -s MASbench/tests -v
```

## 开放模型 smoke run

`results/open_model_qwen25_7b_smoke.md` 记录了服务器上 `Qwen2.5-7B-Instruct` 的首轮 4-case 试跑：3/4 成功（SR=0.75）。唯一失败是模型输出了不可执行的 `"released workflow"`，而环境要求规范动作值 `"release"`；该失败会被严格 checker 保留。`masbench.longhorizon.prompts` 提供固定 prompt 和 JSON contract，后续全量模型实验应直接复用它。模型配置模板见 `configs/models.json`；API key 只从环境变量读取。

Minecraft 任务设计见 `docs/MINECRAFT_TASKS.md`，生成和审计命令为：

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/generate_minecraft_tasks.py
PYTHONPATH=MASbench python3 MASbench/scripts/generate_minecraft_reference_plans.py
PYTHONPATH=MASbench python3 MASbench/scripts/generate_minecraft_episodes.py
PYTHONPATH=MASbench python3 MASbench/scripts/audit_minecraft_tasks.py
```

实验前冻结状态见 `docs/FINAL_READINESS.md`。`scripts/run_all_sanity.py`
会运行两个长时程环境和 Minecraft symbolic backend 的全量 CPU sanity。

真实 rollout 的结果评估入口是：

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/evaluate_real_rollouts.py \
  MASbench/results/monitoring --out MASbench/results/real_eval_summary.json
```

评估器只接收 `score.json` 中 `evaluation_status="evaluated"` 的 episode。
缺少最终 world checker 的记录保存为 `unassessed`，会被排除出 SR 分母并单独
报告。当前 Qwen Minecraft pilot 属于 inference-only rollout；仓库没有把
Qwen 权重再训练成新 checkpoint 的训练脚本，不能把 symbolic sanity 或
4-case smoke 当成训练后的真实模型结果。

真实场景 RL 的轨迹协议和 CLI 见 `docs/REAL_RL_PROTOCOL.md`、
`scripts/train_real_rl.py` 和 `scripts/eval_real_rl.py`。训练 CLI 默认要求
注入 `real_rollout=True` 的 VillagerAgent adapter 与可更新的本地 Qwen/LoRA
policy；`--dry-run` 只测试 symbolic 轨迹序列化，产生的记录明确标记为
`real_rollout=false`，不能进入真实 eval。

闭源模型的 Minecraft 推理基线走独立的 prompt-only CLI：
`scripts/eval_minecraft_prompt.py`。它沿用真实 VillagerAgent/Mineflayer
adapter、reset、工具反馈和 checker，每个 agent 每个环境 turn 调用一次冻结
API，并把 prompt、原始响应、解析动作和 tool feedback 写入 episode trace。
闭源 API 不参与 PPO/GRPO，也不会生成伪 checkpoint；完整协议、provider
factory 和命令见 `docs/MINECRAFT_CLOSED_API.md`。

Minecraft 的默认 real adapter 入口为
`masbench.minecraft.villageragent_real:make_adapter`，训练 policy 可使用
`masbench.minecraft.qwen_lora_policy:QwenLoRAPolicy` 的本地 Transformers/PEFT
封装；两者都要求在 VillagerAgent server 主机上运行，不能替换成 API-only
Qwen 推理服务。

Bench2/Bench3 也使用同一协议，入口是
`scripts/train_real_bench23_rl.py` 和 `scripts/eval_real_bench23_rl.py`，详见
`docs/BENCH23_REAL_RL_PROTOCOL.md`。当前仓库没有外部企业工作流或 incident
服务，因此这两个 CLI 强制要求用户注入声明 `real_rollout=True` 的真实 adapter；
现有 `WorkflowEnv`/`IncidentEnv` 只能用于契约测试，不能生成正式 RL train/test 数据。
