"""Import-time veRL-agent hook for the MASbench Minecraft environment."""

try:
    import agent_system.environments.env_manager as _em
    import agent_system.environments as _env_pkg
    from masbench.verl_minecraft_env import (MinecraftEnvironmentManager,
        build_minecraft_envs, minecraft_projection)

    _original_make_envs = _em.make_envs

    def _make_envs(config):
        if "minecraft" not in str(config.env.env_name).lower():
            return _original_make_envs(config)
        group_n = int(config.env.rollout.n)
        resources = config.env.get("resources_per_worker", {}) if hasattr(config.env, "get") else {}
        val_batch_size = int(getattr(config.data, "val_batch_size", 1) or 1)
        env_kwargs = {
            "max_turns": int(config.env.rollout.max_turns),
            "task_file": str(getattr(config.env.rollout, "task_file", "")),
            "server_root": str(getattr(config.env.rollout, "server_root", "")),
            "world_snapshot": str(getattr(config.env.rollout, "world_snapshot", "")),
            "villageragent_root": str(getattr(config.env.rollout, "villageragent_root", "")),
            "server_port_base": int(getattr(config.env.rollout, "server_port_base", 25600)),
            "agent_port_base": int(getattr(config.env.rollout, "agent_port_base", 6000)),
        }
        train = build_minecraft_envs(
            seed=config.env.seed,
            env_num=int(config.data.train_batch_size),
            group_n=group_n,
            resources_per_worker=resources,
            is_train=True,
            env_kwargs=env_kwargs,
        )
        val = build_minecraft_envs(
            seed=config.env.seed + 1000,
            env_num=val_batch_size,
            group_n=1,
            resources_per_worker=resources,
            is_train=False,
            env_kwargs=env_kwargs,
            candidate_offset=int(config.data.train_batch_size) * group_n,
        )
        return (MinecraftEnvironmentManager(train, minecraft_projection, config),
                MinecraftEnvironmentManager(val, minecraft_projection, config))

    # main_ppo imports ``make_envs`` from the package namespace, while some
    # utilities import it from env_manager. Patch both aliases before Hydra
    # creates the trainer; patching only env_manager leaves the original
    # factory in the package and produces "Environment not supported".
    _em.make_envs = _make_envs
    _env_pkg.make_envs = _make_envs
except Exception:
    # Normal MASbench imports must remain dependency-free. The veRL launcher
    # performs an explicit preflight and will surface import errors.
    pass

# Compatibility for vLLM LoRALRUCache with cachetools >= 7, whose private
# LRUCache method was renamed from __update to __touch.
try:
    import vllm.utils as _vllm_utils

    _lora_cache_cls = getattr(_vllm_utils, "LoRALRUCache", None)
    if _lora_cache_cls is not None:
        def _masbench_lora_cache_touch(self, key):
            updater = getattr(self, "_LRUCache__update", None)
            if updater is None:
                updater = getattr(self, "_LRUCache__touch", None)
            if updater is not None:
                return updater(key)
            return None

        _lora_cache_cls.touch = _masbench_lora_cache_touch
except Exception:
    pass
