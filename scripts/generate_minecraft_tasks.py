"""Generate a VillagerBench-aligned and MAS-hard Minecraft task pool."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
ROOT.mkdir(exist_ok=True)
ACTIONS = ["observe", "move", "gather", "craft", "cook", "place", "attack", "trade", "handoff", "deposit", "submit"]


def subtask(task_id, action, deps, ticks, description, outputs):
    return {"id": task_id, "action": action, "depends_on": list(deps),
            "estimated_ticks": ticks, "description": description, "outputs": outputs}


def split_for(index, total):
    train_end = int(total * 0.60)
    dev_end = train_end + int(total * 0.15)
    return "train" if index < train_end else "dev" if index < dev_end else "test"


def topology_split(topology_index, counts):
    """Assign whole topology families to a split to prevent template leakage."""
    train, dev, test = counts
    if topology_index < train:
        return "train"
    if topology_index < train + dev:
        return "dev"
    if topology_index < train + dev + test:
        return "test"
    raise ValueError("topology index outside split policy")


def parallel_groups(subtasks):
    """Return topological levels; each level is an executable batch."""
    remaining = {s["id"]: set(s["depends_on"]) for s in subtasks}
    resolved, groups = set(), []
    while remaining:
        ready = sorted(node for node, deps in remaining.items() if deps <= resolved)
        if not ready:
            raise ValueError("cyclic task graph")
        groups.append(ready)
        resolved.update(ready)
        for node in ready:
            del remaining[node]
    return groups


def critical_path_ticks(subtasks):
    durations = {s["id"]: s["estimated_ticks"] + 2 for s in subtasks}
    deps = {s["id"]: s["depends_on"] for s in subtasks}
    memo = {}

    def visit(node):
        if node in memo:
            return memo[node]
        memo[node] = durations[node] + max((visit(dep) for dep in deps[node]), default=0)
        return memo[node]

    return max(visit(node) for node in durations)


def action_specs(subtasks):
    specs = []
    for subtask_item in subtasks:
        action = subtask_item["action"]
        specs.append({"subtask_id": subtask_item["id"], "action": action,
                      "args": {"target": subtask_item["id"], "quantity": subtask_item["outputs"]},
                      "preconditions": {"completed": subtask_item["depends_on"]},
                      "effects": subtask_item["outputs"]})
    return specs


def world_config(subfamily, local_index, metadata):
    biome = ["plains", "forest", "taiga", "savanna", "desert"][local_index % 5]
    return {"minecraft_version": "1.19.2", "biome": biome,
            "base_position": [0, 64, 0], "task_origin": [16 + (local_index % 8) * 8, 64, 16],
            "task_radius": 32 + (local_index % 4) * 8,
            "stations": ["crafting_table", "chest"] + (["furnace"] if subfamily == "cooking" else []),
            "metadata": metadata}


def agent_roles(required_agents, subfamily):
    role_names = {"construction": ["builder", "resource_collector", "logistics"],
                  "cooking": ["forager", "processor", "cook", "server"],
                  "escape": ["scout", "switch_operator", "runner"],
                  "mas_hard": ["planner", "collector", "executor", "verifier"]}[subfamily]
    return [{"agent_id": "agent_%d" % i, "role": role_names[i % len(role_names)],
             "capabilities": ["observe", "move", "message"]} for i in range(required_agents)]


def event_specs(events, difficulty):
    specs = []
    for i, event in enumerate(events or []):
        if isinstance(event, dict):
            specs.append(event)
            continue
        effects = {
            "nightfall": {"visibility": 0.5}, "resource_shortage": {"resource_multiplier": 0.7},
            "rain": {"movement_multiplier": 0.85}, "ingredient_shortage": {"missing_ingredient": True},
            "synchronization_window": {"window_ticks": 3 + difficulty}, "darkness": {"visibility": 0.4},
            "door_reset": {"switch_state": "reset"}, "agent_delay": {"action_delay_ticks": 2},
            "resource_competition": {"shared_resource": True}, "world_reset": {"partial_reset": True},
        }.get(event, {"flag": event})
        specs.append({"event_id": "%s_%d" % (event, i), "type": event,
                      "trigger": {"tick": 10 + i * 7, "probability": 0.35 + 0.1 * difficulty},
                      "effects": effects})
    return specs


def base_task(task_id, local_index, total, subfamily, source_alignment, goal, subtasks,
              checks, world_seed, events=None, metadata=None, split=None, topology_id=None):
    difficulty = local_index % 3 + 1
    groups = parallel_groups(subtasks)
    if not any(len(group) >= 2 for group in groups):
        raise ValueError("every Minecraft task must expose a parallel batch")
    serial_ticks = sum(s["estimated_ticks"] + 2 for s in subtasks)
    critical_ticks = critical_path_ticks(subtasks)
    deadline_mix, margin = {"construction": (0.68, 8), "cooking": (0.80, 5),
                            "escape": (0.55, 5), "mas_hard": (0.55, 7)}[subfamily]
    deadline = int(round(critical_ticks + deadline_mix * (serial_ticks - critical_ticks) + margin + difficulty * 2))
    required_agents = 2 + (difficulty == 3)
    concrete_checks = [{"type": "state_predicate", "predicate": check} for check in checks]
    return {
        "task_id": task_id, "family": "minecraft", "subfamily": subfamily,
        "source_alignment": source_alignment, "split": split or split_for(local_index, total),
        "topology_id": topology_id or "%s_topology_%02d" % (subfamily, local_index),
        "difficulty": difficulty, "deadline_ticks": deadline,
        "serial_estimated_ticks": serial_ticks, "critical_path_ticks": critical_ticks,
        "world_seed": world_seed, "goal": goal, "subtasks": subtasks,
        "parallel_groups": groups, "success_checks": checks,
        "environment_events": event_specs(events, difficulty), "required_agents": required_agents,
        "agent_roles": agent_roles(required_agents, subfamily),
        "world_config": world_config(subfamily, local_index, metadata or {}),
        "initial_inventory": {"food": 8, "wood": 4, "stone": 4, "torch": 8},
        "action_specs": action_specs(subtasks), "concrete_success_checks": concrete_checks,
        "baseline": "villageragent_serial", "action_space": ACTIONS,
        "metadata": metadata or {},
    }


def make_construction(local_index):
    styles = ["house", "bridge", "watchtower", "farmhouse", "warehouse"]
    materials = ["oak", "spruce", "stone", "deepslate", "mixed"]
    style, material = styles[local_index % len(styles)], materials[(local_index // 5) % len(materials)]
    size = 3 + local_index % 5
    topology_index = local_index % 20
    p = "c%03d" % local_index
    tasks = [
        subtask(p + "_wood", "gather", [], 8 + size, "collect %s building wood" % material, {"planks": 24 + size * 4}),
        subtask(p + "_stone", "gather", [], 8 + size, "collect foundation stone", {"stone": 16 + size * 3}),
        subtask(p + "_light", "trade", [], 7 + local_index % 4, "obtain lighting materials", {"lantern": 4 + local_index % 4}),
        subtask(p + "_foundation", "place", [p + "_wood", p + "_stone"], 7 + size, "place the %s foundation" % style, {"foundation": 1}),
        subtask(p + "_frame", "place", [p + "_foundation"], 8 + size, "build the structural frame", {"frame": 1}),
        subtask(p + "_walls", "place", [p + "_frame"], 9 + size, "build the enclosing walls", {"walls": 1}),
        subtask(p + "_roof", "place", [p + "_walls", p + "_wood"], 8 + size, "finish the roof", {"roof": 1}),
        subtask(p + "_lights", "place", [p + "_walls", p + "_light"], 5, "place safe interior lighting", {"lighting": 1}),
        subtask(p + "_inspect", "observe", [p + "_roof", p + "_lights"], 3, "inspect against the blueprint", {"blueprint_valid": 1}),
    ]
    extra_check = None
    if topology_index % 4 == 1:
        tasks.append(subtask(p + "_windows", "place", [p + "_walls"], 5, "place windows", {"windows": 1}))
        tasks[-1]["depends_on"] = [p + "_walls"]
        extra_check = "windows==1"
    elif topology_index % 4 == 2:
        tasks.append(subtask(p + "_door", "place", [p + "_walls"], 4, "place a secure door", {"door": 1}))
        extra_check = "door==1"
    elif topology_index % 4 == 3:
        tasks.append(subtask(p + "_storage", "place", [p + "_walls"], 5, "place storage", {"storage": 1}))
        extra_check = "storage==1"
    if extra_check:
        inspect = next(s for s in tasks if s["id"] == p + "_inspect")
        inspect["depends_on"].append(extra_check.split("==")[0] and tasks[-1]["id"])
    checks = ["foundation==1", "walls==1", "roof==1", "lighting==1", "blueprint_valid==1"]
    if extra_check:
        checks.append(extra_check)
    return base_task("minecraft_construction_%03d_d%d" % (local_index, local_index % 3 + 1), local_index, 100,
                     "construction", "villagerbench_construction",
                     "Build and light a %s of size %d using %s materials." % (style, size, material), tasks,
                     checks,
                     21000 + local_index, ["nightfall" if local_index % 4 == 0 else "resource_shortage"],
                     {"blueprint_style": style, "material_family": material, "size": size},
                     topology_split(topology_index, (12, 3, 5)), "construction_topology_%02d" % topology_index)


def make_cooking(local_index):
    recipes = [
        ("cake", [("wheat", "gather", 3), ("sugar", "craft", 2), ("egg", "gather", 1), ("milk", "trade", 3)]),
        ("rabbit_stew", [("cooked_rabbit", "cook", 1), ("baked_potato", "cook", 1), ("carrot", "gather", 1), ("mushroom", "gather", 1), ("bowl", "craft", 1)]),
        ("bread", [("wheat", "gather", 3)]),
        ("golden_carrot", [("carrot", "gather", 1), ("gold_nugget", "trade", 8)]),
        ("suspicious_stew", [("mushroom_a", "gather", 1), ("mushroom_b", "gather", 1), ("flower", "gather", 1), ("bowl", "craft", 1)]),
    ]
    recipe, ingredients = recipes[local_index % len(recipes)]
    topology_index = local_index % 20
    p = "k%03d" % local_index
    gather_ids, tasks = [], []
    for item, action, count in ingredients:
        node = p + "_" + item
        gather_ids.append(node)
        tasks.append(subtask(node, action, [], 6 + count + local_index % 4, "obtain %d %s" % (count, item), {item: count}))
    station = p + "_station"
    tasks.append(subtask(station, "place", [], 4, "prepare the required cooking station", {"station_ready": 1}))
    process = p + "_process"
    tasks.append(subtask(process, "craft", gather_ids + [station], 5 + len(ingredients), "process ingredients for %s" % recipe, {"prepared": 1}))
    cook = p + "_cook"
    final_action = "cook" if recipe == "rabbit_stew" else "craft"
    tasks.append(subtask(cook, final_action, [process], 7 + local_index % 5, "prepare the %s recipe" % recipe, {recipe: 1}))
    serving_dep = cook
    service_node = None
    if topology_index % 4 == 1:
        service_node = p + "_cool"
        tasks.append(subtask(service_node, "observe", [cook], 3, "cool the dish safely", {"cooled": 1}))
        serving_dep = service_node
    elif topology_index % 4 == 2:
        service_node = p + "_package"
        tasks.append(subtask(service_node, "craft", [cook], 4, "package the dish", {"packaged": 1}))
        serving_dep = service_node
    elif topology_index % 4 == 3:
        service_node = p + "_inspect"
        tasks.append(subtask(service_node, "observe", [cook], 3, "inspect the dish", {"dish_valid": 1}))
        serving_dep = service_node
    serve = p + "_serve"
    tasks.append(subtask(serve, "deposit", [serving_dep], 3, "serve the finished dish", {"served": 1}))
    extra_check = {1: "cooled==1", 2: "packaged==1", 3: "dish_valid==1"}.get(topology_index % 4)
    checks = [recipe + ">=1", "served==1"] + ([extra_check] if extra_check else [])
    return base_task("minecraft_cooking_%03d_d%d" % (local_index, local_index % 3 + 1), local_index, 100,
                     "cooking", "villagerbench_cooking",
                     "Farm-to-table preparation of one %s under changing conditions." % recipe, tasks,
                     checks, 22000 + local_index,
                     ["rain" if local_index % 3 == 0 else "ingredient_shortage"],
                     {"recipe": recipe, "ingredient_count": len(ingredients), "portion": 1 + local_index % 3,
                      "station": "furnace_and_crafting_table" if recipe == "rabbit_stew" else "crafting_table",
                      "vanilla_prerequisites": {"rabbit_stew": ["cooked_rabbit", "baked_potato", "carrot", "mushroom", "bowl"],
                                                  "cake": ["wheat", "sugar", "egg", "milk"],
                                                  "bread": ["wheat"], "golden_carrot": ["carrot", "gold_nugget"],
                                                  "suspicious_stew": ["mushroom_a", "mushroom_b", "flower", "bowl"]}[recipe]},
                     topology_split(topology_index, (12, 4, 4)), "cooking_topology_%02d" % topology_index)


def make_escape(local_index):
    difficulty = local_index % 3 + 1
    switch_count = 2 + (local_index % 3)
    topology_index = local_index % 5
    p = "e%03d" % local_index
    tasks = [subtask(p + "_switch_%d" % i, "move", [], 5 + i, "reach switch %d" % i, {"switch_%d": 1}) for i in range(switch_count)]
    switch_ids = [s["id"] for s in tasks]
    extra_checks = []
    if topology_index % 5 in (1, 3):
        token = p + "_token"
        tasks.append(subtask(token, "observe", [], 4, "find the access token", {"token": 1}))
        switch_ids.append(token)
        extra_checks.append("token==1")
    tasks += [
        subtask(p + "_sync", "place", switch_ids, 4, "activate switches in the timing window", {"door_open": 1}),
    ]
    key_dep = p + "_sync"
    if topology_index >= 3:
        trap = p + "_trap"
        tasks.append(subtask(trap, "observe", [p + "_sync"], 4, "disable the reset trap", {"trap_disabled": 1}))
        key_dep = trap
        extra_checks.append("trap_disabled==1")
    tasks += [subtask(p + "_key", "observe", [key_dep], 4 + difficulty, "locate the exit key", {"key": 1}),
              subtask(p + "_exit", "move", [p + "_key"], 8 + difficulty * 2, "reach the exit with the key", {"escaped": 1})]
    return base_task("minecraft_escape_%03d_d%d" % (local_index, difficulty), local_index, 25,
                     "escape", "villagerbench_escape",
                     "Coordinate %d switches, recover the exit key and escape the room." % switch_count, tasks,
                     ["door_open==1", "key==1", "escaped==1"] + extra_checks, 23000 + local_index,
                     ["synchronization_window", "darkness" if local_index % 2 else "door_reset"],
                     {"switch_count": switch_count, "timing_window_ticks": 3 + local_index % 4},
                     topology_split(topology_index, (3, 1, 1)), "escape_topology_%02d" % topology_index)


def make_hard(local_index):
    variant = local_index % 3
    topology_index = local_index % 15
    p = "h%03d" % local_index
    if variant == 0:
        tasks = [
            subtask(p + "_wood", "gather", [], 10, "collect building wood", {"planks": 40}),
            subtask(p + "_stone", "gather", [], 10, "collect foundation stone", {"stone": 32}),
            subtask(p + "_survey", "observe", [], 5, "survey an unstable build site", {"site_safe": 1}),
            subtask(p + "_foundation", "place", [p + "_wood", p + "_stone", p + "_survey"], 9, "place reinforced foundation", {"foundation": 1}),
            subtask(p + "_walls", "place", [p + "_foundation"], 11, "build reinforced walls", {"walls": 1}),
            subtask(p + "_roof", "place", [p + "_walls"], 10, "finish weatherproof roof", {"roof": 1}),
            subtask(p + "_verify", "observe", [p + "_roof"], 4, "verify structure after a disturbance", {"verified": 1}),
        ]
        if topology_index % 3 == 1:
            tasks.insert(-1, subtask(p + "_waterproof", "place", [p + "_roof"], 5, "waterproof the roof", {"waterproof": 1}))
            tasks[-1]["depends_on"] = [p + "_waterproof"]
            checks = ["foundation==1", "walls==1", "roof==1", "waterproof==1", "verified==1"]
        else:
            checks = ["foundation==1", "walls==1", "roof==1", "verified==1"]
        goal = "Build a reinforced shelter while handling a changing site condition."
    elif variant == 1:
        tasks = [
            subtask(p + "_food", "gather", [], 9, "prepare food", {"food": 10}),
            subtask(p + "_tools", "craft", [], 8, "craft expedition tools", {"iron_pickaxe": 1}),
            subtask(p + "_torches", "craft", [], 7, "craft torches", {"torch": 24}),
            subtask(p + "_locate", "observe", [], 5, "locate the moving target", {"target_known": 1}),
            subtask(p + "_navigate", "move", [p + "_food", p + "_tools", p + "_torches", p + "_locate"], 15, "navigate through the changed route", {"at_target": 1}),
            subtask(p + "_recover", "gather", [p + "_navigate"], 10, "recover the target item", {"relic": 1}),
            subtask(p + "_handoff", "handoff", [p + "_recover"], 4, "handoff the relic to the return agent", {"relic_handed": 1}),
            subtask(p + "_return", "move", [p + "_handoff"], 14, "return safely to base", {"at_base": 1}),
        ]
        if topology_index % 3 == 2:
            tasks.insert(4, subtask(p + "_signal", "place", [], 4, "place a route signal", {"signal": 1}))
            tasks[next(i for i, t in enumerate(tasks) if t["id"] == p + "_navigate")]["depends_on"].append(p + "_signal")
        goal, checks = "Run a long-horizon rescue expedition with a dynamic route and handoff.", ["relic==1", "relic_handed==1", "at_base==1"]
    else:
        tasks = [
            subtask(p + "_redstone", "gather", [], 9, "collect redstone components", {"redstone": 12}),
            subtask(p + "_iron", "gather", [], 9, "collect iron components", {"iron": 4}),
            subtask(p + "_stone", "gather", [], 9, "collect circuit stone", {"stone": 16}),
            subtask(p + "_craft", "craft", [p + "_redstone", p + "_iron", p + "_stone"], 8, "craft synchronized mechanisms", {"mechanisms": 3}),
            subtask(p + "_wire", "place", [p + "_craft"], 12, "wire the circuit across separated rooms", {"circuit": 1}),
            subtask(p + "_activate", "place", [p + "_wire"], 6, "activate the circuit simultaneously", {"activated": 1}),
            subtask(p + "_check", "observe", [p + "_activate"], 4, "verify the circuit survived a reset", {"verified": 1}),
        ]
        if topology_index % 3 == 1:
            tasks.insert(-1, subtask(p + "_backup", "craft", [], 5, "craft a backup mechanism", {"backup": 1}))
            tasks[-1]["depends_on"] = [p + "_backup"]
            checks = ["circuit==1", "activated==1", "backup==1", "verified==1"]
        else:
            checks = ["circuit==1", "activated==1", "verified==1"]
        goal = "Construct and synchronize a multi-room redstone mechanism after a reset."
    topology_index = local_index % 15
    return base_task("minecraft_mas_hard_%03d_d%d" % (local_index, local_index % 3 + 1), local_index, 75,
                     "mas_hard", "masbench_extension", goal, tasks, checks, 24000 + local_index,
                     ["agent_delay", "resource_competition", "world_reset"], {"variant": variant},
                     topology_split(topology_index, (9, 2, 4)), "mas_hard_topology_%02d" % topology_index)


rows = [make_construction(i) for i in range(100)]
rows += [make_cooking(i) for i in range(100)]
rows += [make_escape(i) for i in range(25)]
rows += [make_hard(i) for i in range(75)]
with (ROOT / "minecraft_tasks.jsonl").open("w") as handle:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
manifest = {
    "count": len(rows),
    "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")},
    "families": {f: sum(r["subfamily"] == f for r in rows) for f in ("construction", "cooking", "escape", "mas_hard")},
    "source_alignment": {s: sum(r["source_alignment"] == s for r in rows) for s in ("villagerbench_construction", "villagerbench_cooking", "villagerbench_escape", "masbench_extension")},
    "baseline": "villageragent_serial", "protocol": "task graph first; success before time",
}
(ROOT / "minecraft_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, sort_keys=True))
