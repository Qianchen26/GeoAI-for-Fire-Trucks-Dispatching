"""Frozen-policy E1/E2 from task.md. Run with the existing Anaconda Python.

    PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -u station_optimization/run_experiments.py all
    ... run_experiments.py summarize  # only reaggregate persisted simulation outputs

All writes are confined to this directory. No fitting, search, or policy update.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parent / "cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import copy
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
RAW = RESULTS / "raw"
for folder in (RESULTS, RAW, HERE / "logs"):
    folder.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code" / "dispatch_simulator"))
import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
import environment
import planning as P
import policies
import scenarios

SEEDS = list(range(101, 116))
REPS, BOOT_SEED = 2000, 7
MODEL_DIR = ROOT / "data" / "models" / "planning"
E1_DRAWS = [("Medium", 1.0), ("High", 1.0), ("High", 0.0)]
METRICS = ("D_F", "M_F", "M_O")
SETUP = None


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configuration_id(config):
    text = json.dumps(config, sort_keys=True, default=jsonable, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def label(draw, year):
    return f"{draw[0]}_w{draw[1]:g}_{year}"


def predict(model, env, observation):
    action = int(model.predict(observation, action_masks=env.action_masks(), deterministic=True)[0])
    assert env.action_masks()[action], "frozen policy chose an invalid action"
    return action


def load_setup():
    torch.set_num_threads(1)
    world = environment.build_world(verbose=False)
    generator = scenarios.fit_generator(world, verbose=False)
    return (world, generator, *P.prepare(world, generator, verbose=False))


def describe(env, action):
    kind, slot, candidate = env.decode(action)
    if kind == "build":
        return f"build at C{candidate}"
    if kind == "move":
        return f"move {env.config['name'][slot]} to C{candidate}"
    if kind in ("engine", "quint"):
        return f"+{'E' if kind == 'engine' else 'Q'} at {env.config['name'][slot]}"
    return "no-op"


def prepare():
    """Reuse setup and freeze the exact prescribed comparisons before simulations."""
    global SETUP
    started = time.time()
    SETUP = load_setup()
    world, generator, inputs, evaluator, candidates = SETUP
    model = MaskablePPO.load(MODEL_DIR / "final" / "model.zip", device="cpu")
    baseline = pickle.loads((MODEL_DIR / "baselines.pkl").read_bytes())
    existing = pd.read_csv(MODEL_DIR / "final" / "verification.csv")
    actions_expected = pd.read_csv(MODEL_DIR / "final" / "rq3_actions.csv")
    tracts = inputs["tracts"].copy()
    tracts["fire_rate"] = evaluator.fire_tract_rate
    tracts["fire_weight"] = evaluator.fire_tract_rate / evaluator.fire_tract_rate.sum()
    tracts.to_file(RESULTS / "tracts.geojson", driver="GeoJSON", index=False)
    inputs["stations"].drop(columns="vehicle_ids").to_csv(RESULTS / "stations.csv", index=False)
    inputs["sites"].iloc[candidates].assign(candidate=[f"C{i}" for i in range(len(candidates))]).to_csv(RESULTS / "candidate_sites.csv", index=False)
    plan = {"comparisons": [], "configs": {}, "fast_histories": {}, "e2_selection": [], "identity_checks": []}

    def add_comparison(experiment, draw, year, a_name, a_hist, b_name, b_hist):
        a_id, b_id = configuration_id(a_hist["config"]), configuration_id(b_hist["config"])
        for ident, history in [(a_id, a_hist), (b_id, b_hist)]:
            plan["configs"][ident] = history["config"]
        case = {"experiment": experiment, "path": draw[0], "w": draw[1], "year": year,
                "label": label(draw, year), "a": a_name, "b": b_name, "a_config": a_id, "b_config": b_id,
                "identical": a_id == b_id, "fast_a": {m: a_hist[m] for m in METRICS},
                "fast_b": {m: b_hist[m] for m in METRICS}}
        plan["comparisons"].append(case)

    all_draws = list(dict.fromkeys(E1_DRAWS + [("Medium", 0.0)]))
    for draw in all_draws:
        env = P.PlanningEnv(inputs, evaluator, candidates)
        info = P.play(env, lambda e, obs: predict(model, e, obs), draw)
        expected = actions_expected[(actions_expected["path"] == draw[0]) & (actions_expected["w"] == draw[1])]
        for h, (_, row) in zip(info["history"], expected.iterrows(), strict=True):
            assert np.isclose(h["D_F"], row["D_F"], atol=1e-7, rtol=0), (draw, h["year"], h["D_F"], row["D_F"])
        plan["fast_histories"][("learned", draw)] = info
        for policy in ("do nothing", "greedy", "city plan"):
            plan["fast_histories"][(policy, draw)] = baseline[(policy, "named", draw)]
        if draw in E1_DRAWS:
            for index in (1, 3):
                add_comparison("E1", draw, P.STAGE_YEARS[index], "learned", info["history"][index],
                               "greedy", baseline[("greedy", "named", draw)]["history"][index])

    common_config = None
    common_prefix = None
    growth_rows = []
    for w in (0.0, 1.0):
        draw = ("Medium", w)
        def branch_point():
            env = P.PlanningEnv(inputs, evaluator, candidates)
            obs, _ = env.reset(options={"draw": draw})
            for _ in range(4):
                obs, _, _, _, _ = env.step(predict(model, env, obs))
            assert env.stage == 2 and env.sub == 0
            move = predict(model, env, obs)
            assert describe(env, move) == "move GFR_01 to C18"
            obs, _, _, _, _ = env.step(move)
            assert env.stage == 2 and env.sub == 1
            return env, obs
        env, obs = branch_point()
        cfg_id = configuration_id(env.config)
        prefix = copy.deepcopy(env.actions)
        if common_config is None:
            common_config, common_prefix = cfg_id, prefix
        assert cfg_id == common_config and prefix == common_prefix
        chosen = predict(model, env, obs)
        chosen_station = "E62" if w == 0 else "CO_29"
        assert describe(env, chosen) == f"+E at {chosen_station}"
        plan["e2_selection"].append({"state": f"S{int(w)}", "path": "Medium", "w": w,
                                     "decision_year": 2035, "substep": "vehicle", "common_config": cfg_id,
                                     "chosen": describe(env, chosen), "observation": obs.copy(),
                                     "mask": env.action_masks().copy(), "config": copy.deepcopy(env.config)})
        mult = P.growth_multipliers(inputs, *draw)
        for i, year in enumerate((2025, *P.STAGE_YEARS)):
            for z, tract in inputs["tracts"].iterrows():
                growth_rows.append({"w": w, "year": year, "tract": tract["tract"], "population_multiplier": mult[i, z],
                                    "population": mult[i, z] * tract["pop2025"]})
        for action_name, station in [("A", "E62"), ("B", "CO_29")]:
            branch_env, branch_obs = branch_point()
            slot = branch_env.config["name"].index(station)
            action = 1 + branch_env.n_build + branch_env.n_move + slot
            assert branch_env.action_masks()[action]
            branch_obs, _, terminated, _, info = branch_env.step(action)
            while not terminated:
                branch_obs, _, terminated, _, info = branch_env.step(predict(model, branch_env, branch_obs))
            plan["fast_histories"][(action_name, draw)] = info
        for i in (2, 3):
            add_comparison("E2", draw, P.STAGE_YEARS[i], "A", plan["fast_histories"][("A", draw)]["history"][i],
                           "B", plan["fast_histories"][("B", draw)]["history"][i])
    pd.DataFrame(growth_rows).to_csv(RESULTS / "e2_growth_by_tract.csv", index=False)
    # Equal county totals are a necessary matching check, not an extra experiment.
    totals = pd.DataFrame(growth_rows).groupby(["w", "year"])["population"].sum().unstack(0)
    assert np.allclose(totals[0.0], totals[1.0], atol=1e-6, rtol=0)
    plan["targets_fast"] = P.targets_of(evaluator, inputs)
    plan["target_sim"] = {"M_F": float((existing["M_F sim"] - existing["fire over 2025 sim"]).iloc[0]),
                          "M_O": float((existing["M_O sim"] - existing["other over 2025 sim"]).iloc[0])}
    plan["tract_ids"] = inputs["tracts"]["tract"].tolist()
    plan["fire_weight"] = evaluator.fire_tract_rate / evaluator.fire_tract_rate.sum()
    plan["existing_verification"] = existing.to_dict("records")
    (RESULTS / "plan.pkl").write_bytes(pickle.dumps(plan, protocol=5))
    dump_json(RESULTS / "comparisons_planned.json", plan["comparisons"])
    dump_json(RESULTS / "e2_state_action_contract.json", plan["e2_selection"])
    dump_json(RESULTS / "model_architecture.json", {"observation_size": model.observation_space.shape,
        "action_size": model.action_space.n, "policy": str(model.policy), "net_arch": model.policy.net_arch,
        "stage_years": P.STAGE_YEARS, "targets_fast": plan["targets_fast"], "target_sim": plan["target_sim"],
        "current_vehicles": [int(inputs["stations"]["engines"].sum()), int(inputs["stations"]["quints"].sum())],
        "current_stations": len(inputs["stations"]), "tracts": len(inputs["tracts"]),
        "fire_tracts": int((evaluator.fire_tract_rate > 0).sum()), "demand_points": len(inputs["points"]),
        "training_fire_arrivals": float(inputs["points"]["fire_rate"].sum() * P.WINDOW_HOURS),
        "training_other_arrivals": float(inputs["points"]["other_rate"].sum() * P.WINDOW_HOURS)})
    paths = [HERE / "task.md", Path(__file__), MODEL_DIR / "setup.json", MODEL_DIR / "baselines.pkl",
             MODEL_DIR / "final" / "model.zip", MODEL_DIR / "final" / "run.json", MODEL_DIR / "final" / "health.csv",
             environment.DISPATCHES_PARQUET, environment.ENGINES_CSV, environment.ROAD_NETWORK_GRAPHML,
             environment.AUTOMATIC_AID_FY2024, P.CENSUS_2010, P.CENSUS_2020, P.BEBR_FILE]
    paths += list((ROOT / "code" / "dispatch_simulator").glob("*.py"))
    paths += list((MODEL_DIR / "final").glob("*.csv"))
    manifest = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "python": sys.version,
        "packages": {x: importlib.metadata.version(x) for x in ("numpy", "pandas", "geopandas", "simpy", "torch", "sb3-contrib", "stable-baselines3")},
        "simulation_seeds": SEEDS, "bootstrap_replicates": REPS, "bootstrap_seed": BOOT_SEED,
        "model_frozen": True, "training_runs_added": 0,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size} for p in dict.fromkeys(paths)]}
    dump_json(HERE / "inputs_manifest.json", manifest)
    print(f"PREPARED in {time.time()-started:.1f}s; E1 comparisons=6; E2 comparisons=4; matched states and choices verified", flush=True)
    return plan


def tasks_from(plan):
    grouped = {}
    for case in plan["comparisons"]:
        if case["identical"]:
            continue
        key = (case["path"], case["w"], case["year"])
        grouped.setdefault(key, set()).update((case["a_config"], case["b_config"]))
    return [(key, seed, sorted(configs)) for key, configs in grouped.items() for seed in SEEDS]


def raw_path(key, seed, config_id):
    return RAW / f"{label(key[:2], key[2])}_s{seed}_{config_id}.npz"


def simulate_group(task):
    """One demand stream, reused by all needed configurations at that checkpoint."""
    key, seed, config_ids = task
    world, generator, inputs, _, _ = SETUP
    missing = [ident for ident in config_ids if not raw_path(key, seed, ident).exists()]
    if not missing:
        return {"case": label(key[:2], key[2]), "seed": seed, "simulated": 0, "seconds": 0}
    began = time.time()
    multipliers = P.growth_multipliers(inputs, *key[:2])[P.STAGE_YEARS.index(key[2]) + 1]
    scenario = P.grown_scenario(world, generator, inputs, multipliers, seed)
    for ident in missing:
        cfg = PLAN["configs"][ident]
        planned_world, planned_scenario = P.simulator_inputs(world, scenario, inputs, cfg)
        simulated = environment.simulate(planned_world, planned_scenario, policies.closest_engine, verbose=False)
        outcome = P.simulated_outcome(planned_world, planned_scenario, simulated, inputs)
        saved = {k: outcome[k] for k in ("fire_sum", "fire_count", "other_sum", "other_count", "fire_travel", "other_travel", "fire_waited", "other_waited")}
        saved.update(busy_ids=outcome["busy"].index.to_numpy(dtype=str), busy=outcome["busy"].to_numpy())
        target = raw_path(key, seed, ident)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **saved)
        temporary.replace(target)
    return {"case": label(key[:2], key[2]), "seed": seed, "simulated": len(missing), "seconds": time.time()-began}


def run_simulations(plan):
    global PLAN
    PLAN = plan
    tasks = tasks_from(plan)
    planned = sum(len(t[2]) for t in tasks)
    assert planned <= 300
    pending = [task for task in tasks if any(not raw_path(task[0], task[1], ident).exists() for ident in task[2])]
    print(f"SIMULATION PLAN: {planned} unique configuration-stream evaluations in {len(tasks)} groups; pending groups={len(pending)}", flush=True)
    log = HERE / "logs" / "experimental_simulations.jsonl"
    # ponytail: fork shares the existing in-memory world; no new simulator/cache framework.
    with mp.get_context("fork").Pool(min(8, len(pending) or 1)) as pool:
        for i, result in enumerate(pool.imap_unordered(simulate_group, pending), 1):
            with log.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(f"GROUP {i}/{len(pending)} {result['case']} seed={result['seed']} n={result['simulated']} {result['seconds']:.1f}s", flush=True)
    assert all(raw_path(t[0], t[1], ident).exists() for t in tasks for ident in t[2])
    return planned


def load_arrays(case, side):
    key = (case["path"], case["w"], case["year"])
    outcomes = []
    for seed in SEEDS:
        with np.load(raw_path(key, seed, case[f"{side}_config"])) as data:
            outcomes.append({k: data[k] for k in ("fire_sum", "fire_count", "other_sum", "other_count", "fire_waited", "other_waited")})
    return {k: np.stack([o[k] for o in outcomes]) for k in outcomes[0]}


def pooled(array, weights, index=None):
    """Same pooled means/weighted SD as P.pooled_metrics, with explicit missingness."""
    index = np.arange(len(SEEDS)) if index is None else index
    fs, fc, os_, oc = [array[k][index].sum(axis=0) for k in ("fire_sum", "fire_count", "other_sum", "other_count")]
    means = np.divide(fs, fc, out=np.full_like(fs, np.nan), where=fc > 0)
    valid = weights > 0
    missing = int(np.sum(valid & (fc == 0)))
    df = float("nan") if missing else P.weighted_sd(means, weights)
    return {"D_F": df, "M_F": fs.sum()/fc.sum(), "M_O": os_.sum()/oc.sum(), "tract_fire_mean": means,
            "fire_count": fc, "missing_fire_tracts": missing, "fire_incidents": int(fc.sum()),
            "waiting_fire": array["fire_waited"][index].sum()/fc.sum(), "waiting_other": array["other_waited"][index].sum()/oc.sum()}


def summarize(plan):
    weights = np.asarray(plan["fire_weight"])
    indices = np.random.default_rng(BOOT_SEED).integers(0, len(SEEDS), (REPS, len(SEEDS)))
    rows, tract_rows, seed_rows, effect_samples, level_rows = [], [], [], {}, []
    for case in plan["comparisons"]:
        common = {k: case[k] for k in ("experiment", "path", "w", "year", "a", "b", "identical")}
        if case["identical"]:
            old = next(r for r in plan["existing_verification"] if r["draw"] == f"{case['path']} w={case['w']:g}" and r["year"] == case["year"] and r["policy"] == "learned")
            a = b = {m: old[f"{m} sim"] for m in METRICS}
            samples = np.zeros((REPS, len(METRICS)))
            missing = 0
            source = "configuration identity; levels reused from original verification"
            for tract, weight in zip(plan["tract_ids"], weights):
                tract_rows.append({**common, "tract": tract, "fire_weight": weight, "a_minus_b": 0.0, "source": source})
        else:
            aa, bb = load_arrays(case, "a"), load_arrays(case, "b")
            a, b = pooled(aa, weights), pooled(bb, weights)
            assert np.array_equal(a["fire_count"], b["fire_count"]), "paired policies changed the measured fire incident cohort"
            missing = max(a["missing_fire_tracts"], b["missing_fire_tracts"])
            samples = np.array([[pooled(aa, weights, ix)[m] - pooled(bb, weights, ix)[m] for m in METRICS] for ix in indices])
            source = "new paired event simulation"
            for i, tract in enumerate(plan["tract_ids"]):
                tract_rows.append({**common, "tract": tract, "fire_weight": weights[i], "a_mean": a["tract_fire_mean"][i],
                    "b_mean": b["tract_fire_mean"][i], "a_minus_b": a["tract_fire_mean"][i]-b["tract_fire_mean"][i],
                    "fire_incidents": a["fire_count"][i], "source": source})
            for i, seed in enumerate(SEEDS):
                am, bm = pooled(aa, weights, [i]), pooled(bb, weights, [i])
                seed_rows.append({**common, "seed": seed, **{f"a_{m}": am[m] for m in METRICS},
                    **{f"b_{m}": bm[m] for m in METRICS}, "a_missing_fire_tracts": am["missing_fire_tracts"],
                    "b_missing_fire_tracts": bm["missing_fire_tracts"]})
            for side, measured in [("a", a), ("b", b)]:
                level_rows.append({**common, "side": side, "policy_or_action": case[side],
                    **{m: measured[m] for m in METRICS}, "waiting_fire": measured["waiting_fire"],
                    "waiting_other": measured["waiting_other"], "missing_fire_tracts": measured["missing_fire_tracts"],
                    "fire_over_2025": measured["M_F"]-plan["target_sim"]["M_F"],
                    "other_over_2025": measured["M_O"]-plan["target_sim"]["M_O"],
                    "violates_point_target": any(measured[m]-plan["target_sim"][m] > P.TARGET_TOLERANCE_MIN for m in ("M_F", "M_O"))})
        row = {**common, "source": source, "missing_fire_tracts": missing}
        for j, metric in enumerate(METRICS):
            low, high = np.quantile(samples[:, j], [0.025, 0.975]) if np.isfinite(samples[:, j]).all() else (np.nan, np.nan)
            row.update({f"a_{metric}": a[metric], f"b_{metric}": b[metric], f"delta_{metric}": a[metric]-b[metric],
                        f"low_{metric}": low, f"high_{metric}": high,
                        f"invalid_bootstrap_{metric}": int((~np.isfinite(samples[:, j])).sum()),
                        f"fast_delta_{metric}": case["fast_a"][metric]-case["fast_b"][metric]})
        rows.append(row)
        effect_samples[(case["experiment"], case["path"], case["w"], case["year"])] = samples
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "paired_comparisons.csv", index=False)
    pd.DataFrame(tract_rows).to_csv(RESULTS / "spatial_contrasts.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(RESULTS / "paired_seed_metrics.csv", index=False)
    pd.DataFrame(level_rows).to_csv(RESULTS / "simulated_levels.csv", index=False)
    e2_rows = []
    means = {}
    for w in (0.0, 1.0):
        sub = table[(table.experiment == "E2") & (table.w == w)]
        samples = np.mean([effect_samples[("E2", "Medium", w, year)] for year in (2040, 2045)], axis=0)
        means[w] = samples
        row = {"path": "Medium", "w": w, "years": "2040,2045", "contrast": "A(E62) - B(CO_29)",
               "chosen": "A" if w == 0 else "B"}
        for j, metric in enumerate(METRICS):
            lo, hi = np.quantile(samples[:, j], [0.025, 0.975]) if np.isfinite(samples[:, j]).all() else (np.nan, np.nan)
            row.update({f"delta_{metric}": sub[f"delta_{metric}"].mean(), f"low_{metric}": lo, f"high_{metric}": hi,
                        f"fast_delta_{metric}": sub[f"fast_delta_{metric}"].mean()})
        for action_name in ("A", "B"):
            history = plan["fast_histories"][(action_name, ("Medium", w))]["history"][2:]
            row[f"fast_remaining_return_{action_name}"] = -sum(P.stage_cost(h, plan["targets_fast"]) for h in history)
        e2_rows.append(row)
    interaction = {"contrast": "[A-B at w=1] - [A-B at w=0]", "path": "Medium"}
    for j, metric in enumerate(METRICS):
        samples = means[1.0][:, j]-means[0.0][:, j]
        lo, hi = np.quantile(samples, [0.025, 0.975]) if np.isfinite(samples).all() else (np.nan, np.nan)
        interaction.update({f"delta_{metric}": e2_rows[1][f"delta_{metric}"]-e2_rows[0][f"delta_{metric}"], f"low_{metric}": lo, f"high_{metric}": hi})
    pd.DataFrame(e2_rows).to_csv(RESULTS / "e2_remaining_horizon.csv", index=False)
    pd.DataFrame([interaction]).to_csv(RESULTS / "e2_condition_contrast.csv", index=False)
    fast_rows, action_rows = [], []
    for (policy, draw), info in plan["fast_histories"].items():
        for h in info["history"]:
            for i, tract in enumerate(plan["tract_ids"]):
                fast_rows.append({"policy": policy, "path": draw[0], "w": draw[1], "year": h["year"], "tract": tract,
                                  "fire_weight": weights[i], "fire_mean": h["tract_fire_mean"][i]})
            action_rows.append({"policy": policy, "path": draw[0], "w": draw[1], "evaluation_year": h["year"],
                "decision_year": h["year"]-5, "actions": json.dumps(h["actions"], default=jsonable),
                **{m: h[m] for m in METRICS}, "violation": h["violation"], "config_id": configuration_id(h["config"])})
    pd.DataFrame(fast_rows).to_csv(RESULTS / "fast_tract_outcomes.csv", index=False)
    pd.DataFrame(action_rows).to_csv(RESULTS / "frozen_trajectories.csv", index=False)
    np.savez_compressed(RESULTS / "bootstrap_draw_indices.npz", indices=indices)
    manifest = json.loads((HERE / "inputs_manifest.json").read_text())
    changed = [r["path"] for r in manifest["inputs"] if r["path"] != str(Path(__file__).relative_to(ROOT)) and sha(ROOT/r["path"]) != r["sha256"]]
    assert not changed, f"inputs changed during experiments: {changed}"
    tasks = tasks_from(plan)
    report = {"status": "complete", "unique_config_stream_evaluations": sum(len(t[2]) for t in tasks),
              "raw_files": len(list(RAW.glob('*.npz'))), "skipped_identical_E1_checkpoints": int(table[(table.experiment=="E1")].identical.sum()),
              "simulation_seeds": SEEDS, "bootstrap_replicates": REPS, "training_runs_added": 0,
              "original_inputs_unchanged": True, "all_E1_E2_comparisons": len(table),
              "missing_pooled_fire_tracts_max": int(table.missing_fire_tracts.max()),
              "elapsed_note": "Per-group timings are in logs/experimental_simulations.jsonl", "e2_selection_verified": True}
    dump_json(RESULTS / "execution_summary.json", report)
    print(table[["experiment", "path", "w", "year", "delta_D_F", "low_D_F", "high_D_F", "delta_M_F", "delta_M_O"]].to_string(index=False), flush=True)
    print("E2 remaining horizon:\n" + pd.DataFrame(e2_rows).to_string(index=False), flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "summarize":
        summarize(pickle.loads((RESULTS / "plan.pkl").read_bytes()))
    elif mode in ("prepare", "all"):
        plan = prepare()
        if mode == "all":
            run_simulations(plan)
            summarize(plan)
    else:
        raise SystemExit("use all, prepare, or summarize")
