"""Train and evaluate the multi-stage planning policy (research_planning.md 4.6-4.7), reusing train.py's PPO pieces.

Commands, run from the project root (after .aris/traces/planning_rl/validate_evaluator.py has written the setup):
    python code/dispatch_simulator/plan_train.py search     ASHA over PPO settings; objective = validation improvement
    python code/dispatch_simulator/plan_train.py final      the final run with the selected trial's settings (fewest validation
                                                            violations, then the largest D reduction), with the health checks
    python code/dispatch_simulator/plan_train.py baselines  the three baselines on the evaluation draws (cached; no model needed)
    python code/dispatch_simulator/plan_train.py evaluate   the frozen policy (only if healthy) against the three baselines on
                                                            new draws, with the RQ3 action table and maps
    python code/dispatch_simulator/plan_train.py verify     the four policies' configurations in the dispatch simulator
"""
import json
import multiprocessing
import pickle
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import environment
import planning as P
import policies
import scenarios
import train

PLAN_GAMMA = 1.0                                      # [DESIGN] no discount over the four stages
SEARCH_RUNGS = [65_536, 196_608, 589_824]             # [DESIGN] training steps at which trials are compared (8 steps per episode); multiples of every rollout size (8 environments x 512, 1024 or 4096)
SEARCH_TRIALS, SEARCH_WORKERS, ENVS = 12, 4, 8        # [DESIGN] trials, parallel workers, environments per model
VALIDATION_DRAWS, VALIDATION_SEED = 24, 20261116      # [DESIGN] fixed growth draws of the objective and the monitor
EVALUATION_DRAWS, EVALUATION_SEED = 48, 20261117      # [DESIGN] new growth draws of the evaluation
FINAL_STEPS, EVAL_EVERY = 1_048_576, 65_536           # [DESIGN] steps of the final run (about one million, the author's budget; extended only without a convergence trend), steps between two validations
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 7                   # [STANDARD] bootstrap resamples over whole draws
VERIFY_SEEDS = list(range(101, 116))                  # [DESIGN] generated incident streams pooled in the simulator verification, the 15 seeds the panel needed
NAME = "plan_search_fire"


def load_setup():
    """Build the world and the frozen planning setup inside the calling process: (world, generator, inputs, evaluator, candidates)."""
    torch.set_num_threads(1)
    world = environment.build_world(verbose=False)
    generator = scenarios.fit_generator(world, verbose=False)
    return (world, generator, *P.prepare(world, generator))


def vec_env(inputs, evaluator, candidates, base_seed, n_envs=ENVS):
    """n_envs planning environments in this process, with episode returns recorded (VecMonitor around DummyVecEnv)."""
    return VecMonitor(DummyVecEnv([(lambda seed=base_seed + i: ActionMasker(P.PlanningEnv(inputs, evaluator, candidates, seed=seed),
                                                                            lambda env: env.action_masks())) for i in range(n_envs)]))


def play_model(env, model, draw):
    """Play one draw with the deterministic policy; return the final info."""
    return P.play(env, lambda e, observation: int(model.predict(observation, action_masks=e.action_masks(), deterministic=True)[0]), draw)


def summary(info):
    """One row of results for a played draw: return, mean fire inequality D_F and means M_F, M_O over the four years,
    draw-years violating either mean constraint (and each one), and actions taken."""
    history = info["history"]
    return {"return": info["unshaped_return"], "mean D_F": float(np.mean([h["D_F"] for h in history])),
            "mean M_F": float(np.mean([h["M_F"] for h in history])), "mean M_O": float(np.mean([h["M_O"] for h in history])),
            "violations": int(sum(h["violation"] for h in history)), "fire violations": int(sum(h["violation_fire"] for h in history)),
            "other violations": int(sum(h["violation_other"] for h in history)), "actions": int(info["deviations"])}


class Validator:
    """The fixed validation draws, with the do-nothing results computed once.

    Output
        compare(model) -> DataFrame, one row per draw: path, w, the summary columns, improvement (the policy's return
        minus do-nothing's) and D_F reduction (do-nothing's mean D_F minus the policy's).
    """

    def __init__(self, inputs, evaluator, candidates, count=VALIDATION_DRAWS, seed=VALIDATION_SEED):
        self.env = P.PlanningEnv(inputs, evaluator, candidates)
        self.draws = P.validation_draws(count, seed)
        self.nothing = [summary(P.play(self.env, P.do_nothing, draw)) for draw in self.draws]

    def compare(self, model):
        rows = []
        for draw, base in zip(self.draws, self.nothing):
            row = summary(play_model(self.env, model, draw))
            rows.append({"path": draw[0], "w": draw[1], **row, "improvement": row["return"] - base["return"],
                         "D_F reduction": base["mean D_F"] - row["mean D_F"]})
        return pd.DataFrame(rows)


class PlanSignals(BaseCallback):
    """Every eval_every steps, record the validation: eval/improvement_mean, eval/violation_share, eval/DF_reduction_mean."""

    def __init__(self, validator, eval_every):
        super().__init__()
        self.validator, self.eval_every, self.next_eval = validator, eval_every, 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.num_timesteps >= self.next_eval:
            record(self.model, self.validator.compare(self.model))
            self.next_eval = self.num_timesteps + self.eval_every


def record(model, table):
    """Write a validation table's means into the model's training signals."""
    model.logger.record("eval/improvement_mean", float(table["improvement"].mean()))
    model.logger.record("eval/violation_share", float(table["violations"].sum() / (4 * len(table))))
    model.logger.record("eval/DF_reduction_mean", float(table["D_F reduction"].mean()))


def objective(trial):
    """One ASHA trial: train, report the validation improvement at each rung, stop if pruned."""
    _, _, inputs, evaluator, candidates = SETUP
    settings = train.sample_settings(trial, small_system=True)
    seed = 1000 * (trial.number + 1)
    env = vec_env(inputs, evaluator, candidates, seed)
    model = train.build_model(env, settings, seed=seed, log_dir=P.MODEL_DIR / "search_trials" / f"trial_{trial.number}", gamma=PLAN_GAMMA)
    validator = Validator(inputs, evaluator, candidates)
    for rung in SEARCH_RUNGS:
        model.learn(total_timesteps=rung - model.num_timesteps, reset_num_timesteps=False)
        table = validator.compare(model)
        record(model, table)
        model.logger.record("time/total_timesteps", model.num_timesteps)
        model.logger.dump(model.num_timesteps)
        improvement = float(table["improvement"].mean())
        trial.report(improvement, model.num_timesteps)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return improvement


def initial_reward(model, inputs, evaluator, candidates, episodes=100, seed=900_000):
    """Mean episode return of the policy before training, sampling its actions as in training."""
    env = P.PlanningEnv(inputs, evaluator, candidates, seed=seed)
    returns = []
    for _ in range(episodes):
        info = P.play(env, lambda e, observation: int(model.predict(observation, action_masks=e.action_masks(), deterministic=False)[0]), None)
        returns.append(info["unshaped_return"])
    return float(np.mean(returns))


def final_run(settings, name="final", seed=500_000):
    """Train with the chosen settings, validate every EVAL_EVERY steps, save the model and the health checks."""
    _, _, inputs, evaluator, candidates = SETUP
    run_dir = P.MODEL_DIR / name
    env = vec_env(inputs, evaluator, candidates, seed)
    model = train.build_model(env, settings, seed=seed, log_dir=run_dir, gamma=PLAN_GAMMA)
    validator = Validator(inputs, evaluator, candidates)
    start_reward = initial_reward(model, inputs, evaluator, candidates)
    record(model, validator.compare(model))
    model.logger.record("time/total_timesteps", 0)
    model.logger.dump(0)
    started = time.time()
    model.learn(total_timesteps=FINAL_STEPS, callback=PlanSignals(validator, EVAL_EVERY))
    model.save(run_dir / "model")
    final = validator.compare(model)
    final.to_csv(run_dir / "validation_draws.csv", index=False)
    record(model, final)
    model.logger.record("time/total_timesteps", model.num_timesteps)
    model.logger.dump(model.num_timesteps)
    progress = pd.read_csv(run_dir / "progress.csv")
    start_entropy = float(-progress["train/entropy_loss"].dropna().iloc[0])
    checks = train.acceptance(progress, start_reward, entropy_ceiling=np.inf)          # entropy: only the late-trend check (masks change the valid actions)
    checks.to_csv(run_dir / "health.csv")
    (run_dir / "run.json").write_text(json.dumps({"settings": settings, "seed": seed, "initial_reward": start_reward,
                                                  "start_entropy": start_entropy, "hours": (time.time() - started) / 3600}, indent=2))
    print(f"final run: {model.num_timesteps:,} steps in {(time.time() - started) / 3600:.2f} h; untrained return {start_reward:.3f}\n"
          f"{checks.to_string()}\nvalidation: improvement {final['improvement'].mean():.4f}, D_F reduction {final['D_F reduction'].mean():.4f} min, "
          f"violations {int(final['violations'].sum())} of {4 * len(final)} draw-years\n")
    return checks


def select_trial():
    """The search's choice (research_planning.md 4.6): among the trials at the highest rung reached, the fewest
    validation violations (either constraint) first, then the largest reduction of mean D_F. Returns (trial number,
    settings, table)."""
    study = train.open_study(NAME, SEARCH_RUNGS[0], model_dir=P.MODEL_DIR)
    rows = []
    for trial in study.trials:
        if trial.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) or not trial.intermediate_values:
            continue
        progress = pd.read_csv(P.MODEL_DIR / "search_trials" / f"trial_{trial.number}" / "progress.csv")
        last = progress.dropna(subset=["eval/violation_share"]).iloc[-1]
        rows.append({"trial": trial.number, "rung": max(trial.intermediate_values), "violation share": last["eval/violation_share"],
                     "D_F reduction": last["eval/DF_reduction_mean"], "improvement": last["eval/improvement_mean"]})
    table = pd.DataFrame(rows)
    top = table[table["rung"] == table["rung"].max()].sort_values(["violation share", "D_F reduction"], ascending=[True, False])
    chosen = int(top["trial"].iloc[0])
    print(f"trials at the highest rung, in selection order:\n{top.round(4).to_string(index=False)}\nselected trial {chosen}")
    return chosen, study.trials[chosen].params, table


def evaluation_draws():
    """The evaluation draws: EVALUATION_DRAWS new draws plus the named draws (each path with w = 0 and w = 1)."""
    named = [(path, w) for path in P.PATH_WEIGHTS for w in (0.0, 1.0)]
    return [("draw", d) for d in P.validation_draws(EVALUATION_DRAWS, EVALUATION_SEED)] + [("named", d) for d in named]


def play_baseline(task):
    """Play one draw with one baseline (for the process pool); return (policy, kind, draw, info without configurations)."""
    policy, kind, draw = task
    _, _, inputs, evaluator, candidates = SETUP
    env = P.PlanningEnv(inputs, evaluator, candidates)
    choose = {"do nothing": P.do_nothing, "city plan": P.city_plan, "greedy": P.StageGreedy()}[policy]
    return policy, kind, draw, P.play(env, choose, draw)


def baseline_runs(draws):
    """Do-nothing, the city plan and the stage greedy rule on the given (kind, draw) pairs; the results are cached in
    MODEL_DIR/baselines.pkl (keyed by policy, kind and draw), since they do not depend on the learned model."""
    path = P.MODEL_DIR / "baselines.pkl"
    cached = pickle.loads(path.read_bytes()) if path.exists() else {}
    missing = [(policy, kind, draw) for policy in ("do nothing", "city plan", "greedy") for kind, draw in draws if (policy, kind, draw) not in cached]
    if missing:
        with multiprocessing.get_context("fork").Pool(8) as pool:
            for policy, kind, draw, info in pool.map(play_baseline, missing):
                cached[(policy, kind, draw)] = info
        path.write_bytes(pickle.dumps(cached))
    return cached


def evaluate(name="final"):
    """The frozen policy against do-nothing, the city plan and the stage greedy rule on the evaluation draws."""
    _, _, inputs, evaluator, candidates = SETUP
    health = pd.read_csv(P.MODEL_DIR / name / "health.csv", index_col=0)
    if not health["passes"].astype(str).eq("True").all():
        raise SystemExit(f"the final model is not healthy; evaluation not run:\n{health.to_string()}")
    model = MaskablePPO.load(P.MODEL_DIR / name / "model")
    env = P.PlanningEnv(inputs, evaluator, candidates)
    draws = evaluation_draws()
    played = {("learned", kind, draw): play_model(env, model, draw) for kind, draw in draws}
    cached = baseline_runs(draws)
    played.update({key: cached[key] for key in cached if (key[1], key[2]) in draws})
    rows, years = [], []
    for (policy, kind, draw), info in played.items():
        rows.append({"policy": policy, "kind": kind, "path": draw[0], "w": draw[1], **summary(info)})
        for h in info["history"]:
            years.append({"policy": policy, "kind": kind, "path": draw[0], "w": draw[1], "year": h["year"], "M_F": h["M_F"], "D_F": h["D_F"],
                          "M_O": h["M_O"], "M": h["M"], "D": h["D"], "D_fixed": h["D_fixed"], "violation": h["violation"],
                          "violation_fire": h["violation_fire"], "violation_other": h["violation_other"], "waiting": h["waiting"], "waiting_fire": h["waiting_fire"], "station action": h["actions"][0][0],
                          "station slot": h["actions"][0][1], "candidate": h["actions"][0][2], "vehicle action": h["actions"][1][0],
                          "vehicle slot": h["actions"][1][1]})
    results, by_year = pd.DataFrame(rows), pd.DataFrame(years)
    out = P.MODEL_DIR / name
    results.to_csv(out / "evaluation_draws.csv", index=False)
    by_year.to_csv(out / "evaluation_years.csv", index=False)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = results[results["kind"] == "draw"]
    learned = sampled[sampled["policy"] == "learned"].reset_index(drop=True)
    comparisons = []
    for baseline in ("do nothing", "city plan", "greedy"):
        other = sampled[sampled["policy"] == baseline].reset_index(drop=True)
        for metric, sign in (("mean D_F", -1), ("mean M_F", -1), ("mean M_O", -1), ("violations", -1), ("return", 1)):
            gaps = sign * (learned[metric] - other[metric]).to_numpy()            # positive = the learned policy is better
            boots = [gaps[rng.integers(0, len(gaps), len(gaps))].mean() for _ in range(BOOTSTRAP)]
            comparisons.append({"baseline": baseline, "metric": metric, "learned better by (mean)": gaps.mean(),
                                "low 95%": np.quantile(boots, 0.025), "high 95%": np.quantile(boots, 0.975),
                                "draws learned better": int((gaps > 1e-9).sum()), "draws worse": int((gaps < -1e-9).sum())})
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(out / "evaluation_comparisons.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"--- evaluation, {len(learned)} new draws\n{sampled.groupby('policy')[['return', 'mean D_F', 'mean M_F', 'mean M_O', 'violations', 'fire violations', 'other violations', 'actions']].mean().round(4).to_string()}\n"
          f"\n--- paired differences (positive = learned better; D_F, M_F, M_O in minutes; violations in draw-years)\n{comparisons.round(4).to_string(index=False)}\n"
          f"\n--- by path (new draws)\n{sampled.groupby(['path', 'policy'])[['mean D_F', 'mean M_F', 'mean M_O', 'violations']].mean().round(4).to_string()}\n"
          f"\n--- named draws\n{results[results['kind'] == 'named'].round(4).to_string(index=False)}\n")
    rq3_outputs(env, model, name)
    return results, by_year, comparisons


def rq3_outputs(env, model, name):
    """RQ3: the learned actions on the named draws, as a matched table (w = 0 against w = 1 on each path, and paths
    against each other for a fixed w) and as maps of the actions by stage with candidate sites and relocation eligibility."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites, start, out = env.inputs["sites"], env.start, P.MODEL_DIR / name
    named = [(path, w) for path in P.PATH_WEIGHTS for w in (0.0, 1.0)]
    histories = {draw: play_model(env, model, draw)["history"] for draw in named}

    def describe(action):
        kind, slot, k = action
        if kind == "build":
            return f"build at C{k}"
        if kind == "move":
            return f"move {start['name'][slot]} to C{k}"
        if kind in ("engine", "quint"):
            return f"+{'Q' if kind == 'quint' else 'E'} at {start['name'][slot]}"
        return "-"
    rows = [{"path": path, "w": w, "year": h["year"], "station action": describe(h["actions"][0]), "vehicle action": describe(h["actions"][1]),
             "D_F": h["D_F"], "M_F": h["M_F"], "M_O": h["M_O"], "violation": h["violation"]} for (path, w), history in histories.items() for h in history]
    table = pd.DataFrame(rows)
    table.to_csv(out / "rq3_actions.csv", index=False)
    pd.set_option("display.width", 250)
    print("--- RQ3: learned actions on the named draws\n" + table.round(4).to_string(index=False))
    for path in P.PATH_WEIGHTS:
        a, b = table[(table["path"] == path) & (table["w"] == 0.0)], table[(table["path"] == path) & (table["w"] == 1.0)]
        same = int(((a["station action"].to_numpy() == b["station action"].to_numpy()) & (a["vehicle action"].to_numpy() == b["vehicle action"].to_numpy())).sum())
        print(f"{path}: stages with identical actions for w = 0 and w = 1: {same} of 4")
    for w in (0.0, 1.0):
        parts = [table[(table["path"] == path) & (table["w"] == w)] for path in P.PATH_WEIGHTS]
        same = int(np.all([(p["station action"].to_numpy() == parts[0]["station action"].to_numpy()) & (p["vehicle action"].to_numpy() == parts[0]["vehicle action"].to_numpy()) for p in parts], axis=0).sum())
        print(f"w = {w:g}: stages with identical actions on all three paths: {same} of 4")
    figure, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    xy = sites[["x", "y"]].to_numpy() / 1000
    for axis, draw in zip(axes.ravel(), named):
        current = start["site"][start["site"] >= 0]
        axis.scatter(xy[current, 0], xy[current, 1], s=18, c="#555555", label="current station", zorder=3)
        axis.scatter(xy[env.candidates, 0], xy[env.candidates, 1], s=40, facecolors="none", edgecolors="#888888", label="candidate site", zorder=2)
        for slot in np.flatnonzero(start["site"] >= 0):
            gaps = np.hypot(*(xy[env.candidates] - xy[start["site"][slot]]).T)
            for k in np.flatnonzero(gaps <= P.RELOCATION_KM):
                axis.plot([xy[start["site"][slot], 0], xy[env.candidates[k], 0]], [xy[start["site"][slot], 1], xy[env.candidates[k], 1]],
                          color="#dddddd", linewidth=0.6, zorder=1)
        before = start
        for h in histories[draw]:
            kind, slot, k = h["actions"][0]
            if kind == "build":
                axis.scatter(*xy[env.candidates[k]], marker="*", s=180, c="#d17a00", zorder=4)
                axis.annotate(str(h["year"] - 5), xy[env.candidates[k]], textcoords="offset points", xytext=(5, 5), fontsize=8)
            elif kind == "move":                                       # the arrow starts where the station stood before this stage
                axis.annotate("", xy=xy[env.candidates[k]], xytext=xy[before["site"][slot]], arrowprops={"arrowstyle": "->", "color": "#7a3fa0"}, zorder=4)
                axis.annotate(str(h["year"] - 5), xy[env.candidates[k]], textcoords="offset points", xytext=(5, -10), fontsize=8)
            kind, slot, _ = h["actions"][1]
            if kind in ("engine", "quint"):
                position = xy[h["config"]["site"][slot]]
                axis.annotate(f"+{'Q' if kind == 'quint' else 'E'} {h['year'] - 5}", position, textcoords="offset points", xytext=(-30, 8), fontsize=7, color="#2f6f5e")
            before = h["config"]
        axis.set_title(f"{draw[0]} path, w = {draw[1]:g}", fontsize=10)
        axis.set_aspect("equal")
        axis.set_xlabel("UTM 17N easting (km)")
        axis.set_ylabel("northing (km)")
    axes[0, 0].legend(fontsize=8, loc="lower left", frameon=False)
    figure.suptitle("Learned plan on the named draws: stars = new stations, arrows = moves, +E/+Q = added vehicles (labels: decision year); "
                    "grey lines = moves allowed in 2025", fontsize=10)
    figure.savefig(out / "rq3_action_maps.png", dpi=140)
    print(f"wrote rq3_actions.csv and rq3_action_maps.png to {out}")


def verify_simulate(task):
    """Simulate one configuration on one stored scenario (for the process pool); return (key, seed, policy, outcome)."""
    key, seed, policy, config = task
    world, _, inputs, _, _ = SETUP
    planned_world, planned_scenario = P.simulator_inputs(world, VERIFY_SCENARIOS[(key, seed)], inputs, config)
    simulated = environment.simulate(planned_world, planned_scenario, policies.closest_engine, verbose=False)
    return key, seed, policy, P.simulated_outcome(planned_world, planned_scenario, simulated, inputs)


def verify(name="final"):
    """Simulate the four policies' configurations on the medium-trend, high-trend and high-proportional named draws at
    2035 and 2045, on VERIFY_SEEDS generated incident streams shared by the policies (pooled as in the panel), against the
    simulator's own 2025 comparator (the current configuration at 2025 demand on the same seeds)."""
    global VERIFY_SCENARIOS
    from scipy import stats
    world, generator, inputs, evaluator, candidates = SETUP
    health = pd.read_csv(P.MODEL_DIR / name / "health.csv", index_col=0)
    if not health["passes"].astype(str).eq("True").all():
        raise SystemExit(f"the final model is not healthy; verification not run:\n{health.to_string()}")
    model = MaskablePPO.load(P.MODEL_DIR / name / "model")
    env = P.PlanningEnv(inputs, evaluator, candidates)
    cases = [("Medium", 1.0), ("High", 1.0), ("High", 0.0)]
    cached = baseline_runs([("named", draw) for draw in cases])
    played = {draw: {"learned": play_model(env, model, draw), **{policy: cached[(policy, "named", draw)] for policy in ("do nothing", "city plan", "greedy")}}
              for draw in cases}
    demands = {("2025", 0): np.ones(len(inputs["tracts"]))}
    for draw in cases:
        multipliers = P.growth_multipliers(inputs, *draw)
        demands.update({(f"{draw[0]} w={draw[1]:g}", year): multipliers[index] for index, year in [(2, 2035), (4, 2045)]})
    VERIFY_SCENARIOS = {(key, seed): P.grown_scenario(world, generator, inputs, m, seed) for key, m in demands.items() for seed in VERIFY_SEEDS}
    tasks = [(("2025", 0), seed, "current", P.start_configuration(inputs)) for seed in VERIFY_SEEDS]
    for draw in cases:
        for index, year in [(2, 2035), (4, 2045)]:
            for policy, info in played[draw].items():
                tasks += [((f"{draw[0]} w={draw[1]:g}", year), seed, policy, info["history"][index - 1]["config"]) for seed in VERIFY_SEEDS]
    with multiprocessing.get_context("fork").Pool(8) as pool:
        outcomes = {(key, seed, policy): outcome for key, seed, policy, outcome in pool.map(verify_simulate, tasks)}
    target = P.pooled_metrics([outcomes[(("2025", 0), s, "current")] for s in VERIFY_SEEDS], inputs)
    seed_target = {s: P.pooled_metrics([outcomes[(("2025", 0), s, "current")]], inputs) for s in VERIFY_SEEDS}
    half = lambda values: stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
    rows = []
    for draw in cases:
        label = f"{draw[0]} w={draw[1]:g}"
        for index, year in [(2, 2035), (4, 2045)]:
            for policy, info in played[draw].items():
                sim = P.pooled_metrics([outcomes[((label, year), s, policy)] for s in VERIFY_SEEDS], inputs)
                seeds = {s: P.pooled_metrics([outcomes[((label, year), s, policy)]], inputs) for s in VERIFY_SEEDS}
                nothing = {s: P.pooled_metrics([outcomes[((label, year), s, "do nothing")]], inputs) for s in VERIFY_SEEDS}
                fast = info["history"][index - 1]
                row = {"draw": label, "year": year, "policy": policy, "M_F sim": sim["M_F"], "D_F sim": sim["D_F"], "M_O sim": sim["M_O"],
                       "P90_F sim": sim["P90_F"], "waiting fire sim": sim["waiting_fire"],
                       "fire over 2025 sim": sim["M_F"] - target["M_F"], "fire over half": half([seeds[s]["M_F"] - seed_target[s]["M_F"] for s in VERIFY_SEEDS]),
                       "other over 2025 sim": sim["M_O"] - target["M_O"], "other over half": half([seeds[s]["M_O"] - seed_target[s]["M_O"] for s in VERIFY_SEEDS]),
                       "violation sim": (sim["M_F"] - target["M_F"] > P.TARGET_TOLERANCE_MIN) or (sim["M_O"] - target["M_O"] > P.TARGET_TOLERANCE_MIN),
                       "D_F minus do nothing sim": sim["D_F"] - P.pooled_metrics([outcomes[((label, year), s, "do nothing")] for s in VERIFY_SEEDS], inputs)["D_F"],
                       "D_F minus do nothing half": half([seeds[s]["D_F"] - nothing[s]["D_F"] for s in VERIFY_SEEDS]),
                       "M_F fast": fast["M_F"], "D_F fast": fast["D_F"], "M_O fast": fast["M_O"], "violation fast": fast["violation"]}
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(P.MODEL_DIR / name / "verification.csv", index=False)
    pd.set_option("display.width", 260)
    print(f"simulator comparator (current configuration, 2025, {len(VERIFY_SEEDS)} seeds): M_F {target['M_F']:.4f}, M_O {target['M_O']:.4f}, D_F {target['D_F']:.4f}")
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "worker":
        SETUP = load_setup()
        train.open_study(NAME, SEARCH_RUNGS[0], worker=int(sys.argv[3]), model_dir=P.MODEL_DIR).optimize(objective, n_trials=int(sys.argv[2]))
    elif command == "search":
        train.run_search(NAME, "worker", SEARCH_RUNGS[0], SEARCH_WORKERS, SEARCH_TRIALS, script=__file__, model_dir=P.MODEL_DIR)
    elif command == "final":                                        # "final N": trial N's settings instead of the selected trial's
        SETUP = load_setup()
        settings = (train.open_study(NAME, SEARCH_RUNGS[0], model_dir=P.MODEL_DIR).trials[int(sys.argv[2])].params if len(sys.argv) > 2
                    else select_trial()[1])
        final_run(settings)
    elif command == "baselines":                                    # the baselines on the evaluation draws, cached before the model exists
        SETUP = load_setup()
        baseline_runs(evaluation_draws())
    elif command == "evaluate":
        SETUP = load_setup()
        evaluate()
    elif command == "verify":
        SETUP = load_setup()
        verify()
