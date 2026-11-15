"""Train the learned dispatcher: the learnability gate on the small system, ASHA searches over PPO settings, and the final run.

Commands, run from the project root:
    python code/dispatch_simulator/train.py e2-search   ASHA on the small system; the objective is the exact share of the
                                                        protocol-to-optimum improvement the agent captures at fire weight 8
    python code/dispatch_simulator/train.py e2-gate     the gate with the best small-system settings, at every fire weight
                                                        (e2-gate shaping: the same with the readiness shaping term)
    python code/dispatch_simulator/train.py search      ASHA on the full environment; the objective is the improvement over
                                                        the protocol on fixed validation days
    python code/dispatch_simulator/train.py final       the final run with the best settings of the search ("final N": trial
                                                        N's settings; "final N SEED NAME": also its seed and folder); it
                                                        writes the model, the training signals and the acceptance checks

ASHA (asynchronous successive halving) starts many trials with a small budget, keeps the best third at each rung and
gives them three times the budget. Optuna's SuccessiveHalvingPruner implements it. Several worker processes share one
study, so a trial is promoted or stopped as soon as its rung has results, without waiting for the others.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import log

import numpy as np
import optuna
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import e2_small_system as e2
from environment import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "dispatch_rl"     # [DESIGN] models, training signals and search studies
GAMMA = 0.97                                                     # [DESIGN] discount per decision, fixed by the design
REDUCTION_FACTOR = 3                                             # [STANDARD] ASHA keeps the best third of a rung and triples the budget
E2_ENVS = 8                                                      # [DESIGN] small-system copies stepped together in one process
E2_RUNGS = [98_304, 294_912, 884_736, 2_654_208]                 # [DESIGN] training decisions at which small-system trials are reported; multiples of every rollout size (8 x 512, 1024 or 4096), so each is reached exactly; pruning starts at the second
E2_TRIALS, E2_WORKERS = 27, 9                                    # [DESIGN] trials and parallel workers of the small-system search
E2_GATE_SEEDS = [101, 102]                                       # [DESIGN] fresh training seeds of the gate; every seed must pass
ELAPSED_HALF_LIFE_H = 5.4                                        # [DESIGN] half-life of the elapsed-time discount compared with the per-decision discount (the full system's half-life of gamma)
SEARCH_RUNGS = [401_408, 1_204_224, 3_612_672]                   # [DESIGN] training decisions at which full-environment trials are compared; multiples of every rollout size (4 x 256, 512 or 1024)
SEARCH_TRIALS, SEARCH_WORKERS, SEARCH_ENVS = 12, 3, 4            # [DESIGN] trials, parallel workers and environments per trial
FINAL_STEPS, FINAL_ENVS = 6_000_000, 8                           # [DESIGN] decisions and environments of the final run (8 keeps the search's update size exactly); the search's settings reached their policy within 0.1M
VALIDATION_DAYS = 12                                             # [DESIGN] fixed synthetic validation days for the objective and the monitor
VALIDATION_SEED = 20250401                                       # [DESIGN] seed of the validation days; training episodes draw from other seeds
EVAL_EVERY = 200_000                                             # [DESIGN] decisions between two evaluations in the final run
CHECKPOINT_EVERY = 500_000                                       # [DESIGN] decisions between two saved copies of the model in the final run, so a stopped run keeps its progress
DESIGN_SETTINGS = {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 4096, "n_epochs": 10, "gae_lambda": 0.95, "ent_coef": 0.01,
                   "clip_range": 0.2, "vf_coef": 0.5, "width": 128, "bc_epochs": 20}   # [DESIGN] the agreed PPO settings, tried first in the full-environment search
BC_DAYS = 100                                                    # [DESIGN] simulated days of protocol decisions for the behaviour-cloning warm start (about 11,000 decisions)
BC_EPOCHS, BC_BATCH = 20, 256                                    # [TUNED] passes and batch size of behaviour cloning; held-out agreement 0.995 with 200 days, 0.97 with 5 passes of 1,024
BASELINE_EPISODES, BASELINE_SEED = 100, 700_000                  # [DESIGN] episodes and seed that measure the untrained policy's reward, the start of the reward curve


def sample_settings(trial, small_system=False):
    """Draw one set of PPO settings for a search trial.

    Input
        trial        : the optuna trial (optuna.Trial)
        small_system : True for the small system, which has no warm start and allows longer rollouts (bool)
    Output
        The settings (dict): learning_rate, n_steps (per environment), batch_size, n_epochs, gae_lambda, ent_coef,
        clip_range, vf_coef, width (units in each of the two hidden layers of the actor and of the critic) and, for the
        full environment, bc_epochs (passes of behaviour cloning before PPO; 0 = none, so PPO starts from scratch).
    Example
        sample_settings(trial) -> {'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 2048, ...}
    """
    settings = {"learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 4096] if small_system else [256, 512, 1024]),
                "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
                "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
                "gae_lambda": trial.suggest_categorical("gae_lambda", [0.5, 0.8, 0.9, 0.95]),
                "ent_coef": trial.suggest_float("ent_coef", 1e-4, 2e-2, log=True),
                "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2]),
                "vf_coef": trial.suggest_categorical("vf_coef", [0.5, 1.0]),
                "width": trial.suggest_categorical("width", [64, 128, 256])}
    if not small_system:
        settings["bc_epochs"] = trial.suggest_categorical("bc_epochs", [0, BC_EPOCHS])
    return settings


def build_model(env, settings, seed, log_dir=None):
    """Create a MaskablePPO model with the given settings and the design's discount.

    Input
        env      : the vectorized environment (VecEnv)
        settings : PPO settings (dict from sample_settings)
        seed     : the model's seed (int)
        log_dir  : None, or a folder where the training signals are written as progress.csv (Path)
    Output
        The model (MaskablePPO).
    Example
        model = build_model(env, {"learning_rate": 3e-4, "n_steps": 1024, ...}, seed=0)
    """
    width = settings["width"]
    model = MaskablePPO("MlpPolicy", env, gamma=GAMMA, learning_rate=settings["learning_rate"], n_steps=settings["n_steps"],
                        batch_size=settings["batch_size"], n_epochs=settings["n_epochs"], gae_lambda=settings["gae_lambda"],
                        ent_coef=settings["ent_coef"], clip_range=settings["clip_range"], vf_coef=settings["vf_coef"],
                        policy_kwargs={"net_arch": {"pi": [width, width], "vf": [width, width]}}, seed=seed, verbose=0)
    if log_dir is not None:
        model.set_logger(configure(str(log_dir), ["csv"]))
    return model


# ----------------------------------------------------------------------------------------------------------------------
# The small system
# ----------------------------------------------------------------------------------------------------------------------

def e2_env(fire_weight, seed, shaping=False):
    """E2_ENVS copies of the small system, stepped together in one process, each with its own seed.

    Input
        fire_weight : the weight of a fire call (float); seed : the base seed (int)
        shaping     : whether the readiness shaping term is added to the reward (bool)
    Output
        The vectorized environment (DummyVecEnv of ActionMasker-wrapped SmallDispatchEnv).
    Example
        env = e2_env(8.0, seed=0)
    """
    return DummyVecEnv([lambda i=i: ActionMasker(e2.SmallDispatchEnv(fire_weight, seed=seed * 100 + i, shaping=shaping), lambda e: e.action_masks())
                        for i in range(E2_ENVS)])


def e2_evaluate(model, fire_weight):
    """Evaluate a trained small-system agent exactly: its greedy policy's value against the protocol and the optimum.

    Input
        model       : a MaskablePPO model trained on the small system
        fire_weight : the fire weight it was trained with (float)
    Output
        A dict with:
        - the values of the protocol, the optimum and the agent from an idle system (units of ten minutes);
        - the optimum's improvement in percent of the protocol's cost, and the share of it the agent captured (NaN when
          there is no improvement);
        - the number of states where the agent's rank differs from the optimum's;
        - whether the optimal policy stays the same under an elapsed-time discount with a half-life of
          ELAPSED_HALF_LIFE_H;
        - the decision states where the optimum or the agent leaves the closest engine (DataFrame with columns optimal
          rank, agent rank and the agent's probability of the optimal rank).
    Example
        e2_evaluate(model, 8.0)["agent share of improvement"] -> 0.0 when the agent learned the closest-engine rule only
    """
    optimal_values, optimal_ranks = e2.solve(fire_weight)
    protocol_values, _ = e2.solve(fire_weight, np.zeros((8, e2.N_CALL_TYPES), dtype=int))
    agent_ranks = e2.greedy_ranks(model)
    agent_values, _ = e2.solve(fire_weight, agent_ranks)
    _, clock_ranks = e2.solve(fire_weight, discount_rate=log(2) / ELAPSED_HALF_LIFE_H)
    optimal, protocol, agent = e2.start_value(optimal_values), e2.start_value(protocol_values), e2.start_value(agent_values)
    improvement = optimal - protocol
    labels = [f"{kind} at {'ABMR'[cell]}" for kind, cell, _ in e2.CALL_TYPES]
    observations, masks = zip(*[e2.observe(busy, call_type) for busy in range(e2.FULL) for call_type in range(e2.N_CALL_TYPES)])
    with torch.no_grad():
        probabilities = model.policy.get_distribution(torch.as_tensor(np.array(observations)), action_masks=np.array(masks)).distribution.probs.numpy()
    states = pd.DataFrame({"optimal rank": optimal_ranks[:e2.FULL].ravel(), "agent rank": agent_ranks[:e2.FULL].ravel(),
                           "agent probability of optimal rank": probabilities[np.arange(len(probabilities)), optimal_ranks[:e2.FULL].ravel()].round(3)},
                          index=pd.MultiIndex.from_product([[format(b, "03b")[::-1] for b in range(e2.FULL)], labels], names=["busy A,B,C", "call"]))
    return {"protocol value": protocol, "optimal value": optimal, "agent value": agent,
            "optimum improvement % of protocol cost": 100 * improvement / -protocol,
            "agent share of improvement": (agent - protocol) / improvement if improvement > 1e-9 else np.nan,
            "states where agent differs from optimum": int((agent_ranks[:e2.FULL] != optimal_ranks[:e2.FULL]).sum()),
            "optimal policy same under elapsed-time discount": bool((clock_ranks[:e2.FULL] == optimal_ranks[:e2.FULL]).all()),
            "states": states[(states[["optimal rank", "agent rank"]] != 0).any(axis=1)]}


def e2_objective(trial):
    """One ASHA trial on the small system: train at fire weight 8, report the exact share at each rung, stop if pruned.

    Input
        trial : the optuna trial (optuna.Trial)
    Output
        The share of the improvement captured at the last rung (float).
    Example
        study.optimize(e2_objective, n_trials=3)
    """
    torch.set_num_threads(1)
    model = build_model(e2_env(8.0, trial.number), sample_settings(trial, small_system=True), seed=trial.number)
    share = 0.0
    for rung in E2_RUNGS:
        model.learn(total_timesteps=rung - model.num_timesteps, reset_num_timesteps=False)
        share = e2_evaluate(model, 8.0)["agent share of improvement"]
        trial.report(share, model.num_timesteps)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return share


def e2_gate_weight(fire_weight, settings, train_steps, seed, shaping=False):
    """Train one small-system agent at one fire weight and evaluate it exactly (one row of the gate).

    Input
        fire_weight : the weight of a fire call (float); settings : PPO settings (dict)
        train_steps : training decisions (int); seed : the training seed (int)
        shaping     : whether the agent trains with the readiness shaping term (bool)
    Output
        The dict from e2_evaluate.
    Example
        e2_gate_weight(32.0, settings, 2_654_208, seed=101)["agent share of improvement"] -> at least 0.8 to pass
    """
    torch.set_num_threads(1)
    model = build_model(e2_env(fire_weight, seed, shaping), settings, seed=seed)
    model.learn(total_timesteps=train_steps)
    return e2_evaluate(model, fire_weight)


def e2_gate(settings, train_steps=E2_RUNGS[-1], seeds=E2_GATE_SEEDS, shaping=False):
    """Run the learnability gate: train at every fire weight with fresh seeds and compare with the exact optimum.

    The gate can train with or without the readiness shaping term; the agents are always evaluated without it.

    Input
        settings    : PPO settings, frozen after the search (dict); train_steps : training decisions per agent (int)
        seeds       : training seeds the search never used; one agent is trained per fire weight and seed (list of int)
        shaping     : whether the agents train with the readiness shaping term (bool)
    Output
        One row per fire weight and seed (DataFrame): the values, the optimum's improvement, the agent's share, the
        states where the agent differs from the optimum, the elapsed-time check, and whether the agent passes (its share
        is at least e2.GATE_SHARE; None at a fire weight without improvement). The gate passes when every agent at the
        fire weights in e2.GATE_WEIGHTS passes. The decision states where the optimum or the agent leaves the closest
        engine are printed for each agent.
    Example
        e2_gate(settings)["passes"] -> None at fire weight 2 (no improvement), True at 8 and 32 when it passes
    """
    jobs = [(fire_weight, seed) for fire_weight in e2.FIRE_WEIGHTS for seed in seeds]
    with ProcessPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(e2_gate_weight, *zip(*[(w, settings, train_steps, s, shaping) for w, s in jobs])))
    for (fire_weight, seed), result in zip(jobs, results):
        print(f"--- fire weight {fire_weight}, seed {seed}: decision states where the optimum or the agent leaves the closest engine\n"
              f"{result.pop('states').to_string()}\n")
        result["passes"] = bool(result["agent share of improvement"] >= e2.GATE_SHARE) if fire_weight in e2.GATE_WEIGHTS else None
    gate = pd.DataFrame(results, index=pd.MultiIndex.from_tuples(jobs, names=["fire weight", "seed"]))
    passed = bool(gate["passes"].dropna().all())
    print(f"--- learnability gate ({train_steps:,} decisions per agent; shaping {'on' if shaping else 'off'}; values from an idle "
          f"system in units of ten minutes; gate share {e2.GATE_SHARE})\n{gate.to_string()}\ngate passes: {passed}\n")
    return gate


# ----------------------------------------------------------------------------------------------------------------------
# The full environment
# ----------------------------------------------------------------------------------------------------------------------

def make_dispatch_env(seed, shaping=True):
    """A function that builds one dispatch environment; the world is built inside the process that calls it.

    Input
        seed    : the seed of the environment's episode draws (int)
        shaping : whether the term that moves each run's estimated cost to its dispatch is added (bool)
    Output
        A function with no argument that returns an ActionMasker-wrapped DispatchEnv.
    Example
        SubprocVecEnv([make_dispatch_env(seed) for seed in range(4)])
    """
    def build():
        import environment, rl_env, scenarios
        torch.set_num_threads(1)
        world = environment.build_world(verbose=False)
        generator = scenarios.fit_generator(world, verbose=False)
        env = rl_env.DispatchEnv(world, generator, rl_env.fit_rl_inputs(world, verbose=False), seed=seed, shaping=shaping)
        return ActionMasker(env, lambda e: e.action_masks())
    return build


def dispatch_vec_env(n_envs, base_seed, shaping=True):
    """n_envs dispatch environments in separate processes, with episode returns and lengths recorded.

    Input
        n_envs    : how many environments (int)
        base_seed : the first environment's seed; the others follow (int). A model built on this environment reseeds
                    environment i with its own seed + i at the first reset, so callers give the model the same seed.
        shaping   : whether the term that moves each run's estimated cost to its dispatch is added (bool)
    Output
        The vectorized environment (VecMonitor around SubprocVecEnv).
    Example
        env = dispatch_vec_env(4, base_seed=1000)
    """
    return VecMonitor(SubprocVecEnv([make_dispatch_env(base_seed + i, shaping) for i in range(n_envs)], start_method="spawn"))


class Evaluator:
    """Fixed validation days, and the weighted response totals of the protocol, the closest engine and the agent on them.

    The days are drawn like training episodes (a training day, a condition, a fresh seed) from a seed that training
    never uses, so the comparison measures how well the agent learned the training distribution; no April to September
    call enters it. All policies play the same calls with the same travel draws. The protocol is simulated directly,
    so its choice counts even when its engine is not among the agent's five candidates. The closest engine without the
    protocol's quint rule (always the first candidate) is scored too. Because the reward does not value engine type,
    dropping the quint rule alone scores above the protocol; an agent has learned something beyond the closest engine
    only when it beats that score.

    Input (constructor)
        days : how many validation days (int); seed : the seed that draws them (int)
    Output
        An object whose compare(model) returns a DataFrame with one row per validation day: condition, day, protocol
        return, closest return (the first candidate every time), agent return (unshaped, every arrival of the day's calls
        counted to the end of the scenario), improvement (agent minus protocol, in units of ten weighted minutes;
        positive = the agent is better), improvement over closest (agent minus closest), the agent's share of
        decisions that differ from the protocol's, and its closest agreement (the share of decisions where it takes
        the first candidate).
    Example
        evaluator = Evaluator(12, VALIDATION_SEED); evaluator.compare(model)["improvement"].mean() -> 0.13 for an agent
        that always takes the closest engine
    """

    def __init__(self, days=VALIDATION_DAYS, seed=VALIDATION_SEED):
        import environment, policies, rl_env, scenarios
        world = environment.build_world(verbose=False)
        generator = scenarios.fit_generator(world, verbose=False)
        rl = rl_env.fit_rl_inputs(world, verbose=False)
        self.env = rl_env.DispatchEnv(world, generator, rl, seed=seed)
        rng = np.random.default_rng(seed)
        self.scenarios = [rl_env.sample_scenario(world, generator, rng) for _ in range(days)]
        self.protocol = []
        for scenario in self.scenarios:
            prepared = rl_env.prepare_scenario(world, rl, scenario)
            simulated = environment.simulate(world, scenario, policies.closest_engine, verbose=False)
            self.protocol.append(rl_env.accrued_reward(prepared, simulated, rl_env.new_tally(prepared)))
        self.closest = [self.play(scenario, None)[0] for scenario in self.scenarios]

    def play(self, scenario, model):
        """Run one validation day to its end with the model (None = always the first candidate); return (unshaped return,
        share of decisions unlike the protocol's, share of decisions that take the first candidate)."""
        observation, _ = self.env.reset(options={"scenario": scenario, "full_day": True})
        first = 0
        while True:
            action = 0 if model is None else int(model.predict(observation, action_masks=self.env.action_masks(), deterministic=True)[0])
            first += action == 0
            observation, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                decisions = max(info["decisions"], 1)
                return info["unshaped_return"], info["deviations"] / decisions, first / decisions

    def compare(self, model):
        rows = []
        for scenario, protocol_return, closest_return in zip(self.scenarios, self.protocol, self.closest):
            agent_return, deviation_share, closest_agreement = self.play(scenario, model)
            rows.append({"condition": scenario["condition"], "day": scenario["day"], "protocol return": protocol_return,
                         "closest return": closest_return, "agent return": agent_return, "improvement": agent_return - protocol_return,
                         "improvement over closest": agent_return - closest_return, "deviation share": deviation_share,
                         "closest agreement": closest_agreement})
        return pd.DataFrame(rows)


def collect_protocol_decisions(days, seed):
    """Collect the observation, the mask and the protocol's slot of every decision of simulated days played by the protocol.

    A decision where the protocol's engine is not among the five candidates cannot be imitated. Such a decision is left
    out and the closest engine is sent instead; the number left out is printed.

    Input
        days : how many simulated days (int); seed : the seed of their draws (int)
    Output
        Three arrays: observations (decisions x 207), masks (decisions x 5) and the protocol's slots (decisions).
    Example
        observations, masks, slots = collect_protocol_decisions(100, seed=7); (slots == 0).mean() -> about 0.94
    """
    env = make_dispatch_env(seed)().env
    observations, masks, slots, left_out = [], [], [], 0
    for _ in range(days):
        observation, _ = env.reset()
        while True:
            slot = env.protocol_slot
            if slot is None:
                left_out += 1
            else:
                observations.append(observation)
                masks.append(env.action_masks())
                slots.append(slot)
            observation, _, terminated, truncated, _ = env.step(0 if slot is None else slot)
            if terminated or truncated:
                break
    env.close()
    print(f"protocol decisions for behaviour cloning: {len(slots):,} kept from {days} days; {left_out} left out because the "
          f"protocol's engine is not among the five candidates; the protocol leaves the closest engine in {np.mean(np.array(slots) != 0):.3f}")
    return np.array(observations), np.array(masks), np.array(slots)


def behaviour_clone(model, observations, masks, slots, epochs=BC_EPOCHS, batch_size=BC_BATCH, learning_rate=1e-3):
    """Fit the actor to the protocol's choices by maximum likelihood (the warm start); the critic is not changed.

    Input
        model : the MaskablePPO model; observations, masks, slots : the arrays from collect_protocol_decisions
        epochs : passes over the data (int); batch_size : decisions per gradient step (int); learning_rate (float)
    Output
        The share of decisions where the cloned actor's greedy slot equals the protocol's (float).
    Example
        behaviour_clone(model, observations, masks, slots) -> about 0.998
    """
    policy = model.policy
    actor_parameters = [p for name, p in policy.named_parameters() if not name.startswith(("value_net", "mlp_extractor.value_net"))]
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    obs, acts = torch.as_tensor(observations, dtype=torch.float32), torch.as_tensor(slots, dtype=torch.long)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(obs)), max(1, len(obs) // batch_size)):
            loss = -policy.get_distribution(obs[batch], action_masks=masks[batch]).log_prob(acts[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        greedy = policy.get_distribution(obs, action_masks=masks).get_actions(deterministic=True).numpy()
    return float((greedy == slots).mean())


class SignalCallback(BaseCallback):
    """Record the unshaped return of finished training episodes and, every eval_every decisions, the validation result.

    Input (constructor)
        evaluator  : an Evaluator, or None to skip validation
        eval_every : decisions between two evaluations (int)
    Output
        A stable-baselines3 callback that adds rollout/unshaped_return_mean and rollout/deviation_share (means over the
        last 100 episodes) and eval/improvement_mean, eval/improvement_over_closest_mean, eval/deviation_share and
        eval/closest_agreement to the training signals.
    Example
        model.learn(1_000_000, callback=SignalCallback(evaluator, 200_000))
    """

    def __init__(self, evaluator, eval_every):
        super().__init__()
        self.evaluator, self.eval_every, self.next_eval = evaluator, eval_every, 0
        self.unshaped, self.deviation = [], []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "unshaped_return" in info:
                self.unshaped.append(info["unshaped_return"])
                self.deviation.append(info["deviations"] / max(info["decisions"], 1))
        return True

    def _on_rollout_end(self):
        if self.unshaped:
            self.logger.record("rollout/unshaped_return_mean", float(np.mean(self.unshaped[-100:])))
            self.logger.record("rollout/deviation_share", float(np.mean(self.deviation[-100:])))
        if self.evaluator is not None and self.num_timesteps >= self.next_eval:
            comparison = self.evaluator.compare(self.model)
            self.logger.record("eval/improvement_mean", float(comparison["improvement"].mean()))
            self.logger.record("eval/improvement_over_closest_mean", float(comparison["improvement over closest"].mean()))
            self.logger.record("eval/deviation_share", float(comparison["deviation share"].mean()))
            self.logger.record("eval/closest_agreement", float(comparison["closest agreement"].mean()))
            self.next_eval = self.num_timesteps + self.eval_every


def dispatch_objective(trial):
    """One ASHA trial on the full environment: train, report the validation improvement at each rung, stop if pruned.

    The trial's training signals (one row per PPO update, with the validation improvement at each rung) are written to
    progress.csv in its own folder next to the models, so that every trial can be inspected while the search runs.

    Input
        trial : the optuna trial (optuna.Trial)
    Output
        The mean improvement over the protocol per validation day at the last rung (float, units of ten weighted minutes).
    Example
        study.optimize(dispatch_objective, n_trials=4)
    """
    torch.set_num_threads(1)
    settings = sample_settings(trial)
    seed = 1000 * (trial.number + 1)                                # keeps the trials' episode streams apart
    env = dispatch_vec_env(SEARCH_ENVS, base_seed=seed)
    try:
        model = build_model(env, settings, seed=seed, log_dir=MODEL_DIR / "dispatch_search_trials" / f"trial_{trial.number}")
        if settings["bc_epochs"]:
            agreement = behaviour_clone(model, *collect_protocol_decisions(BC_DAYS, seed=trial.number), epochs=settings["bc_epochs"])
            model.logger.record("bc/agreement", agreement)
        evaluator, improvement, callback = Evaluator(), -np.inf, SignalCallback(None, 0)
        for rung in SEARCH_RUNGS:
            model.learn(total_timesteps=rung - model.num_timesteps, reset_num_timesteps=False, callback=callback)
            improvement = float(evaluator.compare(model)["improvement"].mean())
            model.logger.record("eval/improvement_mean", improvement)
            model.logger.record("time/total_timesteps", model.num_timesteps)
            model.logger.dump(model.num_timesteps)                   # written now, before the trial may be stopped
            trial.report(improvement, model.num_timesteps)
            if trial.should_prune():
                raise optuna.TrialPruned()
    finally:
        env.close()
    return improvement


def open_study(name, min_resource, worker=0):
    """Open (or create) a shared optuna study with the ASHA pruner, stored next to the models in a journal file.

    The journal file locks each write, so parallel workers never take the same queued trial (SQLite storage can hand
    one queued trial to two workers at once).

    Input
        name : the study's name, also its file name (str); min_resource : the first rung in decisions (int)
        worker : the worker's index, which seeds its sampler so that parallel workers draw different settings (int)
    Output
        The optuna study (maximized).
    Example
        open_study("e2_search", E2_RUNGS[1], worker=3)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(MODEL_DIR / f"{name}.journal")))
    return optuna.create_study(study_name=name, storage=storage, direction="maximize", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=worker, n_startup_trials=8, constant_liar=True),
                               pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=min_resource, reduction_factor=REDUCTION_FACTOR))


def run_search(name, worker_command, min_resource, workers, trials, first_settings=()):
    """Run an ASHA search with parallel worker processes of this script, then write and print its trials.

    Input
        name : the study's name (str); worker_command : "e2-worker" or "worker" (str)
        min_resource : the first rung at which trials can be stopped, in decisions (int)
        workers : parallel processes (int); trials : trials in total (int)
        first_settings : settings to try before the sampler's own draws, when the study is new (list of dicts)
    Output
        The best settings (dict), also written as <name>_best_settings.json; every trial is written as <name>_trials.csv
        and each worker's output as <worker_command>_<i>.log, all next to the models.
    Example
        run_search("e2_search", "e2-worker", E2_RUNGS[1], 9, 27)
    """
    study = open_study(name, min_resource)
    if not study.trials:
        for settings in first_settings:
            study.enqueue_trial(settings)
    per_worker = int(np.ceil(trials / workers))
    processes = [subprocess.Popen([sys.executable, __file__, worker_command, str(per_worker), str(i)],
                                  stdout=open(MODEL_DIR / f"{worker_command}_{i}.log", "w"), stderr=subprocess.STDOUT) for i in range(workers)]
    for process in processes:
        process.wait()
    study = open_study(name, min_resource)
    trials = study.trials_dataframe()
    trials.to_csv(MODEL_DIR / f"{name}_trials.csv", index=False)
    columns = ["number", "value", "state", "duration"] + [c for c in trials.columns if c.startswith("params_")]
    print(f"{name}: best value {study.best_value:.4f} (trial {study.best_trial.number}) with {study.best_params}\n"
          f"{trials[columns].sort_values('value', ascending=False).to_string()}\n")
    (MODEL_DIR / f"{name}_best_settings.json").write_text(json.dumps(study.best_params, indent=2))
    return study.best_params


def late_change(values):
    """Measure how much a series' straight-line trend changes across it; used to judge whether a signal has levelled off.

    Input
        values : the series, in time order (sequence of floats)
    Output
        The fitted slope x the number of points (float, in the series' units).
    Example
        late_change([1.0, 1.1, 1.2, 1.3]) -> 0.4
    """
    return float(np.polyfit(np.arange(len(values)), np.asarray(values, dtype=float), 1)[0] * len(values))


def initial_episode_reward(model, world_inputs, episodes=BASELINE_EPISODES, seed=BASELINE_SEED):
    """Measure the training reward of the policy before any optimization: the starting level of the reward curve.

    The policy samples its actions as it does in training, on training-style episodes (a training day, a condition, the
    six-hour warm-up, the end at the first decision after midnight) with the same shaped reward, drawn from a seed that
    training never uses.

    Input
        model        : the MaskablePPO model before training
        world_inputs : (world, generator, rl), the inputs of DispatchEnv (tuple of dicts)
        episodes     : completed episodes to average (int); seed : the seed of their draws (int)
    Output
        The mean episode reward (float, reward units), comparable with the training signal rollout/ep_rew_mean.
    Example
        initial_episode_reward(model, (world, generator, rl)) -> about -147 for a freshly initialized policy
    """
    import rl_env
    env = rl_env.DispatchEnv(*world_inputs, seed=seed)
    totals = []
    for _ in range(episodes):
        observation, _ = env.reset()
        total = 0.0
        while True:
            action = int(model.predict(observation, action_masks=env.action_masks(), deterministic=False)[0])
            observation, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        totals.append(total)
    env.close()
    return float(np.mean(totals))


def acceptance(progress, initial_reward, tail=0.3):
    """Check a training run's signals against the acceptance criteria.

    Training is accepted when the reward rises from the untrained policy's level and levels off, both on the training
    episodes and on the fixed validation days, and when the critic, the size of the policy updates and the policy
    entropy behave healthily. The health checks use no fixed targets that ignore how a run starts: a fixed explained
    variance, a minimum KL or a falling entropy would reject a run that starts from the cloned, nearly deterministic
    protocol.

    Input
        progress       : the training signals (DataFrame read from progress.csv: one row per PPO update, and the
                         validation improvement on the rows of an evaluation, including one of the final model)
        initial_reward : the training reward of the policy before any optimization (float, from initial_episode_reward)
        tail           : the share of the run that counts as "late" (float)
    Output
        One row per criterion (DataFrame with columns value, threshold, passes):
        - training reward rises: D = mean episode reward over the late updates minus initial_reward (> 0);
        - training reward converged: the change of its straight-line trend across the late updates, as a share of D (< 0.1);
        - training reward plateau: the range of its means over three consecutive equal parts of the late updates, as a
          share of D (< 0.1);
        - validation improvement rises: mean of the last three evaluations minus the evaluation before training (> 0);
        - validation improvement converged: the change of its trend across the late evaluations, as a share of its rise
          (< 0.1);
        - explained variance: positive over the late updates (> 0) and not falling (late mean minus the mean of the middle
          third > -0.05);
        - approx KL: the largest value over the second half of the updates (<= 0.05: no runaway update);
        - entropy: the late entropy is below half of the uniform policy's (< ln 5 / 2 nats: the policy holds clear
          preferences) and does not drift upward (change of its late trend < 0.05 nats).
    Example
        acceptance(pd.read_csv("progress.csv"), -147.3)["passes"].all() -> True for an accepted model
    """
    rows = progress.dropna(subset=["train/explained_variance"]).reset_index(drop=True)
    n = len(rows)
    late = max(3, int(n * tail))
    reward = rows["rollout/ep_rew_mean"].ffill()
    rise = reward.iloc[-late:].mean() - initial_reward
    reward_change = late_change(reward.iloc[-late:])
    block_means = [part.mean() for part in np.array_split(reward.iloc[-late:].to_numpy(), 3)]
    block_range = max(block_means) - min(block_means)
    evaluations = progress.dropna(subset=["eval/improvement_mean"])["eval/improvement_mean"].reset_index(drop=True)
    late_evaluations = evaluations.iloc[-max(3, int(len(evaluations) * tail)):]
    eval_rise = evaluations.iloc[-3:].mean() - evaluations.iloc[0]
    eval_change = late_change(late_evaluations)
    variance = rows["train/explained_variance"]
    variance_late, variance_middle = variance.iloc[-late:].mean(), variance.iloc[n // 3: 2 * n // 3].mean()
    kl_max = rows["train/approx_kl"].iloc[n // 2:].max()
    entropy = -rows["train/entropy_loss"]
    entropy_late, entropy_change = entropy.iloc[-late:].mean(), late_change(entropy.iloc[-late:])
    checks = {"training reward rises (late minus untrained)": (rise, "> 0", rise > 0),
              "training reward converged (late trend change / rise)": (abs(reward_change) / rise if rise > 0 else np.inf, "< 0.1",
                                                                       rise > 0 and abs(reward_change) < 0.1 * rise),
              "training reward plateau (late block range / rise)": (block_range / rise if rise > 0 else np.inf, "< 0.1",
                                                                    rise > 0 and block_range < 0.1 * rise),
              "validation improvement rises (last three minus first)": (eval_rise, "> 0", eval_rise > 0),
              "validation improvement converged (late trend change / rise)": (abs(eval_change) / eval_rise if eval_rise > 0 else np.inf, "< 0.1",
                                                                              eval_rise > 0 and abs(eval_change) < 0.1 * eval_rise),
              "explained variance (late mean)": (variance_late, "> 0", variance_late > 0),
              "explained variance not falling (late minus middle)": (variance_late - variance_middle, "> -0.05", variance_late - variance_middle > -0.05),
              "approx KL (largest, second half)": (kl_max, "<= 0.05", kl_max <= 0.05),
              "entropy late (nats)": (entropy_late, f"< {np.log(5) / 2:.3f}", entropy_late < np.log(5) / 2),
              "entropy late trend change (nats)": (entropy_change, "< 0.05", entropy_change < 0.05)}
    return pd.DataFrame(checks, index=["value", "threshold", "passes"]).T


def final_run(settings, steps=FINAL_STEPS, n_envs=FINAL_ENVS, name="final", seed=500_000):
    """Train the final model with the chosen settings, validate it every EVAL_EVERY decisions and check acceptance.

    The search ran SEARCH_ENVS environments; with n_envs here, the steps per environment are scaled so that each PPO
    update sees the same number of decisions and the same batches as in the search.

    Input
        settings : PPO settings from the search (dict); steps : training decisions (int)
        n_envs   : environments (int; SEARCH_ENVS x n_steps must divide by it)
        name     : the run's folder name next to the models (str)
        seed     : the seed of the model and of the training episodes, apart from the search's (int)
    Output
        The acceptance table (DataFrame). The model, progress.csv, the settings, the untrained policy's reward, the
        acceptance table and the final validation days are written to the run's folder, and a copy of the model every
        CHECKPOINT_EVERY decisions to its checkpoints folder.
    Example
        final_run(open_study("dispatch_search", SEARCH_RUNGS[0]).trials[3].params)
    """
    torch.set_num_threads(2)
    assert settings["n_steps"] * SEARCH_ENVS % n_envs == 0, "the update size of the search cannot be kept"
    settings = {**settings, "n_steps": settings["n_steps"] * SEARCH_ENVS // n_envs}
    run_dir = MODEL_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    env = dispatch_vec_env(n_envs, base_seed=seed)
    model = build_model(env, settings, seed=seed, log_dir=run_dir)
    if settings.get("bc_epochs"):
        agreement = behaviour_clone(model, *collect_protocol_decisions(BC_DAYS, seed=11), epochs=settings["bc_epochs"])
        print(f"behaviour cloning: greedy agreement with the protocol {agreement:.3f}")
    evaluator = Evaluator()
    initial_reward = initial_episode_reward(model, (evaluator.env.world, evaluator.env.generator, evaluator.env.rl))
    (run_dir / "initial_reward.json").write_text(json.dumps({"initial_episode_reward": initial_reward, "episodes": BASELINE_EPISODES,
                                                             "seed": BASELINE_SEED}, indent=2))
    print(f"untrained policy: mean episode reward {initial_reward:.3f} over {BASELINE_EPISODES} episodes")
    start = time.time()
    starting = evaluator.compare(model)                                    # the starting policy
    model.logger.record("eval/improvement_mean", float(starting["improvement"].mean()))
    model.logger.record("eval/improvement_over_closest_mean", float(starting["improvement over closest"].mean()))
    model.logger.record("eval/closest_agreement", float(starting["closest agreement"].mean()))
    model.logger.record("time/total_timesteps", 0)
    model.logger.dump(0)
    checkpoints = CheckpointCallback(save_freq=CHECKPOINT_EVERY // n_envs, save_path=str(run_dir / "checkpoints"), name_prefix="model")
    model.learn(total_timesteps=steps, callback=CallbackList([SignalCallback(evaluator, EVAL_EVERY), checkpoints]))
    model.save(run_dir / "model")
    env.close()
    final = evaluator.compare(model)                                       # the final model's evaluation enters acceptance
    final.to_csv(run_dir / "validation_days.csv", index=False)
    model.logger.record("eval/improvement_mean", float(final["improvement"].mean()))
    model.logger.record("eval/improvement_over_closest_mean", float(final["improvement over closest"].mean()))
    model.logger.record("eval/closest_agreement", float(final["closest agreement"].mean()))
    model.logger.record("time/total_timesteps", model.num_timesteps)
    model.logger.dump(model.num_timesteps)
    checks = acceptance(pd.read_csv(run_dir / "progress.csv"), initial_reward)
    checks.to_csv(run_dir / "acceptance.csv")
    print(f"final run: {model.num_timesteps:,} decisions in {(time.time() - start) / 3600:.1f} h\n{checks.to_string()}\n"
          f"validation improvement per day: mean {final['improvement'].mean():.3f} over the protocol, "
          f"{final['improvement over closest'].mean():.3f} over the closest engine; agreement with the closest engine "
          f"{final['closest agreement'].mean():.4f}\n{final.round(3).to_string()}\n")
    return checks


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "e2-worker":
        open_study("e2_search", E2_RUNGS[1], worker=int(sys.argv[3])).optimize(e2_objective, n_trials=int(sys.argv[2]))
    elif command == "worker":
        open_study("dispatch_search", SEARCH_RUNGS[0], worker=int(sys.argv[3])).optimize(dispatch_objective, n_trials=int(sys.argv[2]))
    elif command == "e2-search":
        run_search("e2_search", "e2-worker", E2_RUNGS[1], E2_WORKERS, E2_TRIALS)
    elif command == "e2-gate":                                      # add "shaping" to train with the readiness shaping term
        e2_gate(json.loads((MODEL_DIR / "e2_search_best_settings.json").read_text()), shaping=sys.argv[2:] == ["shaping"])
    elif command == "search":
        e2_best = json.loads((MODEL_DIR / "e2_search_best_settings.json").read_text())
        run_search("dispatch_search", "worker", SEARCH_RUNGS[0], SEARCH_WORKERS, SEARCH_TRIALS,
                   [DESIGN_SETTINGS, {**DESIGN_SETTINGS, "bc_epochs": 0}, {**e2_best, "n_steps": 512, "bc_epochs": BC_EPOCHS}])
    elif command == "final":                                        # "final N [seed name]": trial N's settings instead of the best
        settings = (open_study("dispatch_search", SEARCH_RUNGS[0]).trials[int(sys.argv[2])].params if len(sys.argv) > 2
                    else json.loads((MODEL_DIR / "dispatch_search_best_settings.json").read_text()))
        final_run(settings, **({"seed": int(sys.argv[3]), "name": sys.argv[4]} if len(sys.argv) > 4 else {}))
