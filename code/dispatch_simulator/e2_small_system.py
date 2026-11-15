"""The learnability gate: a small dispatch system whose optimal policy is known exactly.

Before the learned dispatcher trains on the full simulator, it must show that it can find a good policy where the best
policy is known. The small system has three stations with one engine each and four places where calls come from. Value
iteration gives the exact optimal policy. The same kind of agent as in the full design (a ranked action over the free
engines, MaskablePPO, the same discount per decision) then trains on a simulation of the same system, and its greedy
policy is evaluated exactly with the same equations. The gate passes when the agent captures at least
GATE_SHARE of the improvement from the closest-engine rule to the optimum, at the fire weights where an improvement
exists.

Running this module prints the exact problem: the protocol's and the optimum's values at each fire weight.
"""
from itertools import combinations
from math import comb

import gymnasium as gym
import numpy as np
import pandas as pd

TRAVEL_MIN = np.array([[1.0, 5.0, 7.0, 14.0],               # [DESIGN] modeled travel from station A to cells A, B, M, R (minutes)
                       [5.0, 1.0, 3.0, 10.0],               # [DESIGN] from station B
                       [9.0, 4.0, 2.5, 5.0]])               # [DESIGN] from station C; a call at M is 2.5 min from C, 3.0 from B
CALL_TYPES = [("medical", 0, 0.5), ("medical", 1, 0.5), ("medical", 2, 0.4), ("fire", 3, 0.1)]  # [DESIGN] (class, cell, calls per hour)
SERVICE_MEAN_MIN = 45.0                                     # [DESIGN] an engine is busy from dispatch until available again, exponential with this mean
OUTSIDE_HELP_MIN = 20.0                                     # [DESIGN] a call that finds no free engine gets outside help after this many minutes
GAMMA = 0.97                                                # [DESIGN] discount per decision, as in the full design
FIRE_WEIGHTS = [2.0, 8.0, 32.0]                             # [DESIGN] fire weights tried; medical calls weigh 1
GATE_SHARE = 0.80                                           # [DESIGN] share of the protocol-to-optimum improvement the agent must capture
GATE_WEIGHTS = [8.0, 32.0]                                  # [DESIGN] the fire weights where the optimum beats the closest-engine rule
EPISODE_DECISIONS = 200                                     # [DESIGN] decisions per training episode (a truncation; the value is bootstrapped)
N_ENGINES, N_CALL_TYPES, FULL = 3, len(CALL_TYPES), 7       # [DESIGN] three engines; busy sets are bit masks, 7 = all busy
RATES = np.array([rate for _, _, rate in CALL_TYPES])       # calls per hour
CALL_PROBABILITY = RATES / RATES.sum()
SERVICE_RATE = 60.0 / SERVICE_MEAN_MIN                      # completions per hour


def call_weights(fire_weight):
    """Return the reward weight of each call type: 1 for medical calls and fire_weight for the fire call type.

    Input
        fire_weight : the weight of a fire call (float), e.g. 8.0
    Output
        One weight per call type, in the order of CALL_TYPES (array of 4 floats).
    Example
        call_weights(8.0) -> array([1., 1., 1., 8.])
    """
    return np.array([fire_weight if kind == "fire" else 1.0 for kind, _, _ in CALL_TYPES])


def free_engines_ranked(busy, call_type):
    """List the free engines, closest first, for a call of the given type: the candidates of the ranked action.

    Input
        busy      : the busy engines as a bit mask (int, 0-7; bit i set means engine i is busy), e.g. 4 (engine C busy)
        call_type : index into CALL_TYPES (int), e.g. 2 (a medical call at M)
    Output
        The free engine indices sorted by travel to the call's cell (list of int).
    Example
        free_engines_ranked(0, 2) -> [2, 1, 0]      # at M: C 2.5 min, B 3.0 min, A 7.0 min
    """
    cell = CALL_TYPES[call_type][1]
    return sorted((engine for engine in range(N_ENGINES) if not busy >> engine & 1), key=lambda engine: TRAVEL_MIN[engine, cell])


def transition_weights(discount_rate=0.0):
    """Compute how the busy set changes between two call arrivals, optionally discounted by the time it takes.

    Between arrivals, which come at rate RATES.sum(), each busy engine finishes at rate SERVICE_RATE. For a busy set b
    after a dispatch, the weight of busy set b2 (a subset of b) at the next arrival is E[exp(-discount_rate T) x the
    chance that exactly the engines of b2 are still busy after T], with T exponential.

    Input
        discount_rate : the discount per hour of an elapsed-time clock (float); 0 gives plain probabilities
    Output
        An 8 x 8 array: entry [b, b2] is the weight of reaching busy set b2 at the next arrival from busy set b.
    Example
        transition_weights()[4, 0] -> chance that engine C finishes before the next call, 1.333 / (1.5 + 1.333) = 0.471
    """
    arrival_rate, weights = RATES.sum(), np.zeros((8, 8))
    for busy in range(8):
        members = [engine for engine in range(N_ENGINES) if busy >> engine & 1]
        for size in range(len(members) + 1):
            for still_busy in combinations(members, size):
                finished = len(members) - size
                weights[busy, sum(1 << engine for engine in still_busy)] = sum(
                    comb(finished, j) * (-1) ** j * arrival_rate / (arrival_rate + discount_rate + (size + j) * SERVICE_RATE)
                    for j in range(finished + 1))
    return weights


def solve(fire_weight, policy=None, discount_rate=None, tolerance=1e-12):
    """Compute the value of every decision state, for the optimal policy or for a given policy.

    A decision state is (busy set, call type) with at least one free engine. Sending free engine e to call type k costs
    w_k x travel / 10 at once. Until the next decision, every call that finds all engines busy costs w x
    OUTSIDE_HELP_MIN / 10 in the same step. The next decision is discounted by GAMMA (per decision), or, when
    discount_rate is given, by exp(-discount_rate x elapsed hours) instead.

    Input
        fire_weight   : the weight of a fire call (float)
        policy        : None for the optimal policy, or the rank chosen in each state (int array, 8 x 4; entries for the
                        all-busy set are ignored)
        discount_rate : None for the per-decision discount, or a discount per hour (float)
        tolerance     : the largest change in value at which the iteration stops (float)
    Output
        Two arrays of shape 8 x 4: the value of each decision state (negative cost, in units of ten minutes) and the
        rank the policy takes there (the optimal one when policy is None).
    Example
        values, ranks = solve(32.0)
        ranks[0, 2] -> 1        # with all engines free, the optimum sends B (second closest) to a call at M
    """
    weights_k = call_weights(fire_weight)
    penalty = float(CALL_PROBABILITY @ weights_k) * OUTSIDE_HELP_MIN / 10
    plain = discount_rate is None
    transitions = transition_weights(0.0 if plain else discount_rate)
    step_discount = GAMMA if plain else 1.0
    values, ranks = np.zeros((8, N_CALL_TYPES)), np.zeros((8, N_CALL_TYPES), dtype=int)
    while True:
        next_value = values @ CALL_PROBABILITY * step_discount           # value at the next decision, for each busy set
        after = np.zeros(8)                                              # value right after a dispatch leaves busy set b
        stay = transitions[FULL, FULL]
        after[FULL] = (stay * -penalty + transitions[FULL, :FULL] @ next_value[:FULL]) / (1 - stay)
        for busy in range(FULL):
            after[busy] = transitions[busy, :FULL] @ next_value[:FULL]
        new = np.zeros_like(values)
        for busy in range(FULL):
            for call_type in range(N_CALL_TYPES):
                cell = CALL_TYPES[call_type][1]
                options = [-weights_k[call_type] * TRAVEL_MIN[engine, cell] / 10 + after[busy | 1 << engine]
                           for engine in free_engines_ranked(busy, call_type)]
                rank = int(np.argmax(options)) if policy is None else min(int(policy[busy, call_type]), len(options) - 1)
                new[busy, call_type], ranks[busy, call_type] = options[rank], rank
        change = np.abs(new - values).max()
        values = new
        if change < tolerance:
            return values, ranks


def potential(busy, fire_weight):
    """The readiness potential of a decision state: minus the weighted travel from the nearest free engines.

    R_F is the travel from the nearest free engine to the fire cell R, and R_N the call-rate-weighted travel from the
    nearest free engine to the medical cells; a cell with no free engine counts OUTSIDE_HELP_MIN, the finite cap.

    Input
        busy        : the busy engines as a bit mask (int, 0-7)
        fire_weight : the weight of a fire call (float); medical calls weigh 1
    Output
        Phi = -(fire_weight x R_F + R_N) / 10 (float, units of ten weighted minutes).
    Example
        potential(0, 8.0) -> -(8 x 5.0 + (0.5 x 1 + 0.5 x 1 + 0.4 x 2.5) / 1.4) / 10 = -4.143
    """
    free = [engine for engine in range(N_ENGINES) if not busy >> engine & 1]
    nearest = TRAVEL_MIN[free].min(axis=0) if free else np.full(TRAVEL_MIN.shape[1], OUTSIDE_HELP_MIN)
    medical = [(cell, rate) for kind, cell, rate in CALL_TYPES if kind == "medical"]
    readiness_N = sum(rate * nearest[cell] for cell, rate in medical) / sum(rate for _, rate in medical)
    readiness_F = nearest[[cell for kind, cell, _ in CALL_TYPES if kind == "fire"][0]]
    return -(fire_weight * readiness_F + readiness_N) / 10


def observe(busy, call_type):
    """Build the agent's observation for a decision: busy flags, call type, and the extra travel of each ranked slot.

    Input
        busy      : the busy engines as a bit mask (int, 0-6)
        call_type : index into CALL_TYPES (int)
    Output
        The observation (float32 array of 10): three busy flags, a one-hot of the four call types, and for slots 0-2 the
        candidate's extra travel over the closest free engine in units of ten minutes (0 for an empty slot); and the
        action mask (bool array of 3), True for slots that hold a free engine.
    Example
        observe(0, 2) -> ([0, 0, 0, 0, 0, 1, 0, 0.0, 0.05, 0.45], [True, True, True])    # C, B, A at M
    """
    ranked = free_engines_ranked(busy, call_type)
    cell = CALL_TYPES[call_type][1]
    extra = [(TRAVEL_MIN[engine, cell] - TRAVEL_MIN[ranked[0], cell]) / 10 for engine in ranked] + [0.0] * (N_ENGINES - len(ranked))
    flags = [float(busy >> engine & 1) for engine in range(N_ENGINES)]
    observation = np.array(flags + list(np.eye(N_CALL_TYPES)[call_type]) + extra, dtype=np.float32)
    return observation, np.array([slot < len(ranked) for slot in range(N_ENGINES)])


class SmallDispatchEnv(gym.Env):
    """The small system as a gymnasium environment: calls arrive, the agent ranks a free engine, engines finish later.

    Each step is one decision. Its reward is the decision's cost plus the outside-help cost of every call that found all
    engines busy before the next decision, the same accounting as solve. With shaping, the readiness term
    GAMMA x Phi(next state) - Phi(state) is added; the gate compares training with and without it to decide whether the
    full system trains with it. An episode ends (truncation) after EPISODE_DECISIONS decisions.

    Input (constructor)
        fire_weight : the weight of a fire call (float)
        seed        : the random seed of the call and service draws (int)
        shaping     : whether the potential-based shaping term is added (bool)
    Output
        A gymnasium environment with a Box(10) observation and a Discrete(3) action; action_masks() returns the mask.
    Example
        env = SmallDispatchEnv(8.0, seed=1); observation, info = env.reset(); env.step(0)
    """

    def __init__(self, fire_weight, seed=0, shaping=False):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=2.0, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ENGINES)
        self.weights_k, self.fire_weight, self.shaping = call_weights(fire_weight), fire_weight, shaping
        self.rng = np.random.default_rng(seed)

    def _next_decision(self):
        cost = 0.0
        while True:
            self.now += self.rng.exponential(1 / RATES.sum())
            self.free_at = np.where(self.free_at <= self.now, 0.0, self.free_at)
            call_type = int(self.rng.choice(N_CALL_TYPES, p=CALL_PROBABILITY))
            busy = sum(1 << engine for engine in range(N_ENGINES) if self.free_at[engine] > self.now)
            if busy != FULL:
                self.busy, self.call_type = busy, call_type
                return cost
            cost += self.weights_k[call_type] * OUTSIDE_HELP_MIN / 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.now, self.free_at, self.decisions = 0.0, np.zeros(N_ENGINES), 0
        self._next_decision()
        return observe(self.busy, self.call_type)[0], {}

    def action_masks(self):
        return observe(self.busy, self.call_type)[1]

    def step(self, action):
        ranked = free_engines_ranked(self.busy, self.call_type)
        engine = ranked[min(int(action), len(ranked) - 1)]
        cell = CALL_TYPES[self.call_type][1]
        reward = -self.weights_k[self.call_type] * TRAVEL_MIN[engine, cell] / 10
        before = potential(self.busy, self.fire_weight)
        self.free_at[engine] = self.now + self.rng.exponential(SERVICE_MEAN_MIN / 60)
        reward -= self._next_decision()
        if self.shaping:
            reward += GAMMA * potential(self.busy, self.fire_weight) - before
        self.decisions += 1
        return observe(self.busy, self.call_type)[0], reward, False, self.decisions >= EPISODE_DECISIONS, {}


def greedy_ranks(model):
    """Read the trained agent's greedy rank in every decision state of the small system.

    Input
        model : a trained MaskablePPO model for SmallDispatchEnv
    Output
        The rank the agent takes in each state (int array, 8 x 4; the all-busy row is 0).
    Example
        greedy_ranks(model)[0, 2] -> 1 when the agent learned to keep C for the rural cell
    """
    ranks = np.zeros((8, N_CALL_TYPES), dtype=int)
    for busy in range(FULL):
        for call_type in range(N_CALL_TYPES):
            observation, mask = observe(busy, call_type)
            ranks[busy, call_type] = int(model.predict(observation, action_masks=mask, deterministic=True)[0])
    return ranks


def start_value(values):
    """Average the decision value over the first call when every engine is free: the summary compared across policies.

    Input
        values : the decision values from solve (array 8 x 4)
    Output
        The expected value of the first decision of a day with all engines free (float, negative cost).
    Example
        start_value(solve(8.0)[0]) -> the optimum's value from an idle system
    """
    return float(values[0] @ CALL_PROBABILITY)


if __name__ == "__main__":
    for fire_weight in FIRE_WEIGHTS:                                   # the exact problem first: does the optimum beat the protocol?
        optimal_values, optimal_ranks = solve(fire_weight)
        protocol_values, _ = solve(fire_weight, np.zeros((8, N_CALL_TYPES), dtype=int))
        print(f"fire weight {fire_weight}: protocol {start_value(protocol_values):.4f}, optimum {start_value(optimal_values):.4f}, "
              f"improvement {100 * (start_value(optimal_values) - start_value(protocol_values)) / -start_value(protocol_values):.2f}% "
              f"of the protocol's cost; optimal rank with all free at M: {optimal_ranks[0, 2]}")
