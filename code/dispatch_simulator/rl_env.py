"""The learned dispatcher's environment: the simulator as a gymnasium environment, one synthetic day per episode.

Each episode samples a training day, a seed and a condition (nominal, more calls, a short surge, engines out, a closed
station or a large fire), builds that day's scenario and runs the simulator. Every dispatch decision of the day goes to
the agent: it picks one of CANDIDATES eligible free engines, the protocol's engine first and the closest other engines
after it, so that the protocol is always the first action. The simulator runs in a thread; the policy
callback hands the observation to the agent and waits for its action, so environment.simulate is used unchanged.

- Observation: 208 numbers that describe the call, the time, the system (coverage, the estimated cost of the runs
  still on their way and the share of quints free), the five candidates and every engine, built only from what is known
  at the decision moment.
- Reward: each arrival of an engine at a call of the day costs w_class x (minutes from the call to the arrival) / 10;
  a call's later arrivals cost LATER_ARRIVAL_WEIGHT times as much; a call released while waiting with no engine costs
  its wait. Each dispatch also costs its engine type as the protocol prices it (type_cost): an elevator call without a
  quint, a brush fire with a quint, or another call with a quint costs the protocol's quint penalty in seconds. Outage
  stand-ins are not calls and cost nothing. Almost half of the arrivals come after the next decision,
  so a potential-based term moves each sent run's estimated cost to the decision that sent it: the potential is minus
  the estimated cost of the runs still on their way. The term keeps the discounted objective's ordering of policies.
- Episode: the six hours before the day are simulated under the protocol (warm-up); the episode ends at the first
  decision after midnight, whose observation is the bootstrap state. If the scenario runs out of calls first, the
  episode terminates instead (no bootstrap, potential 0).
"""
import queue
import threading
import traceback

import gymnasium as gym
import numpy as np
import pandas as pd
import pyproj

from environment import (CALL_CLASSES, DAY_ZERO, OUT_OF_SERVICE_PROTOCOL, PROJECTED_CRS, READINESS_CELL_M, TRAINING_MONTHS_END,
                         VERBOSE, build_readiness_cells, make_readiness_tools, share_of_way_home, simulate)
from measures import HOTSPOT_STATIONS
from policies import closest_engine
from scenarios import (SURGE_EXTRA_CALLS, add_demand_growth, add_large_fires, add_outages, add_short_surges, build_scenario,
                       close_station, generate_calls)

CANDIDATES = 5                                              # [DESIGN] K: the action picks the protocol's engine or one of the four closest other eligible free engines
WARMUP_HOURS = 6                                            # [DESIGN] hours before the day simulated under the protocol, so engines start the day realistically
TAIL_HOURS = 6                                              # [DESIGN] hours after the day kept in the scenario, so that a decision after midnight exists (the night hours 0-6 hold about 14 calls of a day)
REWARD_UNIT_MIN = 10.0                                      # [DESIGN] rewards are counted in units of ten minutes
MINUTES_CAP = 120.0                                         # [DESIGN] minutes on the current run are capped here
HOME_SCALE_MIN = 30.0                                       # [DESIGN] minutes until home are divided by this
CLASS_WEIGHTS = {"study fire": 8.0, "high-acuity medical": 1.0, "low-acuity": 0.5, "other": 1.0}  # [DESIGN] the representative weight point of the grid
LATER_ARRIVAL_WEIGHT = 0.5                                  # [DESIGN] beta: a call's later arrivals weigh this much of its first
GAMMA = 0.97                                                # [DESIGN] discount per decision; the shaping term uses the same value
CONDITION_MIX = {"nominal": 0.30, "more calls": 0.20, "surge": 0.20, "engines out": 0.15, "closed station": 0.10, "large fire": 0.05}  # [DESIGN] share of episodes per condition
GROWTH_FACTORS = [1.1, 1.3, 1.5]                            # [DESIGN] demand factors of the "more calls" condition
OUTAGE_COUNTS = [1, 2, 3]                                   # [PLACEHOLDER] engines out of service in the "engines out" condition
FIRST_DAY, LAST_DAY = 1, 180                                # [DESIGN] episode days, so that the day before and after are also training days
ELEVATOR_PROTOCOL = "F56"                                   # [SOURCE] MPDS protocol of elevator calls, which CAD treats as a quint call
OBSERVATION_SIZE = 208                                      # [DESIGN] call 12 + time 3 + system 8 + candidates 5 x 8 + engines 29 x 5
PENDING_INDEX = 21                                          # [DESIGN] position of the estimated cost of the runs on their way (the last system number)


class EndOfEpisode(Exception):
    """Raised inside the simulation thread to stop the simulation when the episode is over.

    SimPy stops the run only after it has processed the other events of the same second, so another decision in that
    same second must also raise at once instead of waiting for an action (DispatchEnv.stopping)."""


def fit_rl_inputs(world, verbose=VERBOSE):
    """Fit what the learned dispatcher needs beyond the world: the non-fire demand map and finite coverage tools.

    Input
        world   : the dict from environment.build_world
        verbose : whether to print the non-fire map (bool)
    Output
        A dict: tau_max (the largest finite modeled travel from any station to any demand cell, minutes; it replaces the
        travel of a cell without an eligible idle engine); readiness_F, opportunity_F, readiness_N, opportunity_N (the
        finite readiness tools for the fire and the non-fire surface); cells_F, cells_N (the cells with their call
        shares); hour_share_F, hour_share_N; lambda_scale (the largest cell share x hour share over both surfaces);
        turnout_s (the median recorded turnout of the training months, used to estimate when an engine on its way will
        arrive); class_shares (the share of training-month runs in each call class); engine_is_gfr, engine_is_quint and
        engine_position (per engine, in the engines table's order); in_service_from_s, in_service_to_s (each engine's service dates in
        seconds since DAY_ZERO).
    Example
        rl = fit_rl_inputs(world, verbose=False)
        rl["tau_max"] -> the farthest finite travel in minutes, 63.4
    """
    cells_N, travel_N, hour_share_N = build_readiness_cells(world["dispatches"], world["graph"], world["engines"], world["home_nodes"],
                                                            world["aid_protocols"], world["seconds_from_node"], world["node_position"],
                                                            world["travel_model"], fire=False, verbose=verbose)
    cells_F, travel_F, hour_share_F = world["readiness_cells"], world["travel_to_cells"], world["fire_hour_share"]
    tau_max = float(max(np.nanmax(np.where(np.isfinite(travel_F), travel_F, np.nan)), np.nanmax(np.where(np.isfinite(travel_N), travel_N, np.nan)))) / 60
    readiness_F, opportunity_F = make_readiness_tools(cells_F, travel_F, world["engines"], tau_max)
    readiness_N, opportunity_N = make_readiness_tools(cells_N, travel_N, world["engines"], tau_max)
    share_F = (cells_F["weight_aid"] + cells_F["weight_own"]).to_numpy()
    share_N = (cells_N["weight_aid"] + cells_N["weight_own"]).to_numpy()
    record, runs = world["record"], world["runs"]
    training = (world["dispatches"]["call_received_at"] < TRAINING_MONTHS_END).to_numpy()
    class_shares = pd.Series(record["call_class"].to_numpy()[training]).value_counts(normalize=True).reindex(range(4), fill_value=0.0)
    training_runs = runs["call_received_at"] < (TRAINING_MONTHS_END - DAY_ZERO).total_seconds()
    turnout = (runs["enroute_at"] - runs["dispatched_at"])[training_runs].dropna()
    engines = world["engines"]
    rl = {"tau_max": tau_max, "readiness_F": readiness_F, "opportunity_F": opportunity_F, "readiness_N": readiness_N,
          "opportunity_N": opportunity_N, "cells_F": cells_F, "cells_N": cells_N, "hour_share_F": hour_share_F, "hour_share_N": hour_share_N,
          "cell_share_F": dict(zip(zip(cells_F["cell_x"], cells_F["cell_y"]), share_F)),
          "cell_share_N": dict(zip(zip(cells_N["cell_x"], cells_N["cell_y"]), share_N)),
          "lambda_scale": max(share_F.max() * hour_share_F.max(), share_N.max() * hour_share_N.max()),
          "turnout_s": float(turnout.median()), "class_shares": class_shares,
          "engine_is_gfr": (engines["engine_agency"] == "GFR").to_numpy(),
          "engine_is_quint": world["quint_indicator"].reindex(engines["engine_id"]).fillna(0.0).to_numpy() > 0,
          "engine_position": pd.Series(np.arange(len(engines)), index=engines["engine_id"]),
          "in_service_from_s": (engines["in_service_from"] - DAY_ZERO).dt.total_seconds().to_numpy(),
          "in_service_to_s": (engines["in_service_to"] - DAY_ZERO).dt.total_seconds().to_numpy()}
    if verbose:
        print(f"tau_max {tau_max:.1f} min; median turnout {rl['turnout_s']:.0f} s; training runs by class "
              f"{dict(zip(CALL_CLASSES, class_shares.round(3)))}\n")
    return rl


def sample_scenario(world, generator, rng, verbose=False):
    """Build one episode's scenario: a random training day, a fresh seed and a condition from CONDITION_MIX.

    The calls of the day before, the day and the day after are generated, the condition is applied to the day, and the
    calls from WARMUP_HOURS before the day to TAIL_HOURS after it are kept.

    Input
        world     : the dict from environment.build_world
        generator : the dict from scenarios.fit_generator
        rng       : the random generator of the episode draws (numpy Generator)
        verbose   : whether the scenario builder prints its checks (bool)
    Output
        The scenario (dict from scenarios.build_scenario) with three more keys: day (days since DAY_ZERO), condition (str)
        and day_start_s, day_end_s (seconds since DAY_ZERO).
    Example
        scenario = sample_scenario(world, generator, np.random.default_rng(1))
        scenario["condition"], scenario["day"] -> e.g. ('surge', 57)
    """
    day = int(rng.integers(FIRST_DAY, LAST_DAY + 1))
    condition = str(rng.choice(list(CONDITION_MIX), p=list(CONDITION_MIX.values())))
    seed = int(rng.integers(1, 2**31 - 1))
    pool, counts, locations = generator["pool"], generator["counts"], generator["locations"]
    calls, day_factors = generate_calls(pool, counts, generator["dispersion"], [day - 1, day, day + 1], seed, verbose=False)
    outages = pd.DataFrame(columns=["engine_id", "start_s", "end_s"])
    if condition == "more calls":
        calls = add_demand_growth(calls, day_factors, pool, counts, locations, float(rng.choice(GROWTH_FACTORS)), seed, verbose=False)
    elif condition == "surge":
        calls = add_short_surges(calls, pool, locations, [day], int(rng.choice(SURGE_EXTRA_CALLS)), seed, verbose=False)[0]
    elif condition == "engines out":
        outages = add_outages(world["engines"], [day], int(rng.choice(OUTAGE_COUNTS)), seed, verbose=False)
    elif condition == "closed station":
        outages = close_station(world["engines"], str(rng.choice(HOTSPOT_STATIONS)), [day - 1, day, day + 1], verbose=False)
    elif condition == "large fire":
        calls = add_large_fires(calls, pool, [day], seed, verbose=False)
    start_s, end_s = day * 86400 - WARMUP_HOURS * 3600, (day + 1) * 86400 + TAIL_HOURS * 3600
    calls = calls[(calls["received_s"] >= start_s) & (calls["received_s"] < end_s)].reset_index(drop=True)
    scenario = build_scenario(world, generator, calls, outages, seed, verbose=verbose)
    scenario.update(day=day, condition=condition, day_start_s=day * 86400.0, day_end_s=(day + 1) * 86400.0)
    return scenario


def prepare_scenario(world, rl, scenario):
    """Precompute the per-run arrays the observation and the reward read, so that each decision only looks them up.

    Input
        world    : the dict from environment.build_world
        rl       : the dict from fit_rl_inputs
        scenario : a scenario dict (from sample_scenario, or the record with day_start_s and day_end_s added)
    Output
        A dict of arrays indexed by run row: call_received_s, dispatch_s (the run's dispatch time in the scenario,
        seconds since DAY_ZERO), call_class, elevator, brush, in_gfr, share_F, share_N (the call cell's share of fire and
        non-fire calls), incident (an integer code per call), weight (the class weight), in_day (whether the call was
        received during the day), stand_in (whether the run is an outage stand-in), penalty_s (the protocol's quint
        penalty for the run's call type, seconds), plus rows_of_incident (the run rows of each incident code) and
        quint_engines (the ids of the quints).
    Example
        prepared = prepare_scenario(world, rl, scenario); prepared["call_class"][:5] -> e.g. [1, 3, 3, 2, 1]
    """
    table, runs = scenario["dispatches"], scenario["runs"]
    lon, lat = table["incident_lon"].to_numpy(dtype=float), table["incident_lat"].to_numpy(dtype=float)
    x, y = pyproj.Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform(lon, lat)
    located = np.isfinite(x) & np.isfinite(y)                       # outage stand-ins have no location
    cells = list(zip(np.where(located, np.floor(np.where(located, x, 0) / READINESS_CELL_M), 0).astype(int),
                     np.where(located, np.floor(np.where(located, y, 0) / READINESS_CELL_M), 0).astype(int)))
    incident = pd.factorize(runs["incident_id"])[0]
    call_class = scenario["call_class"].to_numpy()
    received = runs["call_received_at"].to_numpy(dtype=float)
    rows_of_incident = pd.Series(np.arange(len(runs))).groupby(incident).apply(list).to_dict()
    return {"call_received_s": received, "dispatch_s": runs["dispatched_at"].to_numpy(dtype=float), "call_class": call_class,
            "elevator": table["protocol"].eq(ELEVATOR_PROTOCOL).fillna(False).to_numpy(dtype=bool),
            "brush": scenario["brush"].to_numpy(dtype=bool), "in_gfr": table["jurisdiction"].eq("GFR").fillna(False).to_numpy(dtype=bool),
            "share_F": np.array([rl["cell_share_F"].get(cell, 0.0) for cell in cells]) * located,
            "share_N": np.array([rl["cell_share_N"].get(cell, 0.0) for cell in cells]) * located,
            "incident": incident, "rows_of_incident": rows_of_incident,
            "weight": np.array([CLASS_WEIGHTS[CALL_CLASSES[c]] for c in call_class]),
            "in_day": (received >= scenario["day_start_s"]) & (received < scenario["day_end_s"]),
            "stand_in": table["protocol"].eq(OUT_OF_SERVICE_PROTOCOL).fillna(False).to_numpy(dtype=bool),
            "penalty_s": scenario["quint_penalty"].to_numpy(dtype=float),
            "quint_engines": set(world["quint_indicator"][world["quint_indicator"] > 0].index)}


def minutes_until_home(world, scenario, engine_id, last_clear, now):
    """Estimate how many minutes a free engine still needs to reach its station; 0 when it is there.

    Input
        world, scenario : the world and the scenario being simulated (dicts)
        engine_id       : the engine (str)
        last_clear      : the engine's last clear as kept by simulate (tuple or None)
        now             : the current simulation time, seconds since DAY_ZERO (float)
    Output
        The remaining drive home in minutes (float): the share of the way not yet covered x the fitted slowdown x the
        modeled drive home.
    Example
        minutes_until_home(world, scenario, "E3", (cleared_s, row), cleared_s + 300) -> about 6.8 for a 600 s drive home
    """
    share, from_scene = share_of_way_home(engine_id, last_clear, now, scenario["call_nodes"], world["home_nodes"], world["seconds_from_node"],
                                          world["node_position"], world["travel_model"], world["dwell_s"], world["slowdown"])
    if share is None:
        return 0.0
    drive_home_s = float(world["travel_model"](from_scene[world["node_position"][world["home_nodes"][engine_id]]]))
    return (1 - share) * world["slowdown"] * drive_home_s / 60


def observe(world, rl, scenario, prepared, run, available, travel_s, context, dispatched):
    """Build the agent's observation for one decision from what is known at that moment.

    Input
        world, rl, scenario, prepared : the world, the inputs from fit_rl_inputs, the scenario and its prepared arrays
        run        : the run being dispatched (Series; only its row label is read)
        available  : the engines in service and free now (list of str)
        travel_s   : each engine's modeled travel from where it is now to the call, seconds (Series; NaN = not allowed)
        context    : what simulate knows now (dict: now, engine_state, busy_runs, simulated, waiting_runs)
        dispatched : the environment's own record of earlier dispatches, row -> (dispatch time, modeled travel seconds,
                     estimated cost from estimated_cost)
    Output
        Three objects: the observation (float32 array of OBSERVATION_SIZE numbers, in the order call, time, system,
        candidates, engines), the action mask (bool array of CANDIDATES) and the candidate engine ids: the protocol's
        engine first, then the other eligible free engines closest first (list of str). The system block ends with the estimated cost of the runs on their way / 10 (PENDING_INDEX) and
        the share of quints free.
    Example
        observation, mask, candidates = observe(world, rl, scenario, prepared, run, available, travel_s, context, {})
        observation.shape, mask, candidates[:2] -> ((208,), [True, True, True, True, True], ['E62', 'E3'])
    """
    now, row, simulated, engine_state = context["now"], run.name, context["simulated"], context["engine_state"]
    eligible = travel_s[available].dropna().sort_values(kind="stable")
    protocol_engine = closest_engine(world, scenario, run, available, travel_s)
    candidates = [protocol_engine] + [engine_id for engine_id in eligible.index if engine_id != protocol_engine][:CANDIDATES - 1]
    closest_min = float(eligible.iloc[0]) / 60
    arrived, responding, remaining = 0, 0, []
    for other in prepared["rows_of_incident"][prepared["incident"][row]]:
        if other == row:
            continue
        if not np.isnan(simulated.at[other, "arrived_at"]):
            arrived += 1
        elif other in dispatched and np.isnan(simulated.at[other, "cleared_at"]) and isinstance(simulated.at[other, "engine_id"], str):
            responding += 1
            sent_at, modeled_s, _ = dispatched[other]
            remaining.append(max(0.0, sent_at + rl["turnout_s"] + modeled_s - now) / 60)
    hour = (now % 86400) / 3600
    call_class = np.eye(4)[prepared["call_class"][row]]
    lambda_F = prepared["share_F"][row] * rl["hour_share_F"].iloc[int(hour)] / rl["lambda_scale"]
    lambda_N = prepared["share_N"][row] * rl["hour_share_N"].iloc[int(hour)] / rl["lambda_scale"]
    call = [*call_class, prepared["elevator"][row], prepared["brush"][row], arrived / 4, responding / 4,
            (min(remaining) if remaining else 0.0) / 10, closest_min / 10, lambda_F, lambda_N]
    weekend = float((DAY_ZERO + pd.Timedelta(seconds=now)).dayofweek >= 5)
    time = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), weekend]
    idle = np.zeros(len(rl["engine_position"]), dtype=bool)
    idle[rl["engine_position"][available].to_numpy()] = True
    readiness_F, _, uncovered_F = rl["readiness_F"](idle)
    readiness_N, _, uncovered_N = rl["readiness_N"](idle)
    arrived_at, cleared_at = simulated["arrived_at"].to_numpy(dtype=float), simulated["cleared_at"].to_numpy(dtype=float)
    pending = sum(cost for other, (_, _, cost) in dispatched.items() if cost and np.isnan(arrived_at[other]) and np.isnan(cleared_at[other]))
    system = [len(available) / len(idle), min(context["waiting_runs"], 10) / 10, readiness_F / 10, readiness_N / 10, uncovered_F,
              uncovered_N, pending / 10, rl["engine_is_quint"][idle].sum() / rl["engine_is_quint"].sum()]
    effects_F = rl["opportunity_F"](available, candidates, uncovered=True)
    effects_N = rl["opportunity_N"](available, candidates, uncovered=True)
    in_gfr = prepared["in_gfr"][row]
    home_minutes = {engine_id: minutes_until_home(world, scenario, engine_id, engine_state.get(engine_id), now) for engine_id in available}
    slots = []
    for slot in range(CANDIDATES):
        if slot >= len(candidates):
            slots += [0.0] * 8
            continue
        engine_id = candidates[slot]
        position = rl["engine_position"][engine_id]
        slots += [(float(eligible[engine_id]) / 60 - closest_min) / 10,
                  home_minutes[engine_id] / HOME_SCALE_MIN,
                  float(world["quint_indicator"][engine_id]), float(rl["engine_is_gfr"][position] != in_gfr),
                  10 * effects_F.at[engine_id, "oc"], 10 * effects_N.at[engine_id, "oc"],
                  effects_F.at[engine_id, "uncovered_rise"], effects_N.at[engine_id, "uncovered_rise"]]
    engines_block = []
    busy_runs, available_set = context["busy_runs"], set(available)
    in_service = (rl["in_service_from_s"] <= now) & (rl["in_service_to_s"] >= now)
    for position, engine_id in enumerate(rl["engine_position"].index):
        holding = busy_runs.get(engine_id)
        out = not in_service[position] or (holding is not None and bool(prepared["stand_in"][holding]))
        started = np.nan if holding is None or out else simulated.at[holding, "dispatched_at"]
        # an engine sent in this same second may not have its run recorded yet: it counts as on a run for 0 minutes
        on_scene = not np.isnan(started) and not np.isnan(simulated.at[holding, "arrived_at"])
        on_run = 0.0 if np.isnan(started) else min((now - started) / 60, MINUTES_CAP) / MINUTES_CAP
        engines_block += [float(engine_id in available_set), float(out), float(on_scene), on_run, home_minutes.get(engine_id, 0.0) / HOME_SCALE_MIN]
    observation = np.array(call + time + system + slots + engines_block, dtype=np.float32)
    mask = np.array([slot < len(candidates) for slot in range(CANDIDATES)])
    return observation, mask, candidates


def type_cost(prepared, dispatched, row, engine_id):
    """Price the engine type of a dispatch as the protocol does, so that dropping the quint rule is not free.

    The protocol ranks a quint as if its travel were longer by the call type's quint penalty (shorter when the penalty is
    negative). As a cost: an elevator call (penalty -260 s) costs 260 s when the engine sent is not a quint, a brush fire
    (+260 s) costs 260 s when it is a quint, and any other call (+10 s) costs 10 s when it is a quint. It is a chosen type
    preference, priced as CAD's choices imply. On the costs known at the decision (the estimated arrival cost plus this
    type cost) the protocol's engine is the cheapest for the current call; the realized reward can differ, because travel
    is drawn after the choice, a call's later arrivals are weighted by arrival order, and costs that come later are
    discounted.

    Input
        prepared   : the scenario's prepared arrays (dict from prepare_scenario)
        dispatched : the runs sent earlier (any container of run rows: membership is read)
        row        : the run being sent (int); engine_id : the engine sent (str)
    Output
        weight x share x the type cost in seconds / 60 / 10 (float, reward units, zero or positive); share is 1 for the
        first engine sent to the call and LATER_ARRIVAL_WEIGHT otherwise, as for its arrival; 0 for a run outside the day
        or an outage stand-in.
    Example
        type_cost(prepared, {}, row_of_an_elevator_call, "E3") -> 0.433 for a medical call (weight 1): 260 / 600
    """
    if not prepared["in_day"][row] or prepared["stand_in"][row]:
        return 0.0
    penalty = prepared["penalty_s"][row]
    cost_s = max(penalty, 0.0) if engine_id in prepared["quint_engines"] else max(-penalty, 0.0)
    first = not any(other in dispatched for other in prepared["rows_of_incident"][prepared["incident"][row]] if other != row)
    return prepared["weight"][row] * (1.0 if first else LATER_ARRIVAL_WEIGHT) * cost_s / 60 / REWARD_UNIT_MIN


def estimated_cost(rl, prepared, dispatched, row, now, modeled_s):
    """When a run is sent, estimate the reward it will cost on arrival from what is known at that moment.

    Input
        rl, prepared : the inputs from fit_rl_inputs and the scenario's prepared arrays (dicts)
        dispatched   : the environment's record of earlier dispatches (dict, row -> (time, modeled seconds, cost))
        row          : the run being sent (int); now : the dispatch time, seconds since DAY_ZERO (float)
        modeled_s    : the modeled travel of the engine sent, seconds (float)
    Output
        weight x share x (minutes from the call to the expected arrival: now + median turnout + modeled travel) / 10
        (float, reward units); share is 1 for the first engine sent to the call and LATER_ARRIVAL_WEIGHT otherwise;
        0 for a run outside the day or an outage stand-in, which the reward never charges.
    Example
        estimated_cost(rl, prepared, {}, 12, call_s + 60, 300.0) -> about 0.68 for a medical call: (60 + 46 + 300) / 600
    """
    if not prepared["in_day"][row] or prepared["stand_in"][row]:
        return 0.0
    first = not any(other in dispatched for other in prepared["rows_of_incident"][prepared["incident"][row]] if other != row)
    share = 1.0 if first else LATER_ARRIVAL_WEIGHT
    return prepared["weight"][row] * share * (now + rl["turnout_s"] + modeled_s - prepared["call_received_s"][row]) / 60 / REWARD_UNIT_MIN


def potential(observation):
    """The shaping potential of a state: minus the estimated cost of the runs on their way, in reward units.

    A run sent now is on its way at the next decision, so its estimated cost enters the potential there and the
    shaping term GAMMA x Phi(next) - Phi(now) charges about that cost on the transition that sent it. When the run
    arrives, the estimate leaves the potential and the reward charges the true cost. Over a day the term telescopes, so
    the discounted ordering of policies is kept.

    Input
        observation : an observation from observe (array); position PENDING_INDEX holds the estimated cost / 10
    Output
        Phi(s) = -(estimated cost of the runs dispatched and not yet arrived or released) (float, reward units).
    Example
        potential(observation) -> -1.4 when two runs with estimated costs 0.6 and 0.8 are on their way
    """
    return -10.0 * float(observation[PENDING_INDEX])


def new_tally(prepared):
    """Start the bookkeeping of counted reward events for one scenario.

    Input
        prepared : the prepared arrays of the scenario (dict from prepare_scenario)
    Output
        A dict: arrival and release (boolean arrays, one per run row: whether that event was counted; all False) and
        first_arrival_s (one per incident code: the time of the call's first counted arrival in seconds since DAY_ZERO;
        all inf).
    Example
        tally = new_tally(prepared); np.isinf(tally["first_arrival_s"]).all() -> True
    """
    runs, incidents = len(prepared["incident"]), int(prepared["incident"].max()) + 1
    return {"arrival": np.zeros(runs, dtype=bool), "release": np.zeros(runs, dtype=bool), "first_arrival_s": np.full(incidents, np.inf)}


def accrued_reward(prepared, simulated, tally, complete_before=np.inf):
    """Sum the reward of the day's arrivals and releases that have happened and are not counted yet.

    The run table holds an arrival or a release from the moment it happens, so the events not counted yet are exactly
    those since the previous count, including any in the same second as the previous decision. A call's first arrival
    costs its class weight x minutes since the call / 10, and each later arrival LATER_ARRIVAL_WEIGHT times that. A run
    released with no engine costs its weight x the minutes it waited / 10, unless the call had an engine on scene by the
    moment of release (an arrival in the same second counts as before it). A release is therefore settled only once its
    second is complete: releases at or after complete_before wait for a later count. Outage stand-ins are not calls and
    cost nothing.

    Input
        prepared        : the prepared arrays of the scenario (dict from prepare_scenario)
        simulated       : the simulator's run table so far (DataFrame with arrived_at, cleared_at and engine_id)
        tally           : the events counted so far (dict from new_tally; updated in place)
        complete_before : the simulation time before which every event has happened, seconds since DAY_ZERO (float):
                          the current decision's time, or inf when the simulation has finished
    Output
        The reward of the newly counted events (float, units of ten weighted minutes; zero or negative).
    Example
        accrued_reward(prepared, simulated, new_tally(prepared)) -> the unshaped return of a finished simulation, e.g. -126.8
    """
    arrived = simulated["arrived_at"].to_numpy(dtype=float)
    cleared = simulated["cleared_at"].to_numpy(dtype=float)
    calls = prepared["in_day"] & ~prepared["stand_in"]
    reward = 0.0
    arrivals = np.flatnonzero(calls & ~np.isnan(arrived) & ~tally["arrival"])
    for row in arrivals[np.argsort(arrived[arrivals], kind="stable")]:
        incident = prepared["incident"][row]
        share = LATER_ARRIVAL_WEIGHT if np.isfinite(tally["first_arrival_s"][incident]) else 1.0
        reward -= prepared["weight"][row] * share * (arrived[row] - prepared["call_received_s"][row]) / 60 / REWARD_UNIT_MIN
        tally["first_arrival_s"][incident] = min(tally["first_arrival_s"][incident], arrived[row])
        tally["arrival"][row] = True
    releases = np.flatnonzero(calls & simulated["engine_id"].isna().to_numpy() & (cleared < complete_before) & ~tally["release"])
    for row in releases:
        if tally["first_arrival_s"][prepared["incident"][row]] > cleared[row]:    # no engine on scene by the release
            reward -= prepared["weight"][row] * (cleared[row] - prepared["dispatch_s"][row]) / 60 / REWARD_UNIT_MIN
        tally["release"][row] = True
    return reward


class DispatchEnv(gym.Env):
    """The full dispatch system as a gymnasium environment for MaskablePPO; one synthetic day per episode.

    Input (constructor)
        world     : the dict from environment.build_world
        generator : the dict from scenarios.fit_generator
        rl        : the dict from fit_rl_inputs
        seed      : the seed of the episode draws (int)
        shaping   : whether the potential-based term that moves each run's estimated cost to its dispatch is added (bool)
    Output
        A gymnasium environment with a Box(OBSERVATION_SIZE) observation and a Discrete(CANDIDATES) action;
        action_masks() gives the mask. At the end of an episode, info holds the unshaped return, the number of
        decisions, how many decisions sent another engine than the protocol would, and end_s: the time of the decision
        after midnight that ends the episode, or inf when the scenario ran out of calls (releases from that second on
        are not counted). At the current decision, protocol_engine is the protocol's engine, protocol_slot its slot
        (always 0) and closest_slot the slot of the engine with the least modeled travel (in a tie, the first listed). reset(options={"scenario": s, "full_day": True}) replays a
        given scenario and lets the agent decide every call to its end, so that two policies can be compared on the
        same calls; the episode then ends when the scenario runs out of calls.
    Example
        env = DispatchEnv(world, generator, rl, seed=1); observation, info = env.reset(); env.step(0)
    """

    def __init__(self, world, generator, rl, seed=0, shaping=True):
        super().__init__()
        self.world, self.generator, self.rl, self.shaping = world, generator, rl, shaping
        self.rng = np.random.default_rng(seed)
        assert 12 + 3 + 8 + 8 * CANDIDATES + 5 * len(rl["engine_position"]) == OBSERVATION_SIZE, "the observation layout changed"
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(CANDIDATES)
        self.thread, self.stopping = None, False

    def _policy(self, world, scenario, run, available, travel_s, context):
        if self.stopping:                                           # the episode is over: a decision in the same second ends too
            raise EndOfEpisode                                      # (checked first, so the counted table kept by step stays in place)
        now = context["now"]
        self.simulated = context["simulated"]                       # the live run table; simulate fills it in place
        if now < scenario["day_start_s"] or travel_s[available].dropna().empty:   # warm-up, or no engine may go: the protocol decides
            engine_id = closest_engine(world, scenario, run, available, travel_s)
        else:
            observation, mask, candidates = observe(world, self.rl, scenario, self.prepared, run, available, travel_s, context, self.dispatched)
            self.candidates, self.protocol_engine = candidates, closest_engine(world, scenario, run, available, travel_s)
            self.protocol_slot = candidates.index(self.protocol_engine) if self.protocol_engine in candidates else None
            self.closest_slot = candidates.index(travel_s[available].dropna().idxmin())   # ties: the first engine in table order, as the stable sort ranks it
            self.outbox.put(("decision", observation, mask, now))
            action = self.inbox.get()
            if action is None:
                raise EndOfEpisode
            engine_id = candidates[min(int(action), len(candidates) - 1)]
        if engine_id is not None:
            modeled_s = float(travel_s[engine_id])
            self.type_costs += type_cost(self.prepared, self.dispatched, run.name, engine_id)   # charged on the transition of this decision
            self.dispatched[run.name] = (now, modeled_s, estimated_cost(self.rl, self.prepared, self.dispatched, run.name, now, modeled_s))
        return engine_id

    def _run(self):
        try:
            simulate(self.world, self.scenario, self._policy, verbose=False)
            self.outbox.put(("finished", None, None, np.inf))
        except EndOfEpisode:
            pass
        except Exception:                                           # hand any simulation error to the waiting caller
            self.outbox.put(("error", traceback.format_exc(), None, np.nan))

    def _receive(self):
        kind, observation, mask, now = self.outbox.get()
        if kind == "error":
            raise RuntimeError(f"the simulation thread failed:\n{observation}")
        return kind, observation, mask, now

    def _stop(self):
        if self.thread is not None and self.thread.is_alive():
            self.stopping = True                                    # set before the wake-up, so the thread sees it
            self.inbox.put(None)
            self.thread.join()
        self.thread, self.stopping = None, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._stop()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        given = (options or {}).get("scenario")
        self.full_day = bool((options or {}).get("full_day", False))  # evaluation: decide every call to the end, count every arrival
        while True:                                                 # a sampled scenario without any decision is drawn again
            self.scenario = given or sample_scenario(self.world, self.generator, self.rng)
            self.prepared = prepare_scenario(self.world, self.rl, self.scenario)
            self.dispatched, self.tally, self.type_costs = {}, new_tally(self.prepared), 0.0
            self.outbox, self.inbox = queue.Queue(), queue.Queue()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            kind, self.observation, self.mask, _ = self._receive()
            if kind == "decision":
                break
            if given is not None:
                raise ValueError("the given scenario has no dispatch decision after its warm-up")
        self.unshaped, self.decisions, self.deviations = 0.0, 0, 0
        return self.observation, {"condition": self.scenario["condition"], "day": self.scenario["day"]}

    def action_masks(self):
        return self.mask

    def step(self, action):
        engine_id = self.candidates[min(int(action), len(self.candidates) - 1)]
        self.decisions += 1
        self.deviations += int(engine_id != self.protocol_engine)   # read before the action resumes the simulation, which moves on to the next decision
        previous_observation = self.observation
        self.inbox.put(int(action))
        kind, observation, mask, now = self._receive()
        terminated = kind == "finished"                             # the scenario ran out of calls: nothing follows
        truncated = kind == "decision" and now >= self.scenario["day_end_s"] and not self.full_day
        reward = accrued_reward(self.prepared, self.simulated, self.tally, now)   # releases in this decision's second wait for the next count (at truncation, for the bootstrap)
        reward, self.type_costs = reward - self.type_costs, 0.0     # the engine types sent since the last decision, this one included
        self.unshaped += reward
        if kind == "decision":
            self.observation, self.mask = observation, mask
        if self.shaping:
            next_potential = 0.0 if terminated else potential(self.observation)
            reward += GAMMA * next_potential - potential(previous_observation)
        info = {}
        if terminated or truncated:
            info = {"unshaped_return": self.unshaped, "decisions": self.decisions, "deviations": self.deviations,
                    "condition": self.scenario["condition"], "end_s": now}
            self.simulated = self.simulated.copy()                  # the table as counted; the stopping thread may still write this second's events
            self._stop()
        return self.observation, float(reward), terminated, truncated, info

    def close(self):
        self._stop()
