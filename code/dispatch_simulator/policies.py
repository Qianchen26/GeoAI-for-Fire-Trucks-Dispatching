"""Dispatch policies: the functions that choose which engine to send to a run.

A policy sees one dispatch decision and names the engine to send. The simulator calls a policy as
policy(world, scenario, run, available_engines, travel_s, context) and then draws the real travel of the chosen engine.
The context holds what is known at the decision (the current time and each engine's status); the two policies below do
not need it, and a learned policy builds its observation from it. A policy
therefore chooses only the engine and never changes any engine's travel. Two policies are built in:
- closest_engine is the protocol: CAD sends the available engine with the least modeled travel. When CAD ranks the
  engines, it adds the call type's quint penalty to the modeled travel of each quint. Every other policy is compared
  with the protocol.
- opportunity_cost_policy keeps the protocol's engine on most calls. On a low-acuity call, it may send an engine that is
  ranked almost as close but whose departure costs less fire readiness.
A learned policy takes the same inputs and returns an engine in the same way.
"""
import numpy as np
import pandas as pd

ACTION_WINDOW_S = 60    # [DESIGN] on a low-acuity call, an engine ranked within this many seconds of the protocol's engine may be sent instead (provisional)
OC_TIE_S = 1.0          # [DESIGN] the rule changes the engine only when the change saves more than this much fire readiness, in seconds; two engines of one station are a few metres apart, so their opportunity costs can differ by a tiny amount that should not change the engine


def closest_engine(world, scenario, run, available_engines, travel_s, context=None):
    """Return the protocol's engine, which is the available engine with the least modeled travel to the call.

    A quint is an engine that also carries an aerial ladder. When the policy ranks the engines, it adds the call type's
    quint penalty to the modeled travel of each quint, as CAD's choices in the record imply. The penalty changes only
    the ranking; it does not change the travel of the engine that is sent. When two engines tie in the ranking, the
    policy sends the one listed first in the engines table.

    Input
        world             : the dict from build_world; this policy reads world["quint_indicator"], which maps each
                            engine id to 1.0 for a quint and 0.0 otherwise (Series)
        scenario          : the calls being simulated (dict); this policy reads scenario["quint_penalty"], which maps
                            each run row to the seconds added to a quint's modeled travel (Series). For example, the
                            penalty is -260.0 on elevator calls, so a quint ranks as if it were 260 seconds closer.
        run               : the run being dispatched (Series). It holds incident_id, the record's engine_id, and the
                            times of the call, dispatch, en route, arrival and clear, in seconds. run.name is the run's
                            row label.
        available_engines : the engines in service and free now (list of str), for example ["E1", "E2", "E4", ...]
        travel_s          : the modeled travel of each engine from where it is now to this call, in seconds (Series
                            indexed by engine id). The value is NaN where the agency line or the engine's service dates
                            rule the engine out.
        context           : what is known at the decision (dict from simulate); the protocol does not use it
    Output
        The engine to send (str), for example "E62". The policy returns None when no available engine may go, and the
        run then waits until an engine clears.
    Example
        closest_engine(world, world["record"], world["runs"].loc[0], ["E62", "E3"], travel_s) -> "E62"   # E62 is 257.05 s away
    """
    candidates = travel_s[available_engines].dropna()
    if candidates.empty:
        return None
    ranked = candidates + (world["quint_indicator"] * scenario["quint_penalty"][run.name]).reindex(candidates.index, fill_value=0.0)
    return ranked.idxmin()


def rule_pick(costs, protocol_engine):
    """Pick the engine whose departure costs the least fire readiness, unless the saving is too small to matter.

    The saving is the opportunity cost of the protocol's engine minus the lowest opportunity cost in costs. The
    protocol's engine stays when the saving is OC_TIE_S seconds or less, and also when every cost is infinite.

    Input
        costs           : the opportunity cost of each engine in the action window, in minutes (Series indexed by
                          engine id)
        protocol_engine : the protocol's engine (str), which is one of the engines in costs
    Output
        The engine to send (str).
    Example
        rule_pick(pd.Series({"E21": 0.179, "E24": 0.108, "E28": 0.002}), "E21") -> "E28"     # saves 0.177 min = 10.6 s
    """
    with np.errstate(invalid="ignore"):                          # when every cost is infinite, the saving inf - inf is NaN; NaN fails the comparison below, so the protocol's engine stays
        saving = costs[protocol_engine] - costs.min()
    return costs.idxmin() if saving > OC_TIE_S / 60 else protocol_engine


def opportunity_cost_choice(world, scenario, run, available_engines, travel_s, action_window_s=ACTION_WINDOW_S):
    """Return the protocol's engine, the engine the opportunity-cost rule sends, and the opportunity costs it compared.

    The rule keeps the protocol's engine unless the call is low-acuity. On a low-acuity call, the rule ranks the
    available engines by modeled travel plus the quint penalty, as the protocol does. It then looks at the engines
    ranked within action_window_s seconds of the protocol's engine. Among them, it sends the engine whose departure
    costs the least fire readiness, unless the saving is too small to matter (rule_pick).

    Input
        world, scenario, run, available_engines, travel_s : as in closest_engine. The rule also reads two more entries:
                            world["opportunity_cost"] computes the opportunity costs, and scenario["low_acuity"] marks
                            each run row as low-acuity or not (Series of bool).
        action_window_s   : how many seconds behind the protocol's engine an engine may be ranked and still be sent
                            (float), for example 60
    Output
        Three objects: the protocol's engine (str or None), the engine the rule sends (str or None), and the opportunity
        costs of the engines in the window, in minutes (Series indexed by engine id). The Series is empty when the call
        is not low-acuity or when the window holds only one engine; the rule then sends the protocol's engine.
    Example
        opportunity_cost_choice(world, world["record"], world["runs"].loc[25], available, travel_s)
        -> ('E21', 'E28', E21 0.179 / E24 0.108 / E28 0.002)       # E28 is 49 s farther but costs less fire readiness
    """
    protocol_engine = closest_engine(world, scenario, run, available_engines, travel_s)
    costs = pd.Series(dtype=float)
    if protocol_engine is not None and scenario["low_acuity"][run.name]:
        rank_penalty = world["quint_indicator"] * scenario["quint_penalty"][run.name]
        ranked = (travel_s[available_engines] + rank_penalty.reindex(available_engines, fill_value=0.0)).dropna()
        near = ranked.index[ranked <= ranked[protocol_engine] + action_window_s]
        if len(near) > 1:
            costs = world["opportunity_cost"](available_engines, near)
    chosen = rule_pick(costs, protocol_engine) if len(costs) else protocol_engine
    return protocol_engine, chosen, costs


def opportunity_cost_policy(world, scenario, run, available_engines, travel_s, context=None, action_window_s=ACTION_WINDOW_S):
    """Return only the engine that opportunity_cost_choice sends, so that the opportunity-cost rule works as a policy.

    The simulator calls a policy with six arguments and cannot pass action_window_s. To try another action window,
    pass functools.partial(opportunity_cost_policy, action_window_s=...) as the policy.

    Input
        world, scenario, run, available_engines, travel_s, action_window_s : as in opportunity_cost_choice
        context           : what is known at the decision (dict from simulate); this rule does not use it
    Output
        The engine to send (str), or None when no available engine may go.
    Example
        opportunity_cost_policy(world, world["record"], world["runs"].loc[25], available, travel_s) -> "E28"
    """
    return opportunity_cost_choice(world, scenario, run, available_engines, travel_s, action_window_s)[1]
