"""Run the whole study: fit the environment, check the simulator against the record, and compare dispatch policies.

Run from the project root: python code/dispatch_simulator/main.py

The steps run in order and write their tables to OUTPUT_DIR:
1. The environment's fixed inputs are fitted once, and the tables that check each fit are written.
2. The 2025 calls are simulated under the protocol and compared with the record.
3. First-due coverage and fire readiness are measured for the record and for the protocol run.
4. The opportunity-cost rule is tested in three ways. A census counts the room for another engine on the protocol run.
   The rule's year is simulated and compared with the protocol under several travel seeds and action windows. Paired
   branch simulations test whether the fire readiness that a single decision saves turns into later benefit.
5. Synthetic scenarios: nominal scenarios are checked against the record of October to March. Then the nominal
   condition and each stress condition are run under the protocol and the rule, and short surges are measured in
   every first-due area.
"""
import functools

import numpy as np
import pandas as pd

import environment
import measures
import policies
import scenarios
from environment import PROJECT_ROOT, QUINT_RULE_CALL_TYPES, TRAINING_MONTHS_END, TRAVEL_SPREAD_SEED, DAY_ZERO
from measures import HOTSPOT_STATIONS
from policies import ACTION_WINDOW_S
from scenarios import SURGE_EXTRA_CALLS

OUTPUT_DIR = PROJECT_ROOT / "data" / "reports" / "dispatch_simulator"     # [DESIGN] where every table is written
ACTION_WINDOWS_S = [30, ACTION_WINDOW_S, 120]                             # [DESIGN] action windows of the rule in seconds, for a sensitivity test; the census counts the room in each window, and the rule's year is simulated with each window
POLICY_SEEDS = [TRAVEL_SPREAD_SEED, 1, 2, 3, 4]                           # [DESIGN] seeds of the travel draws, each shared by both policies; the first seed is the record's own seed
SCENARIO_SEEDS = [101, 102, 103, 104]                                     # [DESIGN] seeds of the nominal scenarios; each scenario covers a whole October to March and is checked against the record
DEMAND_FACTORS = [0.8, 1.1, 1.3, 1.5]                                     # [SOURCE] demand factors for other calls (every call except the study fires): 0.8 to 1.3 come from the study design, and 1.5 represents about ten years of growth at the 2012-22 rate
OUTAGE_ENGINE_COUNTS = [1, 2, 3]                                          # [PLACEHOLDER] the number of engines out of service each day; the records hold no outages, so these counts are a stress sweep


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. The environment's fixed inputs, fitted once
    world = environment.build_world()
    record, dispatches, engines = world["record"], world["dispatches"], world["engines"]
    for name, table in world["fits"].items():
        table.to_csv(OUTPUT_DIR / f"fit_{name}.csv")
    print(f"wrote {', '.join(f'fit_{name}.csv' for name in world['fits'])} to {OUTPUT_DIR}\n")

    # 2. The 2025 calls under the protocol, compared with the record
    protocol_report = {}
    protocol_run = environment.simulate(world, record, policies.closest_engine, protocol_report)
    origins = measures.describe_engine_origins(world, record, protocol_report["dispatches"])
    brush_dispatch = origins["run"].map(record["brush"]).astype(bool).to_numpy()
    print("travel draws, runs other than brush fires:")
    measures.describe_travel_draws(origins[~brush_dispatch], world["travel_spread"], record["inside_city"])
    print("travel draws, brush-fire runs:")
    measures.describe_travel_draws(origins[brush_dispatch], world["brush_spread"], record["inside_city"])
    measures.describe_call_fates(record["runs"], protocol_run, protocol_report["dispatches"], record["fates"])
    call_type_for_quint = dispatches["protocol"].map(QUINT_RULE_CALL_TYPES).fillna("all other calls")
    measures.describe_engine_types(dispatches, protocol_run, [("all", call_type_for_quint.notna()),
                                                              *[(label, call_type_for_quint == label) for label in world["quint_penalty_by_type"]]])
    protocol_table = measures.simulated_dispatch_table(dispatches, protocol_run, engines)
    record_curves = measures.response_curves(dispatches, verbose=False)
    protocol_curves = measures.response_curves(protocol_table)
    curve_errors = measures.compare_curves(protocol_curves, record_curves)
    states = measures.check_run_states(record["runs"], protocol_run, dispatches, protocol_report)
    strata = measures.compare_curves_by_stratum(protocol_table, dispatches)
    travel_error_min = world["fits"]["travel_model"].loc["linear on network time", "mean |error| min"]
    matrix = pd.DataFrame({
        "2025 record": measures.evaluation_column(record["runs"], record["runs"], dispatches, dispatches, dispatches,
                                                  measures.compare_curves(record_curves, record_curves, verbose=False), np.nan),
        "protocol run": measures.evaluation_column(record["runs"], protocol_run, dispatches, protocol_table, dispatches,
                                                   curve_errors, travel_error_min)})
    print(f"--- the simulator beside the record\n{matrix.round(3).to_string()}\n")
    measures.passes_record_check(states, strata)
    simulator_check = {"states": states.to_frame("value"), "strata": strata,
                       "engines": measures.compare_engines(record["runs"], protocol_run, dispatches), "matrix": matrix,
                       "fire_arrivals": measures.compare_fire_arrivals(dispatches, {"protocol run": protocol_table}),
                       "brush_quints": measures.compare_quint_calls(dispatches, {"protocol run": protocol_table})}
    protocol_table.to_parquet(OUTPUT_DIR / "simulator_check_dispatches.parquet", index=False)
    protocol_curves.to_csv(OUTPUT_DIR / "simulator_check_curves.csv", index=False)
    curve_errors.to_csv(OUTPUT_DIR / "simulator_check_curve_errors.csv", index=False)
    for name, table in simulator_check.items():
        table.to_csv(OUTPUT_DIR / f"simulator_check_{name}.csv")
    print(f"wrote simulator_check_dispatches.parquet, simulator_check_curves.csv, simulator_check_curve_errors.csv, "
          f"{', '.join(f'simulator_check_{name}.csv' for name in simulator_check)} to {OUTPUT_DIR}\n")

    # 3. First-due coverage and fire readiness, for the record and the protocol run
    first_due_record = measures.check_first_due(dispatches, record["runs"], record["travel"], engines)
    first_due_protocol = measures.check_first_due(dispatches, protocol_run, record["travel"], engines)
    first_due_summary, first_due_by_station = measures.summarize_first_due({"2025 record": first_due_record, "protocol run": first_due_protocol})
    readiness = {name: measures.fire_readiness(world, runs_table) for name, runs_table in [("2025 record", record["runs"]), ("protocol run", protocol_run)]}
    coverage_tables = {
        "first_due_calls": pd.concat({"2025 record": first_due_record, "protocol run": first_due_protocol}, names=["version"]),
        "first_due_summary": first_due_summary, "first_due_by_station": first_due_by_station,
        "readiness_summary": pd.concat({name: result[2] for name, result in readiness.items()}, axis=1),
        "readiness_cells": pd.concat({name: result[1] for name, result in readiness.items()}, names=["version"])}
    for name, table in coverage_tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv")
    pd.concat({name: result[0] for name, result in readiness.items()}, names=["version"]).to_parquet(OUTPUT_DIR / "readiness_segments.parquet")
    print(f"wrote {', '.join(f'{name}.csv' for name in coverage_tables)}, readiness_segments.parquet to {OUTPUT_DIR}\n")

    # 4. The opportunity-cost rule: the room for another engine, a simulated year against the protocol, and paired branch simulations
    print(f"low-acuity runs (SOC low-priority medical and service calls): {int(record['low_acuity'].sum()):,} of {len(dispatches):,}\n")
    census = measures.census_decisions(world, record, protocol_run, ACTION_WINDOWS_S)
    comparisons = {}
    for seed in POLICY_SEEDS:
        quantile = record["spread_quantile"] if seed == TRAVEL_SPREAD_SEED else pd.Series(np.random.default_rng(seed).random(len(record["runs"])),
                                                                                           index=record["runs"].index)
        seeded = {**record, "spread_quantile": quantile}
        seed_protocol_run = protocol_run if seed == TRAVEL_SPREAD_SEED else environment.simulate(world, seeded, policies.closest_engine, verbose=False)
        seed_protocol_table = measures.simulated_dispatch_table(dispatches, seed_protocol_run, engines, verbose=False)
        seed_first_due = measures.check_first_due(dispatches, seed_protocol_run, record["travel"], engines, verbose=False)
        seed_readiness = measures.fire_readiness(world, seed_protocol_run, verbose=False)[0]
        for window_s in (ACTION_WINDOWS_S if seed == TRAVEL_SPREAD_SEED else [ACTION_WINDOW_S]):
            rule_report = {}
            rule_run = environment.simulate(world, seeded, functools.partial(policies.opportunity_cost_policy, action_window_s=window_s),
                                            rule_report, verbose=False)
            if seed == TRAVEL_SPREAD_SEED and window_s == ACTION_WINDOW_S:
                print(f"state checks of the opportunity-cost rule's year (the record's own travel seed, {ACTION_WINDOW_S} s window):")
                rule_states = measures.check_run_states(record["runs"], rule_run, dispatches, rule_report)
            rule_readiness = measures.fire_readiness(world, rule_run, verbose=False)[0]
            print(f"seed {seed}, action window {window_s} s:")
            comparisons[(seed, window_s)] = measures.compare_policies(
                world, record, seed_protocol_table, measures.simulated_dispatch_table(dispatches, rule_run, engines, verbose=False),
                seed_first_due, seed_readiness, rule_readiness)
    policy_comparison = pd.concat(comparisons, names=["seed", "action_window_s", "outcome"])
    over_seeds = measures.summarize_over_seeds(policy_comparison)
    branches, branch_cells = measures.branch_decisions(world, record, protocol_run, census)
    rule_tables = {"census": census, "policy_comparison": policy_comparison, "over_seeds": over_seeds,
                   "rule_states": rule_states.to_frame("value"), "branches": branches, "branch_cells": branch_cells}
    for name, table in rule_tables.items():
        table.to_csv(OUTPUT_DIR / f"oc_{name}.csv")
    print(f"wrote {', '.join(f'oc_{name}.csv' for name in rule_tables)} to {OUTPUT_DIR}\n")

    # 5. Synthetic scenarios: nominal scenarios checked against the record of October to March, then the conditions
    generator = scenarios.fit_generator(world)
    pool, locations = generator["pool"], generator["locations"]
    training_days = range((TRAINING_MONTHS_END - DAY_ZERO).days)
    busiest_cells = pool[~pool["fire"]].groupby(["cell_x", "cell_y"]).size().nlargest(20).index
    no_outages = pd.DataFrame(columns=["engine_id", "start_s", "end_s"])

    # 5a. Nominal scenarios under the protocol, checked against the record; each scenario covers a whole October to March
    record_segments = measures.fire_readiness(world, protocol_run, wait_for_busy=True, verbose=False)[0]
    record_parts = measures.weekly_calibration_parts(dispatches, protocol_run, first_due_protocol, census, record_segments, busiest_cells,
                                                     training_days)
    nominal_scenarios, scenario_parts, scenario_checks = {}, {}, {}
    for scenario_seed in SCENARIO_SEEDS:
        first_seed = scenario_seed == SCENARIO_SEEDS[0]
        scenario_calls, scenario_day_factors = scenarios.generate_calls(pool, generator["counts"], generator["dispersion"], training_days,
                                                                        scenario_seed, verbose=first_seed)
        nominal_scenario = scenarios.build_scenario(world, generator, scenario_calls, no_outages, scenario_seed, verbose=first_seed)
        nominal_report = {}
        nominal_run = environment.simulate(world, nominal_scenario, policies.closest_engine, nominal_report, verbose=False)
        print(f"nominal scenario, seed {scenario_seed}: protocol run simulated")
        scenario_checks[(f"nominal, seed {scenario_seed}", "protocol")] = measures.check_run_states(
            nominal_scenario["runs"], nominal_run, nominal_scenario["dispatches"], nominal_report, verbose=first_seed)
        nominal_first_due, nominal_census, nominal_segments = measures.measure_run(world, nominal_scenario, nominal_run)
        scenario_parts[scenario_seed] = measures.weekly_calibration_parts(nominal_scenario["dispatches"], nominal_run, nominal_first_due,
                                                                          nominal_census, nominal_segments, busiest_cells, training_days)
        nominal_scenarios[scenario_seed] = (scenario_calls, scenario_day_factors, nominal_scenario, nominal_run, nominal_first_due,
                                            nominal_census, nominal_segments)
    calibration = measures.compare_scenarios_with_record(record_parts, scenario_parts)

    # 5b. Each condition on the first seed's calls: what the condition changes, the room for another engine, and the
    # rule against the protocol
    base_seed = SCENARIO_SEEDS[0]
    base_calls, base_day_factors = nominal_scenarios[base_seed][:2]
    surge_sets = {count: scenarios.add_short_surges(base_calls, pool, locations, training_days, count, base_seed, verbose=count == SURGE_EXTRA_CALLS[0])
                  for count in SURGE_EXTRA_CALLS}
    surge_condition_counts = {f"short surges, {count} extra calls": count for count in SURGE_EXTRA_CALLS}
    conditions = {"nominal": (base_calls, no_outages),
                  **{f"other calls x{factor}": (scenarios.add_demand_growth(base_calls, base_day_factors, pool, generator["counts"], locations,
                                                                            factor, base_seed), no_outages) for factor in DEMAND_FACTORS},
                  **{name: (surge_sets[count][0], no_outages) for name, count in surge_condition_counts.items()},
                  **{f"{count} engines out each day": (base_calls, scenarios.add_outages(engines, training_days, count, base_seed))
                     for count in OUTAGE_ENGINE_COUNTS},
                  **{f"station {station} closed": (base_calls, scenarios.close_station(engines, station, training_days)) for station in HOTSPOT_STATIONS},
                  "a large fire each week": (scenarios.add_large_fires(base_calls, pool, training_days, base_seed), no_outages)}
    training_end_s = len(training_days) * 86400
    condition_rooms, condition_comparisons, surge_segments = {}, {}, {}
    for condition_name, (condition_calls, condition_outages) in conditions.items():
        print(f"=== condition: {condition_name}")
        if condition_name == "nominal":
            condition_scenario, condition_protocol_run, condition_first_due, condition_census, condition_segments = nominal_scenarios[base_seed][2:]
        else:
            condition_scenario = scenarios.build_scenario(
                world, generator, condition_calls, condition_outages, base_seed,
                verbose=condition_name in (f"{OUTAGE_ENGINE_COUNTS[0]} engines out each day", "a large fire each week"))
            condition_report = {}
            condition_protocol_run = environment.simulate(world, condition_scenario, policies.closest_engine, condition_report, verbose=False)
            scenario_checks[(condition_name, "protocol")] = measures.check_run_states(
                condition_scenario["runs"], condition_protocol_run, condition_scenario["dispatches"], condition_report, verbose=False)
            condition_first_due, condition_census, condition_segments = measures.measure_run(world, condition_scenario, condition_protocol_run)
        condition_effects, _ = measures.check_scenario_conditions(condition_scenario, condition_protocol_run,
                                                                  verbose=len(condition_outages) > 0 or bool(condition_scenario["large_fire"].any()))
        scenario_checks[(condition_name, "protocol")] = pd.concat([scenario_checks[(condition_name, "protocol")] if condition_name != "nominal"
                                                                   else scenario_checks[(f"nominal, seed {base_seed}", "protocol")], condition_effects])
        condition_rule_report = {}
        condition_rule_run = environment.simulate(world, condition_scenario, policies.opportunity_cost_policy, condition_rule_report, verbose=False)
        scenario_checks[(condition_name, "opportunity-cost rule")] = measures.check_run_states(
            condition_scenario["runs"], condition_rule_run, condition_scenario["dispatches"], condition_rule_report, verbose=False)
        condition_rule_segments = measures.fire_readiness(world, condition_rule_run, wait_for_busy=True, verbose=False)[0]
        if condition_name in surge_condition_counts:
            surge_segments[surge_condition_counts[condition_name]] = (condition_segments, condition_rule_segments)
        condition_rooms[condition_name] = measures.describe_room(condition_scenario["dispatches"], condition_census, condition_first_due,
                                                                 condition_segments, training_days)
        condition_comparisons[condition_name] = measures.compare_policies(
            world, condition_scenario,
            measures.simulated_dispatch_table(condition_scenario["dispatches"], condition_protocol_run, engines, verbose=False),
            measures.simulated_dispatch_table(condition_scenario["dispatches"], condition_rule_run, engines, verbose=False), condition_first_due,
            condition_segments[condition_segments["start_s"] < training_end_s], condition_rule_segments[condition_rule_segments["start_s"] < training_end_s],
            verbose=False).drop(index="fire calls, Jul-Sep")

    # 5c. Short surges in every first-due area, as a stress sweep. For each of the four seeds, the loop measures how much
    # fire readiness a surge costs while it lasts, against the same hours without the surge. For the first seed only, it
    # also measures how the rule changes fire readiness during the surges.
    surge_tables = []
    for count in SURGE_EXTRA_CALLS:
        for scenario_seed in SCENARIO_SEEDS:
            if scenario_seed == base_seed:
                seed_surges = surge_sets[count][1]
                seed_surge_segments, seed_rule_segments = surge_segments[count]
            else:
                seed_surged_calls, seed_surges = scenarios.add_short_surges(nominal_scenarios[scenario_seed][0], pool, locations, training_days,
                                                                            count, scenario_seed, verbose=False)
                seed_scenario = scenarios.build_scenario(world, generator, seed_surged_calls, no_outages, scenario_seed, verbose=False)
                seed_report = {}
                seed_run = environment.simulate(world, seed_scenario, policies.closest_engine, seed_report, verbose=False)
                scenario_checks[(f"short surges, {count} extra calls, seed {scenario_seed}", "protocol")] = measures.check_run_states(
                    seed_scenario["runs"], seed_run, seed_scenario["dispatches"], seed_report, verbose=False)
                seed_surge_segments = measures.fire_readiness(world, seed_run, wait_for_busy=True, verbose=False)[0]
                seed_rule_segments = None
            loss_with_surge = measures.readiness_loss_in_windows(seed_surge_segments, seed_surges["start_s"], seed_surges["end_s"])
            surge_tables.append(seed_surges.assign(
                seed=scenario_seed,
                readiness_loss_without_surge_s=measures.readiness_loss_in_windows(nominal_scenarios[scenario_seed][6], seed_surges["start_s"],
                                                                                  seed_surges["end_s"]),
                readiness_loss_with_surge_s=loss_with_surge,
                rule_minus_protocol_s=np.nan if seed_rule_segments is None else
                measures.readiness_loss_in_windows(seed_rule_segments, seed_surges["start_s"], seed_surges["end_s"]) - loss_with_surge))
            print(f"short surges, {count} extra calls, seed {scenario_seed}: readiness during the surges measured")
    scenario_surges = pd.concat(surge_tables, ignore_index=True)
    scenario_surges["readiness_cost_s"] = scenario_surges["readiness_loss_with_surge_s"] - scenario_surges["readiness_loss_without_surge_s"]
    scenario_surge_areas = measures.summarize_surges(scenario_surges)
    room = pd.DataFrame(condition_rooms)
    scenario_policy_comparison = pd.concat(condition_comparisons, names=["condition", "outcome"])
    print(f"--- the room for another engine under each condition (seed {base_seed}, protocol runs, October-March)\n{room.round(3).to_string()}\n")
    measures.summarize_rule_by_condition(condition_comparisons)
    checks_table = pd.DataFrame(scenario_checks)
    violations = checks_table.loc[["runs never served", "runs sent while their engine was still on an earlier run",
                                   "calls whose number of arrivals differs from 2025", "runs with times out of order",
                                   "runs served without a clear"]].sum()
    print(f"state violations over all scenario runs: {int(violations.sum())}; real runs sent to an engine out of service: "
          f"{int(checks_table.loc['real runs sent to an engine out of service'].sum())}\n")
    scenario_tables = {"calibration": calibration, "room": room, "policy_comparison": scenario_policy_comparison,
                       "checks": checks_table, "surges": scenario_surges, "surge_areas": scenario_surge_areas}
    for name, table in scenario_tables.items():
        table.to_csv(OUTPUT_DIR / f"scenario_{name}.csv")
    print(f"wrote {', '.join(f'scenario_{name}.csv' for name in scenario_tables)} to {OUTPUT_DIR}\n")
