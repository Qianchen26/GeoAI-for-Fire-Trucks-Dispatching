"""Measures of how a simulated run compares with the record, how well it protects fire calls, and how two policies differ.

The functions fall into five groups, in this order:
- the simulated dispatch table and its response curves, and the check of the simulator against the 2025 record;
- printed diagnostics of what the simulator did on concrete runs;
- weekly block bootstrap intervals and the classes of a policy difference;
- first-due coverage, fire readiness, the room for another engine on low-acuity calls, and policy comparisons;
- measures of synthetic scenarios: their calibration against the record, their room, and short surges.
"""
import numpy as np
import pandas as pd
import pyproj
from sklearn.metrics import root_mean_squared_error

from environment import (AREA_LABELS, DAY_ZERO, HIGH_ACUITY_LEVELS, OUT_OF_SERVICE_PROTOCOL, PROJECTED_CRS, READINESS_CELL_M,
                         RETURN_CHECK_SECONDS, STUDY_FIRE_PROTOCOLS, TIME_COLUMNS, TRAVEL_TEST_FROM, VERBOSE, YEAR_END,
                         BRUSH_FIRE_PROTOCOL, call_fates, share_of_way_home, show_table, simulate)
from policies import ACTION_WINDOW_S, closest_engine, opportunity_cost_choice, rule_pick
from scenarios import LARGE_FIRE_ENGINES

RESPONSE_GRID_MINUTES = np.round(np.arange(0, 20.0001, 0.1), 1)             # [SOURCE] grid of the record's response curves, in minutes
COMPLETION_GRID_MINUTES = np.arange(0, 241, 1.0)                            # [SOURCE] grid of the record's completion curves, in minutes
PRIORITY_LABELS = ["Low", "Moderate", "High", "(missing)"]                  # [SOURCE] the record's priority groups
CURVE_RMSE_LIMIT = 0.03                                                     # [DESIGN] the simulator passes its check against the record when the first-on-scene curve RMSE is at most this
FIRE_MARGIN_S = 10                                                          # [SOURCE] m: the smallest difference in fire arrival that matters in practice, in seconds (tentative)
MEDICAL_MARGIN_S = 10                                                       # [SOURCE] delta: the allowed rise in the P90 first arrival at high-acuity medical calls, in seconds (provisional)
BLOCK_BOOTSTRAP_DRAWS = 2000                                                # [DESIGN] weekly block bootstrap draws for an interval
BLOCK_BOOTSTRAP_SEED = 20250701                                             # [DESIGN] seed of the weekly block bootstrap
BOOTSTRAP_INTERVAL = (0.025, 0.975)                                         # [STANDARD] two-sided 95% interval
QUINT_SHARE_TOLERANCE = 0.05                                                # [DESIGN] the simulated share of multi-engine brush fires that got a quint should be within 5 percentage points of the record's share (the sampling standard error is about 2 points on 179 calls)
READINESS_CHUNK = 2000                                                      # [DESIGN] time segments per vectorized chunk of the readiness computation
HOTSPOT_STATIONS = ["GFR_03", "CO_21", "CO_16"]                             # [SOURCE] the first-due stations that together account for 48% of the fire calls whose first-due station had no free engine (subgroup fixed in advance)
DAYTIME_HOURS = range(10, 21)                                               # [SOURCE] hours whose fire readiness loss exceeds the yearly mean (subgroup fixed in advance)
BRANCH_DECISIONS = 300                                                      # [DESIGN] sampled decisions for paired branches
BRANCH_STRATA = 5                                                           # [DESIGN] equal-count strata of the opportunity cost saved, sampled evenly
BRANCH_HORIZON_S = 7200                                                     # [DESIGN] how long a branch runs after the decision, in seconds; about one engine outing and its return
BRANCH_WARMUP_S = 24 * 3600                                                 # [DESIGN] the time before the decision whose calls are simulated to rebuild the engines' state, in seconds
BRANCH_SEED = 20250101                                                      # [DESIGN] seed of the decision sample


# ----------------------------------------------------------------------------------------------------------------------
# The simulated dispatch table, its curves, and the check against the record
# ----------------------------------------------------------------------------------------------------------------------

def seconds_between(later, earlier):
    """Count the whole seconds from one time column to another; the result is missing where either time is missing.

    Input
        later, earlier : two time columns of equal length (Series of Timestamps)
    Output
        The seconds between them, rounded to whole seconds (Series, nullable int).
    Example
        seconds_between(pd.Series(pd.to_datetime(["2024-10-01 01:54:18"])),
                        pd.Series(pd.to_datetime(["2024-10-01 01:53:16"]))).tolist() -> [62]
    """
    return (later - earlier).dt.total_seconds().round().astype("Int64")


def simulated_dispatch_table(dispatches, simulated, engines=None, verbose=VERBOSE):
    """Rebuild the record's 32-column dispatch table from a simulated run.

    The CAD times come from the simulation. The durations, the cancelled flag and the arrival order follow from them
    with the record's own formulas, and the engines per call count the engines sent. Call attributes are inputs and are carried over. Where the
    simulation sent another engine than the record, the engine columns describe that engine as sent from its station,
    and auto_aid is unknown.

    Input
        dispatches : the table the run simulates (DataFrame, the record or a scenario's table)
        simulated  : the simulated run from simulate (DataFrame, same rows as dispatches)
        engines    : the engines (DataFrame), needed when the run sent other engines than the table
        verbose    : whether to print the table and the columns that differ from the input (bool)
    Output
        The simulated dispatch table (DataFrame, same rows and columns as dispatches).
    Example
        simulated_dispatch_table(dispatches, protocol_run, engines).shape -> (37194, 32)
    """
    table = dispatches.copy()
    for column in TIME_COLUMNS:
        seconds = simulated[column]                              # keeps a missing time missing (NaT) and avoids a pandas cast warning
        table[column] = (DAY_ZERO + pd.to_timedelta(seconds.fillna(0), unit="s").where(seconds.notna())).astype(dispatches[column].dtype)
    no_engine = simulated["engine_id"].isna()
    if no_engine.any():                                          # a request released before any engine was free
        for column in ["engine_id", "engine_type", "engine_agency", "engine_lon_at_dispatch", "engine_lat_at_dispatch",
                       "from_station", "at_home", "auto_aid"]:
            table.loc[no_engine, column] = None
    other_engine = (simulated["engine_id"].notna() & (simulated["engine_id"] != dispatches["engine_id"])).to_numpy()
    if other_engine.any():
        sent = engines.set_index("engine_id").loc[simulated.loc[other_engine, "engine_id"]]
        table.loc[other_engine, "engine_id"] = sent.index.to_numpy()
        for column, source in [("engine_type", "engine_type"), ("engine_agency", "engine_agency"),
                               ("engine_lon_at_dispatch", "home_lon"), ("engine_lat_at_dispatch", "home_lat"),
                               ("from_station", "home_station")]:
            table.loc[other_engine, column] = sent[source].to_numpy()
        table.loc[other_engine, "at_home"] = True
        table.loc[other_engine, "auto_aid"] = pd.NA
    table["turnout_s"] = seconds_between(table["enroute_at"], table["dispatched_at"])
    table["travel_s"] = seconds_between(table["arrived_at"], table["enroute_at"])
    table["on_scene_s"] = seconds_between(table["cleared_at"], table["arrived_at"])
    table["cancelled"] = table["arrived_at"].isna().astype("boolean")
    table["arrival_order"] = table.groupby("incident_id")["arrived_at"].rank(method="first").astype("Int64")
    table["n_engines_sent"] = table.groupby("incident_id")["engine_id"].transform("count").astype("Int64")  # counts the engines sent, not the requests
    show_table("simulated dispatch table", table, verbose)
    if verbose:
        differing = [column for column in dispatches.columns if not table[column].equals(dispatches[column])]
        print(f"columns that differ from the input table: {differing}; whole table equal: {table.equals(dispatches)}\n")
    return table


def response_curves(table, verbose=VERBOSE):
    """Compute the share of calls that reached each stage within t minutes of the call, for every curve the record draws.

    The denominator is every call in the group, so a group whose calls were not all reached stays below 100%. The
    curves are: dispatched, en route and first on scene (0 to 20 minutes); first on scene and last cleared (0 to 240
    minutes); first on scene inside and outside the city; and first on scene by priority.

    Input
        table   : a dispatch table (DataFrame, the record or simulated_dispatch_table's output)
        verbose : whether to print the first rows (bool)
    Output
        The curves (DataFrame, one row per curve, series and minute): curve, series, minute, share, incidents.
    Example
        response_curves(dispatches).query("curve == 'stages' and series == 'first_on_scene' and minute == 20.0")["share"]
        -> about 0.856
    """
    per_call = table.groupby("incident_id").agg(
        call=("call_received_at", "min"), dispatched=("dispatched_at", "min"), en_route=("enroute_at", "min"),
        first_on_scene=("arrived_at", "min"), last_cleared=("cleared_at", "max"),
        focal=("focal", "first"), priority=("priority", "first"))
    minutes = {stage: (per_call[stage] - per_call["call"]).dt.total_seconds() / 60
               for stage in ["dispatched", "en_route", "first_on_scene", "last_cleared"]}
    area = per_call["focal"].map(AREA_LABELS)
    priority = per_call["priority"].astype("string").fillna("(missing)")
    specs = [("stages", stage, minutes[stage], RESPONSE_GRID_MINUTES) for stage in ["dispatched", "en_route", "first_on_scene"]]
    specs += [("completion", stage, minutes[stage], COMPLETION_GRID_MINUTES) for stage in ["first_on_scene", "last_cleared"]]
    specs += [("first_on_scene_by_area", label, minutes["first_on_scene"][area == label], RESPONSE_GRID_MINUTES)
              for label in AREA_LABELS.values()]
    specs += [("first_on_scene_by_priority", label, minutes["first_on_scene"][priority == label], RESPONSE_GRID_MINUTES)
              for label in PRIORITY_LABELS]
    pieces = []
    for curve, series, values, grid in specs:
        reached = np.sort(values.dropna().to_numpy())
        pieces.append(pd.DataFrame({"curve": curve, "series": series, "minute": grid,
                                    "share": np.searchsorted(reached, grid, side="right") / len(values),
                                    "incidents": len(values)}))
    curves = pd.concat(pieces, ignore_index=True)
    show_table("simulated response curves", curves, verbose)
    return curves


def compare_curves(simulated_curves, observed_curves, verbose=VERBOSE):
    """Compare two sets of response curves: for each curve, compute the RMSE and the largest absolute difference in share.

    Input
        simulated_curves : curves from response_curves for a simulated run (DataFrame)
        observed_curves  : the record's curves from response_curves (DataFrame)
        verbose          : whether to print the comparison (bool)
    Output
        One row per curve (DataFrame): curve, series, grid points, points found in both, RMSE, largest absolute
        difference, and the call counts of both.
    Example
        compare_curves(curves, curves)[["rmse", "max_abs_difference"]].max().tolist() -> [0.0, 0.0]
    """
    both = simulated_curves.merge(observed_curves, on=["curve", "series", "minute"], how="outer",
                                  suffixes=("_simulated", "_2025"), indicator=True)
    both["abs_difference"] = (both["share_simulated"] - both["share_2025"]).abs()
    by_curve = both.groupby(["curve", "series"], sort=False)
    check = by_curve.agg(
        points=("minute", "size"), points_in_both=("_merge", lambda merge: int((merge == "both").sum())),
        max_abs_difference=("abs_difference", "max"),
        calls_simulated=("incidents_simulated", "first"), calls_2025=("incidents_2025", "first")).reset_index()
    check.insert(4, "rmse", [root_mean_squared_error(part["share_2025"], part["share_simulated"]) for _, part in by_curve])
    if verbose:
        print(f"--- simulated vs 2025 curves {check.shape}\n{check.to_string()}\n")
    return check


def compare_curves_by_stratum(table, dispatches, verbose=VERBOSE):
    """Compare a simulated first-on-scene curve with the record's in four groups of calls that are defined from the record.

    The groups are: all calls, multi-engine calls, calls whose first engine was sent away from its station, and calls
    with an engine from the other agency. Both curves use the same calls.

    Input
        table      : a simulated dispatch table (DataFrame)
        dispatches : the record (DataFrame)
        verbose    : whether to print the comparison (bool)
    Output
        One row per group (DataFrame): calls, RMSE and largest absolute difference of the first-on-scene curve.
    Example
        compare_curves_by_stratum(protocol_table, dispatches).loc["all calls", "rmse"] -> about 0.0103
    """
    first_sent = dispatches.sort_values("dispatched_at").groupby("incident_id").head(1)
    other_agency = (dispatches["engine_agency"] == "GFR") != (dispatches["jurisdiction"] == "GFR")
    strata = {"all calls": dispatches["incident_id"].unique(),
              "multi-engine calls": dispatches.loc[dispatches["n_engines_sent"] > 1, "incident_id"].unique(),
              "first engine sent away from home": first_sent.loc[~first_sent["at_home"].astype(bool), "incident_id"].unique(),
              "calls with an other-agency engine": dispatches.loc[other_agency, "incident_id"].unique()}
    rows = {}
    for name, incidents in strata.items():
        check = compare_curves(response_curves(table[table["incident_id"].isin(incidents)], verbose=False),
                               response_curves(dispatches[dispatches["incident_id"].isin(incidents)], verbose=False),
                               verbose=False).set_index(["curve", "series"]).loc[("stages", "first_on_scene")]
        rows[name] = {"calls": len(incidents), "rmse": check["rmse"], "max_abs_difference": check["max_abs_difference"]}
    by_stratum = pd.DataFrame(rows).T
    if verbose:
        print(f"--- first-on-scene curve, simulated vs 2025, by stratum\n{by_stratum.round(4).to_string()}\n")
    return by_stratum


def compare_engines(runs, simulated, dispatches, verbose=VERBOSE):
    """Count how often the simulation sent the same engine as the record, in four groups of runs.

    The groups are: all runs, the first engine of each call, and runs inside and outside the city. The function also
    prints how the travel changed on the runs where the simulation sent another engine.

    Input
        runs       : the record's runs (DataFrame with engine_id and dispatched_at)
        simulated  : the simulated run (DataFrame with engine_id and times)
        dispatches : the record (DataFrame with focal)
        verbose    : whether to print the comparison (bool)
    Output
        One row per group (DataFrame): runs and the share that got the record's engine.
    Example
        compare_engines(runs, protocol_run, dispatches).loc["all runs", "same_engine_share"] -> share of runs sent the record's engine
    """
    same_engine = simulated["engine_id"] == runs["engine_id"]
    first_engine = runs.sort_values("dispatched_at").groupby("incident_id").head(1).index
    area = dispatches["focal"].map(AREA_LABELS)
    groups = {"all runs": runs.index, "first engine of each call": first_engine,
              **{label: runs.index[area == label] for label in AREA_LABELS.values()}}
    summary = pd.DataFrame({"runs": {name: len(rows) for name, rows in groups.items()},
                            "same_engine_share": {name: same_engine[rows].mean() for name, rows in groups.items()}})
    travel_change_min = ((simulated.arrived_at - simulated.enroute_at) - (runs.arrived_at - runs.enroute_at)) / 60
    by_engine = simulated.sort_values(["engine_id", "dispatched_at"])
    still_on_earlier_run = int((by_engine["dispatched_at"] < by_engine.groupby("engine_id")["cleared_at"].shift()).sum())
    if verbose:
        changed = travel_change_min[~same_engine].dropna()
        print(f"--- engines: simulation vs 2025\n{summary.round(3).to_string()}\n"
              f"runs with another engine and an arrival: {len(changed):,}; travel change (min): median {changed.median():.2f}, "
              f"P10 {changed.quantile(0.1):.2f}, P90 {changed.quantile(0.9):.2f}\n"
              f"runs whose engine was still on an earlier run: {still_on_earlier_run}\n")
    return summary


def compare_first_on_scene(tables, verbose=VERBOSE):
    """Compare the median and P90 minutes from the call to the first engine on scene across several tables.

    Only calls that an engine reached count. The minutes are given for all calls and for calls inside and outside the
    city. The column "difference" gives the last table's value minus the first table's.

    Input
        tables  : the dispatch tables to compare, keyed by name (dict of DataFrames)
        verbose : whether to print the comparison (bool)
    Output
        One row per statistic and one column per table, plus "difference" (DataFrame, minutes).
    Example
        compare_first_on_scene({"2025": dispatches, "rule": protocol_table}).loc["all calls P90", "2025"] -> about 13.22
    """
    columns = {}
    for name, table in tables.items():
        per_call = table.groupby("incident_id").agg(call=("call_received_at", "min"), first_on_scene=("arrived_at", "min"),
                                                     focal=("focal", "first"))
        minutes = (per_call["first_on_scene"] - per_call["call"]).dt.total_seconds() / 60
        area = per_call["focal"].map(AREA_LABELS)
        groups = {"all calls": minutes, **{label: minutes[area == label] for label in AREA_LABELS.values()}}
        columns[name] = pd.Series({**{f"{group} median": values.median() for group, values in groups.items()},
                                   **{f"{group} P90": values.quantile(0.9) for group, values in groups.items()}})
    summary = pd.DataFrame(columns)
    summary["difference"] = summary.iloc[:, -1] - summary.iloc[:, 0]
    if verbose:
        print(f"--- minutes from call to first engine on scene\n{summary.round(2).to_string()}\n")
    return summary


def evaluation_column(runs, simulated, dispatches, table, control, check, travel_test_error_min):
    """Score one run of the simulator on fixed rows, so that the record and the simulator sit side by side.

    The rows are the travel model's test error, the shares of runs sent the record's engine in several groups, the
    curve RMSEs, and the first-on-scene minutes, including the run's P90 minus the control's P90. For the record
    itself, pass the record as simulated, table and control.

    Input
        runs                  : the record's runs (DataFrame)
        simulated             : the run to score (DataFrame with engine_id)
        dispatches            : the record (DataFrame)
        table                 : the run's dispatch table (DataFrame)
        control               : the dispatch table whose P90 is subtracted from the run's P90 (DataFrame); in practice
                                always the record
        check                 : the run's curve comparison from compare_curves (DataFrame)
        travel_test_error_min : the travel model's mean absolute test error, in minutes (float; NaN for the record)
    Output
        One value per row (Series, 15 rows).
    Example
        evaluation_column(runs, protocol_run, dispatches, protocol_table, dispatches, curve_errors, 1.6)["same engine: all runs"]
        -> share of runs sent the record's engine
    """
    same_engine = simulated["engine_id"] == runs["engine_id"]
    first_engine = runs.sort_values("dispatched_at").groupby("incident_id").head(1).index
    at_home = dispatches["at_home"].astype(bool)
    multi_engine = dispatches["n_engines_sent"] > 1
    other_agency = (dispatches["engine_agency"] == "GFR") != (dispatches["jurisdiction"] == "GFR")
    timing = compare_first_on_scene({"rule": table, "control": control}, verbose=False)
    rmse = check.set_index(["curve", "series"])["rmse"]
    return pd.Series({
        "travel estimate: test mean |error| (min)": travel_test_error_min,
        "same engine: all runs": same_engine.mean(),
        "same engine: first engine of each call": same_engine[first_engine].mean(),
        "same engine: single-engine calls": same_engine[~multi_engine].mean(),
        "same engine: multi-engine calls": same_engine[multi_engine].mean(),
        "same engine: 2025 engine at home": same_engine[at_home].mean(),
        "same engine: 2025 engine away from home": same_engine[~at_home].mean(),
        "same engine: 2025 engine from the other agency": same_engine[other_agency].mean(),
        "curve RMSE: first on scene, all calls": rmse[("stages", "first_on_scene")],
        "curve RMSE: first on scene, inside city": rmse[("first_on_scene_by_area", "inside city")],
        "curve RMSE: first on scene, outside city": rmse[("first_on_scene_by_area", "outside city")],
        "curve RMSE: largest of the 11 curves": rmse.max(),
        "first on scene: median (min)": timing.loc["all calls median", "rule"],
        "first on scene: P90 (min)": timing.loc["all calls P90", "rule"],
        "first on scene: P90 rule minus control (min)": timing.loc["all calls P90", "rule"] - timing.loc["all calls P90", "control"],
    })


def check_run_states(runs, simulated, dispatches, report, verbose=VERBOSE):
    """Check that a simulated run never breaks the simulator's rules, and describe its waits.

    Five checks must be 0: runs never served, runs sent while their engine was still on an earlier run, calls whose
    number of arrivals differs from the record's, runs with times out of order, and runs served without a clear time.
    The number of runs that kept their recorded times must not exceed the number of runs of calls without a location.
    The other rows describe the run.

    Input
        runs       : the runs the simulation started from (DataFrame; the record's runs or a scenario's runs)
        simulated  : the simulated run (DataFrame)
        dispatches : the matching dispatch table (DataFrame with incident_lon and priority)
        report     : the report dict that simulate filled
        verbose    : whether to print the checks and the waits by priority (bool)
    Output
        One value per check (Series, 12 rows).
    Example
        check_run_states(runs, protocol_run, dispatches, protocol_report)["runs never served"] -> 0
    """
    served = simulated["engine_id"].notna()
    order = simulated[served].sort_values(["engine_id", "dispatched_at"])
    still_busy = order["dispatched_at"] < order.groupby("engine_id")["cleared_at"].shift()
    out_of_order = (simulated[TIME_COLUMNS].ffill(axis=1).diff(axis=1).iloc[:, 1:] < 0).any(axis=1)
    waits = pd.Series(report["waited_seconds"], dtype=float)
    arrivals = [frame["arrived_at"].notna().groupby(runs["incident_id"]).sum() for frame in (runs, simulated)]
    first_arrival = [frame["arrived_at"].groupby(runs["incident_id"]).min() for frame in (simulated, runs)]
    checks = pd.Series({
        "runs never served": len(report["never_served"]),
        "runs sent while their engine was still on an earlier run": int(still_busy.sum()),
        "calls whose number of arrivals differs from 2025": int((arrivals[0] != arrivals[1]).sum()),
        "runs with times out of order": int(out_of_order.sum()),
        "runs served without a clear": int((served & simulated["cleared_at"].isna()).sum()),
        "runs that kept their 2025 times": len(report["kept_recorded"]),
        "runs of calls without a location": int(dispatches["incident_lon"].isna().sum()),
        "cancelled requests that never got an engine": len(report["cancelled_while_waiting"]),
        "calls whose first arrival is the recorded second (share)": float((first_arrival[0] == first_arrival[1]).mean()),
        "runs that waited for an engine": len(waits),
        "wait seconds: median": waits.median(),
        "wait seconds: max": waits.max(),
    })
    if verbose:
        print(f"--- state checks\n{checks.round(4).to_string()}\n")
        if len(waits):
            ranks = dispatches.loc[waits.index, "priority"].astype("string").fillna("(missing)")
            print(f"waits by priority (count, median s):\n{waits.groupby(ranks).agg(['size', 'median']).round(0).to_string()}\n")
    return checks


def passes_record_check(states, by_stratum, verbose=VERBOSE):
    """Decide whether the simulator passes its check against the record.

    It passes when the first-on-scene curve RMSE over all calls is at most CURVE_RMSE_LIMIT, no state check is violated,
    and no more runs kept their recorded times than there are runs of calls without a location.

    Input
        states     : the state checks from check_run_states (Series)
        by_stratum : the curve comparison from compare_curves_by_stratum (DataFrame)
        verbose    : whether to print the decision (bool)
    Output
        Whether the simulator passes (bool).
    Example
        passes_record_check(states, strata) -> True      # RMSE about 0.0103, no violation
    """
    violation_checks = ["runs never served", "runs sent while their engine was still on an earlier run",
                        "calls whose number of arrivals differs from 2025", "runs with times out of order",
                        "runs served without a clear"]
    violations = states[violation_checks].sum()
    passed = bool(by_stratum.loc["all calls", "rmse"] <= CURVE_RMSE_LIMIT and violations == 0
                  and states["runs that kept their 2025 times"] <= states["runs of calls without a location"])
    if verbose:
        print(f"record check: RMSE all calls {by_stratum.loc['all calls', 'rmse']:.4f} (limit {CURVE_RMSE_LIMIT}), state violations "
              f"{int(violations)}, kept 2025 times {int(states['runs that kept their 2025 times'])} "
              f"(calls without a location {int(states['runs of calls without a location'])}) -> passed: {passed}\n")
    return passed


def compare_fire_arrivals(dispatches, tables, verbose=VERBOSE):
    """Compare the first-engine arrival at the study's fire calls with the record; policies are judged on this outcome.

    The function compares each simulated table for each fire code and for all four codes together, over the whole year
    and over the held-out months (from TRAVEL_TEST_FROM). Each comparison reports: calls reached in both, the mean of
    simulated minus recorded first arrival with its weekly block bootstrap interval, whether that mean is within
    FIRE_MARGIN_S, whether the interval contains 0, where the interval lies against plus or minus FIRE_MARGIN_S, and
    P90 minutes from the call to the first engine in both.

    Input
        dispatches : the record (DataFrame)
        tables     : the simulated dispatch tables, keyed by name (dict of DataFrames)
        verbose    : whether to print the comparison (bool)
    Output
        One row per table, period and code (DataFrame).
    Example
        compare_fire_arrivals(dispatches, {"protocol run": protocol_table}).loc[("protocol run", "Jul-Sep", "all fire calls"), "mean error s"]
        -> about -9
    """
    study_rows = dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False).to_numpy()

    def first_arrivals(table):                                   # one row per call: its code, its call time and its first arrival
        return table[study_rows].groupby("incident_id").agg(code=("protocol", "first"), received=("call_received_at", "first"),
                                                            first_on_scene=("arrived_at", "min"))

    recorded = first_arrivals(dispatches)
    rows = {}
    for name, table in tables.items():
        calls = recorded.join(first_arrivals(table)["first_on_scene"].rename("simulated"), how="inner").dropna()
        calls["error_s"] = (calls["simulated"] - calls["first_on_scene"]).dt.total_seconds()
        calls["week"] = (calls["received"] - DAY_ZERO).dt.days // 7
        for period, in_period in [("whole year", calls["received"].notna()), ("Jul-Sep", calls["received"] >= TRAVEL_TEST_FROM)]:
            for code in ["all fire calls", *STUDY_FIRE_PROTOCOLS]:
                part = calls[in_period & ((calls["code"] == code) if code in STUDY_FIRE_PROTOCOLS else True)]
                low, high = weekly_block_interval(part["error_s"], part["week"])
                rows[(name, period, code)] = {
                    "calls reached in both": len(part), "mean error s": part["error_s"].mean(), "interval low s": low,
                    "interval high s": high, "mean within m": abs(part["error_s"].mean()) <= FIRE_MARGIN_S,
                    "interval contains 0": low <= 0 <= high,
                    "interval vs +-m": ("within +-m" if -FIRE_MARGIN_S <= low and high <= FIRE_MARGIN_S else
                                        "early beyond m" if high < -FIRE_MARGIN_S else
                                        "late beyond m" if low > FIRE_MARGIN_S else "undetermined"),
                    "P90 2025 min": ((part["first_on_scene"] - part["received"]).dt.total_seconds() / 60).quantile(0.9),
                    "P90 simulated min": ((part["simulated"] - part["received"]).dt.total_seconds() / 60).quantile(0.9)}
    fire_check = pd.DataFrame.from_dict(rows, orient="index").rename_axis(["version", "period", "calls"])
    if verbose:
        print(f"--- first-engine arrival at the study's fire calls, simulated minus 2025 (m = {FIRE_MARGIN_S} s; "
              f"{int(100 * (BOOTSTRAP_INTERVAL[1] - BOOTSTRAP_INTERVAL[0]))}% weekly block-bootstrap interval)\n"
              f"{fire_check.to_string(float_format=lambda value: f'{value:.2f}')}\n")
    return fire_check


def compare_quint_calls(dispatches, tables, protocol=BRUSH_FIRE_PROTOCOL, verbose=VERBOSE):
    """Compare the share of one code's multi-engine calls that got at least one quint, in the record and in simulations.

    Multi-engine calls are those with two or more engine runs in the record. The function gives the shares by area and
    for all areas together, and every table uses the same calls. A simulation matches when its share is within
    QUINT_SHARE_TOLERANCE of the record's.

    Input
        dispatches : the record (DataFrame)
        tables     : the simulated dispatch tables, keyed by name (dict of DataFrames)
        protocol   : the call code to look at (str); brush fires by default
        verbose    : whether to print the shares (bool)
    Output
        One row per area plus a row "all" (DataFrame): calls, the record's share, each table's share, and whether it
        matches.
    Example
        compare_quint_calls(dispatches, {"protocol run": protocol_table}).loc["all", "2025"] -> about 0.07
    """
    rows = dispatches.index[dispatches["protocol"].eq(protocol).fillna(False) & (dispatches["n_engines_sent"] >= 2)]
    incident = dispatches.loc[rows, "incident_id"]
    got_quint = pd.DataFrame({name: table.loc[rows, "engine_id"].astype(object).fillna("").str.startswith("Q")
                              for name, table in {"2025": dispatches, **tables}.items()}).groupby(incident).any()
    area = dispatches.loc[rows].groupby("incident_id")["focal"].first().fillna(False).astype(bool).map(AREA_LABELS)
    shares = pd.concat([got_quint.groupby(area).mean(), got_quint.mean().to_frame("all").T])
    shares.insert(0, "calls", pd.concat([area.value_counts(), pd.Series({"all": len(area)})]))
    for name in tables:
        shares[f"{name} within {QUINT_SHARE_TOLERANCE:.0%}"] = (shares[name] - shares["2025"]).abs() <= QUINT_SHARE_TOLERANCE
    if verbose:
        print(f"--- {protocol} calls with two or more engine runs in 2025: share that got at least one quint\n"
              f"{shares.round(3).to_string()}\n")
    return shares


# ----------------------------------------------------------------------------------------------------------------------
# Printed diagnostics: what the simulator did on concrete runs
# ----------------------------------------------------------------------------------------------------------------------

def describe_engine_origins(world, scenario, dispatch_log, verbose=VERBOSE):
    """Show where each engine was when the simulation sent it: at its station, or driving home from an earlier run.

    Each engine is placed with share_of_way_home, exactly as the simulation placed it. The function prints the count of
    each origin beside the record's AVL positions, the share of the way home by minutes since clearing, three examples
    of each origin, and three checks: every share is between 0 and 1; no engine counts as on the way home
    RETURN_CHECK_SECONDS or more after clearing; and every engine at its station got the modeled travel from its
    station.

    Input
        world        : the dict from build_world
        scenario     : the scenario that was simulated (dict with dispatches, travel and call_nodes)
        dispatch_log : report["dispatches"] from simulate (DataFrame)
        verbose      : whether to print (bool)
    Output
        One row per dispatch (DataFrame): run, time, engine_id, origin, minutes since its last clear, share of the way
        home, modeled travel, travel taken, and travel from its station (minutes).
    Example
        describe_engine_origins(world, world["record"], protocol_report["dispatches"])["origin"].value_counts()
        -> counts of "at station", "driving home from a call" and "driving home after a cancelled run"
    """
    records = []
    for run, time_s, engine_id, last_clear, modeled_s, travel_s in dispatch_log.itertuples(index=False):
        share_home, _ = share_of_way_home(engine_id, last_clear, time_s, scenario["call_nodes"], world["home_nodes"],
                                          world["seconds_from_node"], world["node_position"], world["travel_model"],
                                          world["dwell_s"], world["slowdown"])
        origin = ("at station" if share_home is None else
                  "driving home from a call" if len(last_clear) == 2 else "driving home after a cancelled run")
        records.append({"run": run, "time": DAY_ZERO + pd.Timedelta(seconds=time_s), "engine_id": engine_id, "origin": origin,
                        "minutes since clear": np.nan if last_clear is None else (time_s - last_clear[0]) / 60,
                        "share of way home": np.nan if share_home is None else share_home,
                        "modeled min": modeled_s / 60, "taken min": np.nan if travel_s is None else travel_s / 60,
                        "from home min": scenario["travel"].at[run, engine_id] / 60})
    origins = pd.DataFrame(records)
    if verbose:
        recorded = scenario["dispatches"]["at_home"].astype(bool).map({True: "at its station", False: "away from its station"})
        returning = origins[origins["share of way home"].notna()]
        station = origins[origins["origin"] == "at station"]
        print(f"--- where engines were when the simulation sent them\n"
              f"{origins['origin'].value_counts().to_frame('dispatches').assign(share=lambda t: t['dispatches'] / len(origins)).round(3).to_string()}\n"
              f"2025 record, AVL position at assignment: {recorded.value_counts(normalize=True).round(3).to_dict()}\n"
              f"share of the way home by minutes since clearing (engines on the way home):\n"
              f"{returning.groupby(pd.cut(returning['minutes since clear'], [0, 2, 4, 8, 16, 60]), observed=True)['share of way home'].agg(['size', 'median']).round(2).to_string()}\n")
        for origin, part in origins.groupby("origin"):
            print(f"examples, {origin}:\n{part.head(3).round(2).to_string(index=False)}\n")
        print(f"checks: every share of the way home in [0, 1): {bool(returning['share of way home'].between(0, 1, inclusive='left').all())}; "
              f"longest time counted on the way home {returning['minutes since clear'].max():.1f} min (cut-off "
              f"{RETURN_CHECK_SECONDS / 60:.0f}); engines at their station got the travel from home: "
              f"{bool(np.isclose(station['modeled min'], station['from home min'], equal_nan=True).all())}\n")
    return origins


def describe_travel_draws(origins, spread, inside_city, verbose=VERBOSE):
    """Show what the travel draw did: the ratio of the travel taken to the modeled travel, for each dispatch.

    The ratios drawn in each area and bin of modeled travel should match the fitted ratios, because every run draws its
    own quantile from its bin's fitted ratios.

    Input
        origins     : the table from describe_engine_origins, restricted to the runs that use this travel spread
                      (DataFrame)
        spread      : the fitted travel spread those runs drew from (dict, see draw_travel_ratio)
        inside_city : whether each run's call is inside the city, indexed by run row (Series of bool)
        verbose     : whether to print five examples and the table (bool)
    Output
        One row per area and bin (DataFrame): drawn P10, P50 and P90, fitted P10, P50 and P90, and dispatches.
    Example
        describe_travel_draws(origins, world["travel_spread"], inside_city).loc[(False, 0), "drawn P50"] -> about 1.10
    """
    drawn = origins.dropna(subset=["modeled min", "taken min"]).copy()
    drawn["inside_city"] = inside_city[drawn["run"]].to_numpy()
    drawn["ratio"] = drawn["taken min"] / drawn["modeled min"]
    drawn["bin"] = [int(np.searchsorted(spread[inside][0], minutes * 60, side="left"))   # the bin draw_travel_ratio uses
                    for inside, minutes in zip(drawn["inside_city"], drawn["modeled min"])]
    by_bin = drawn.groupby(["inside_city", "bin"])["ratio"].quantile([0.1, 0.5, 0.9]).unstack()
    by_bin.columns = ["drawn P10", "drawn P50", "drawn P90"]
    fitted = pd.DataFrame({(inside, b): np.quantile(ratios, [0.1, 0.5, 0.9])
                           for inside, (_, ratios_by_bin) in spread.items() for b, ratios in enumerate(ratios_by_bin)},
                          index=["fitted P10", "fitted P50", "fitted P90"]).T.rename_axis(["inside_city", "bin"])
    table = by_bin.join(fitted).assign(dispatches=drawn.groupby(["inside_city", "bin"]).size())
    if verbose:
        print(f"--- travel draws: taken = modeled x ratio\n"
              f"{drawn[['run', 'engine_id', 'origin', 'inside_city', 'bin', 'modeled min', 'ratio', 'taken min']].head(5).round(3).to_string(index=False)}\n"
              f"drawn vs fitted ratios by area and bin of modeled travel:\n{table.round(3).to_string()}\n")
    return table


def describe_call_fates(runs, simulated, dispatch_log, fates, verbose=VERBOSE):
    """Show what the call-level fates did: the fate of each run of a multi-run call in the record and in the simulation.

    The function prints the fates side by side, two checks (every call keeps its recorded number of arrivals, and a
    released run has no arrival), and three calls whose first-dispatched run was cancelled in the record, shown run by
    run.

    Input
        runs         : the runs the simulation started from (DataFrame)
        simulated    : the simulated run (DataFrame)
        dispatch_log : report["dispatches"] from simulate (DataFrame)
        fates        : the call fates from call_fates (dict)
        verbose      : whether to print (bool)
    Output
        One row per run of those calls (DataFrame): incident_id, engine_id, fate in 2025 (arrived or cancelled),
        simulated fate (arrived, released or not sent), travel taken, and minutes from the call to dispatch and arrival
        in both.
    Example
        describe_call_fates(runs, protocol_run, protocol_report["dispatches"], fates)["simulated"].value_counts()
        -> counts of "arrived", "released" and "not sent"
    """
    rows = runs.index[runs["incident_id"].isin(fates.keys())]
    call_at = runs.loc[rows, "call_received_at"]
    table = pd.DataFrame({
        "incident_id": runs.loc[rows, "incident_id"], "engine_id": simulated.loc[rows, "engine_id"],
        "2025": np.where(runs.loc[rows, "arrived_at"].notna(), "arrived", "cancelled"),
        "simulated": np.select([simulated.loc[rows, "arrived_at"].notna(), simulated.loc[rows, "engine_id"].isna()],
                               ["arrived", "not sent"], "released"),
        "travel taken min": dispatch_log.set_index("run")["travel_s"].reindex(rows).astype(float) / 60,
        "2025 dispatched min": (runs.loc[rows, "dispatched_at"] - call_at) / 60,
        "2025 arrived min": (runs.loc[rows, "arrived_at"] - call_at) / 60,
        "simulated dispatched min": (simulated.loc[rows, "dispatched_at"] - call_at) / 60,
        "simulated arrived min": (simulated.loc[rows, "arrived_at"] - call_at) / 60})
    if verbose:
        arrivals = table.groupby("incident_id")[["2025", "simulated"]].agg(lambda fate: int((fate == "arrived").sum()))
        first = runs.loc[rows].sort_values(["incident_id", "dispatched_at"]).groupby("incident_id").head(1)
        examples = first.loc[first["arrived_at"].isna(), "incident_id"].head(3)
        print(f"--- call fates: {len(fates):,} calls with two or more runs, {len(table):,} runs\n"
              f"{pd.crosstab(table['2025'], table['simulated']).to_string()}\n"
              f"checks: calls whose number of arrivals differs from 2025: {int((arrivals['2025'] != arrivals['simulated']).sum())}; "
              f"released runs with an arrival: {int(((table['simulated'] != 'arrived') & table['simulated arrived min'].notna()).sum())}\n")
        for incident in examples:
            print(f"call {incident} (first-dispatched run cancelled in 2025):\n"
                  f"{table[table['incident_id'] == incident].drop(columns='incident_id').round(2).to_string()}\n")
    return table


def describe_engine_types(dispatches, simulated, groups, verbose=VERBOSE):
    """Show which engine types the simulation sent in groups of call types, beside the types in the record.

    Input
        dispatches : the record (DataFrame with engine_id and call_type)
        simulated  : the simulated run (DataFrame with engine_id)
        groups     : the call groups to show (list of (name, mask) pairs; each mask is a boolean Series over the
                     record's rows)
        verbose    : whether to print the table and five runs where the simulated type differs from the type CAD sent
                     (bool)
    Output
        One row per group (DataFrame): runs, the share of runs sent a quint in the record and in the simulation, and
        the share sent CAD's engine, overall and split by the type CAD sent.
    Example
        describe_engine_types(dispatches, protocol_run, groups).loc["F56 elevator", "simulated: runs sent a quint"]
        -> close to the record's share
    """
    kind = dispatches["call_type"].astype(object).fillna("(missing)")
    cad_engine = dispatches["engine_id"].astype(object)
    cad_type = cad_engine.str[0]
    sent = simulated["engine_id"].astype(object)
    same = sent == cad_engine
    table = pd.DataFrame({group: {"runs": int(mask.sum()), "2025: runs sent a quint": (cad_type[mask] == "Q").mean(),
                                  "simulated: runs sent a quint": (sent[mask].str[0] == "Q").mean(),
                                  "same engine as CAD": same[mask].mean(),
                                  "same engine, CAD sent an engine": same[mask & (cad_type == "E")].mean(),
                                  "same engine, CAD sent a quint": same[mask & (cad_type == "Q")].mean()}
                          for group, mask in groups}).T
    if verbose:
        differs = cad_type.ne(sent.str[0]) & sent.notna()
        examples = pd.DataFrame({"call type": kind, "CAD sent": cad_engine, "simulated": sent})[differs]
        print(f"--- engine types sent, record and simulation\n{table.round(3).to_string()}\n"
              f"runs where the simulation sent another type than CAD: {int(differs.sum()):,}; examples:\n{examples.head(5).to_string()}\n")
    return table


# ----------------------------------------------------------------------------------------------------------------------
# Weekly block bootstrap and the classes of a difference
# ----------------------------------------------------------------------------------------------------------------------

def weekly_block_interval(values, weeks, denominators=None):
    """Compute the two-sided 95% interval of a mean, or of a ratio of sums, by resampling whole weeks.

    The function draws whole weeks, so the calls of one busy week stay together. It makes BLOCK_BOOTSTRAP_DRAWS draws
    (seed BLOCK_BOOTSTRAP_SEED), and each draw picks as many weeks as the data has, with replacement. Without
    denominators, each draw gives the mean of values over the drawn weeks. With denominators, each draw gives
    sum(values) / sum(denominators) over the drawn weeks.

    Input
        values       : the numbers to average or the numerators (Series)
        weeks        : the week of each value (Series or array with the same index)
        denominators : None, or the denominators (Series with the same index)
    Output
        The interval (low, high) as two floats; (nan, nan) when values is empty.
    Example
        weekly_block_interval(pd.Series([10.0, 30.0, -20.0]), pd.Series([0, 0, 1])) -> (-20.0, 20.0)
        # each draw takes two weeks: means -20, 6.67 or 20 s
        weekly_block_interval(pd.Series([4.0, 2.0]), pd.Series([0, 1]), pd.Series([2.0, 2.0])) -> (1.0, 2.0)
        # each draw takes two weeks: 8/4 = 2, 6/4 = 1.5 or 4/4 = 1
    """
    if values.empty:
        return np.nan, np.nan
    sums = values.groupby(weeks).sum().to_numpy()
    counts = (values.groupby(weeks).size() if denominators is None else denominators.groupby(weeks).sum()).to_numpy()
    picks = np.random.default_rng(BLOCK_BOOTSTRAP_SEED).integers(0, len(sums), size=(BLOCK_BOOTSTRAP_DRAWS, len(sums)))
    low, high = np.quantile(sums[picks].sum(axis=1) / counts[picks].sum(axis=1), BOOTSTRAP_INTERVAL)
    return float(low), float(high)


def weekly_block_resamples(weeks):
    """Yield BLOCK_BOOTSTRAP_DRAWS weekly block resamples; each resample holds the positions of the rows in the drawn weeks.

    Input
        weeks : the week of each row (Series or array)
    Output
        A generator of integer arrays (row positions; the rows of a week drawn twice appear twice).
    Example
        next(weekly_block_resamples(np.array([0, 0, 1]))) -> e.g. array([0, 1, 0, 1]), week 0 drawn twice
    """
    weeks = np.asarray(weeks)
    week_ids = np.unique(weeks)
    members = [np.flatnonzero(weeks == week) for week in week_ids]
    rng = np.random.default_rng(BLOCK_BOOTSTRAP_SEED)
    for _ in range(BLOCK_BOOTSTRAP_DRAWS):
        yield np.concatenate([members[k] for k in rng.integers(0, len(week_ids), len(week_ids))])


def weekly_block_quantile_difference(policy_values, protocol_values, weeks, quantile):
    """Compute the policy minus protocol difference of a quantile over paired calls, with its weekly block interval.

    Input
        policy_values, protocol_values : each call's value under the policy and under the protocol (Series, same index)
        weeks                          : each call's week (Series, same index)
        quantile                       : the quantile, between 0 and 1 (float), e.g. 0.9
    Output
        Three floats: the difference, and the low and high ends of its interval.
    Example
        weekly_block_quantile_difference(policy_s, protocol_s, weeks, 0.9) -> e.g. (4.0, -6.5, 15.1)   # seconds
    """
    policy, protocol = policy_values.to_numpy(), protocol_values.to_numpy()
    draws = [np.quantile(policy[picked], quantile) - np.quantile(protocol[picked], quantile) for picked in weekly_block_resamples(weeks)]
    low, high = np.quantile(draws, BOOTSTRAP_INTERVAL)
    return float(np.quantile(policy, quantile) - np.quantile(protocol, quantile)), float(low), float(high)


def classify_against_margin(low, high, margin):
    """Classify a difference (policy minus protocol; negative means faster) by where its interval lies against a margin.

    Input
        low, high : the interval of the difference (floats)
        margin    : the smallest difference that matters (float), e.g. FIRE_MARGIN_S
    Output
        One of "meaningful improvement" (the interval is below -margin), "meaningful harm" (above +margin), "practical
        equivalence" (inside plus or minus margin) or "unresolved" (it crosses a bound) (str).
    Example
        classify_against_margin(-30.0, -12.0, 10) -> 'meaningful improvement'
    """
    if high < -margin:
        return "meaningful improvement"
    if low > margin:
        return "meaningful harm"
    if -margin <= low and high <= margin:
        return "practical equivalence"
    return "unresolved"


# ----------------------------------------------------------------------------------------------------------------------
# First-due coverage, fire readiness, the room for another engine, and policy comparisons
# ----------------------------------------------------------------------------------------------------------------------

def holding_runs(runs_table, engine_ids, moments):
    """Find the run holding each engine at each moment: the run that was sent at or before the moment and clears after it.

    Input
        runs_table : a simulated run or the record's runs (DataFrame with engine_id, dispatched_at and cleared_at in
                     seconds)
        engine_ids : the engines to look at (list or Index of str)
        moments    : the moments to look at, in seconds since DAY_ZERO (array)
    Output
        An integer array (rows: moments, columns: engines) with the row label of the holding run, or -1 when the engine
        is free.
    Example
        holding_runs(runs, ["E62"], np.array([7000.0]))[0, 0] -> 0       # the record's run 0 holds E62 at 7,000 s
    """
    holders = np.full((len(moments), len(engine_ids)), -1)
    sent = runs_table.dropna(subset=["engine_id", "dispatched_at", "cleared_at"]).sort_values("dispatched_at")
    for column, engine_id in enumerate(engine_ids):
        own = sent[sent["engine_id"] == engine_id]
        if own.empty:                                            # the engine was never sent, so it is free at every moment
            continue
        starts, ends, rows = own["dispatched_at"].to_numpy(), own["cleared_at"].to_numpy(), own.index.to_numpy()
        # ponytail: only the engine's last run started by the moment is checked; overlapping runs of one engine are rare (3 in the record, none in the simulation)
        last = np.searchsorted(starts, moments, side="right") - 1
        held = (last >= 0) & (ends[np.maximum(last, 0)] > moments)
        holders[:, column] = np.where(held, rows[np.maximum(last, 0)], -1)
    return holders


def check_first_due(dispatches, runs_table, estimated_travel, engines, verbose=VERBOSE):
    """For each of the study's fire calls, check whether every engine of its first-due station was busy elsewhere.

    The first-due station is the station of the engine with the least modeled travel from its station to the call;
    only engines in service and allowed by the agency line count. This ranking leaves out the quint penalty that the
    protocol adds, so on a brush fire the first-due station can differ from the protocol's first choice. An engine is busy
    elsewhere when it is on a run of another call at the moment the fire call was first dispatched.

    Input
        dispatches       : the dispatch table (DataFrame, the record or a scenario's table)
        runs_table       : the run to check (DataFrame: simulated, or the record's runs for the record itself)
        estimated_travel : the modeled travel with the agency line from each engine's station to each run's call, in
                           seconds (DataFrame, rows: runs, columns: engines)
        engines          : the engines (DataFrame with home_station)
        verbose          : whether to print the table and examples (bool)
    Output
        One row per fire call (DataFrame indexed by incident_id): code, inside_city, received_at, week,
        first_due_station, station_engines, committed (True when every engine of the station was busy elsewhere),
        held_by (the calls holding those engines), held_by_medical, held_by_fire_category, minutes_to_first_on_scene,
        first_on_scene_engine, and extra_modeled_min (the modeled travel of the first engine on scene minus that of
        the first-due engine, both from their stations, in minutes).
    Example
        check_first_due(dispatches, protocol_run, record_travel, engines)["committed"].mean() -> about 0.074
    """
    station_key = engines.set_index("engine_id")["home_station"].astype(object)
    station_key = station_key.where(station_key.notna(), station_key.index.to_series())   # an engine without a station id counts as its own station
    engine_ids = estimated_travel.columns
    station_of_column = station_key.reindex(engine_ids).to_numpy()
    inside_city = dispatches["focal"].fillna(False).astype(bool)
    call_type, protocol, priority = (dispatches[column].astype(object).fillna("(missing)") for column in ["call_type", "protocol", "priority"])
    rows = dispatches.index[dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False) & dispatches["incident_lon"].notna()]
    sent = runs_table.loc[rows, ["dispatched_at", "arrived_at", "engine_id"]].assign(incident_id=dispatches.loc[rows, "incident_id"])
    first_sent = sent.dropna(subset=["dispatched_at"]).sort_values("dispatched_at").groupby("incident_id").head(1)
    first_on_scene = sent.dropna(subset=["arrived_at"]).sort_values("arrived_at").groupby("incident_id").head(1).set_index("incident_id")
    travel = estimated_travel.loc[first_sent.index].to_numpy()                # modeled travel of every engine from its station to the call (seconds)
    reachable = np.isfinite(travel)
    first_due_column = np.argmin(np.where(reachable, travel, np.inf), axis=1)
    holders = holding_runs(runs_table, engine_ids, first_sent["dispatched_at"].to_numpy())
    records = []
    for i, (row, incident_id) in enumerate(zip(first_sent.index, first_sent["incident_id"])):
        station = station_of_column[first_due_column[i]]
        columns = np.flatnonzero((station_of_column == station) & reachable[i])
        elsewhere = [holders[i, c] for c in columns if holders[i, c] >= 0 and dispatches.at[holders[i, c], "incident_id"] != incident_id]
        received = dispatches.at[row, "call_received_at"]
        arrival = first_on_scene["arrived_at"].get(incident_id, np.nan)
        arrival_engine = first_on_scene["engine_id"].get(incident_id)
        arrival_column = engine_ids.get_loc(arrival_engine) if arrival_engine in engine_ids else None
        records.append({
            "incident_id": incident_id, "code": protocol.at[row], "inside_city": bool(inside_city.at[row]),
            "received_at": received, "week": (received - DAY_ZERO).days // 7,
            "first_due_station": station, "station_engines": ", ".join(engine_ids[columns]),
            "committed": len(elsewhere) == len(columns),
            "held_by": "; ".join(f"{protocol.at[h]} {call_type.at[h]} {priority.at[h]}" for h in elsewhere),
            "held_by_medical": any(call_type.at[h] == "Medical" for h in elsewhere) and len(elsewhere) == len(columns),
            "held_by_fire_category": any(call_type.at[h] == "Fire" for h in elsewhere) and len(elsewhere) == len(columns),
            "minutes_to_first_on_scene": (arrival - (received - DAY_ZERO).total_seconds()) / 60,
            "first_on_scene_engine": arrival_engine,
            "extra_modeled_min": np.nan if arrival_column is None else (travel[i, arrival_column] - travel[i, first_due_column[i]]) / 60})
    calls = pd.DataFrame(records).set_index("incident_id")
    if verbose:
        show_table("first-due check per fire call", calls, verbose)
        examples = pd.concat([calls[calls["committed"]].head(3), calls[~calls["committed"]].head(2)])
        print(f"fire calls checked: {len(calls):,}; first-due station had no free engine: {int(calls['committed'].sum())} "
              f"({calls['committed'].mean():.1%})\nexamples (three committed, two free):\n"
              f"{examples[['code', 'first_due_station', 'station_engines', 'committed', 'held_by', 'first_on_scene_engine', 'minutes_to_first_on_scene', 'extra_modeled_min']].round(2).to_string()}\n")
    return calls


def summarize_first_due(versions, verbose=VERBOSE):
    """Summarize the first-due checks: how often a fire call's first-due station had no free engine, and what followed.

    For each version and group of fire calls (all four codes, each code, inside and outside the city, July to
    September), the summary gives: fire calls; the share whose first-due station had no free engine, with its weekly
    block interval; among those, the share held by a medical call and by a Fire-category call; minutes to the first
    engine on scene (P50, P90) when the station had a free engine and when it had none; and the median extra modeled
    minutes of the first engine on scene when the station had none.

    Input
        versions : the tables from check_first_due, keyed by version name (dict of DataFrames), e.g. the record and the
                   protocol run
        verbose  : whether to print the summary and the ten stations with the most such calls (bool)
    Output
        Two tables. The summary (DataFrame, rows: version and group). The stations (DataFrame, rows: first-due
        stations; columns per version: fire calls, committed and share), sorted by the last version's committed calls.
    Example
        first_due_summary, first_due_by_station = summarize_first_due({"2025 record": first_due_record,
                                                                       "protocol run": first_due_protocol})
    """
    rows, stations = {}, {}
    for name, calls in versions.items():
        groups = [("all fire calls", calls.index), *[(code, calls.index[calls["code"] == code]) for code in STUDY_FIRE_PROTOCOLS],
                  ("inside city", calls.index[calls["inside_city"]]), ("outside city", calls.index[~calls["inside_city"]]),
                  ("Jul-Sep", calls.index[calls["received_at"] >= TRAVEL_TEST_FROM])]
        for group, ids in groups:
            part = calls.loc[ids]
            committed, free = part[part["committed"]], part[~part["committed"]]
            low, high = weekly_block_interval(part["committed"].astype(float), part["week"])
            rows[(name, group)] = {
                "fire calls": len(part), "no free first-due engine": part["committed"].mean(), "interval low": low, "interval high": high,
                "held by medical": committed["held_by_medical"].mean(), "held by Fire category": committed["held_by_fire_category"].mean(),
                "first on scene min, station free: P50": free["minutes_to_first_on_scene"].median(),
                "station free: P90": free["minutes_to_first_on_scene"].quantile(0.9),
                "station committed: P50": committed["minutes_to_first_on_scene"].median(),
                "station committed: P90": committed["minutes_to_first_on_scene"].quantile(0.9),
                "extra modeled min when committed: P50": committed["extra_modeled_min"].median()}
        station = calls.groupby("first_due_station").agg(fire_calls=("committed", "size"), committed=("committed", "sum"))
        station["share"] = station["committed"] / station["fire_calls"]
        stations[name] = station
    first_due_summary = pd.DataFrame(rows).T.rename_axis(["version", "calls"])
    first_due_by_station = pd.concat(stations, axis=1).sort_values((list(versions)[-1], "committed"), ascending=False)
    if verbose:
        print(f"--- fire calls whose first-due station had no free engine (95% weekly block-bootstrap interval)\n"
              f"{first_due_summary.round(3).to_string()}\n"
              f"by first-due station (where), ten with the most such calls:\n{first_due_by_station.head(10).round(3).to_string()}\n")
    return first_due_summary, first_due_by_station


def fire_readiness(world, runs_table, wait_for_busy=False, verbose=VERBOSE):
    """Measure fire readiness over a run: the mean modeled travel from the nearest idle engine to where fire calls happen.

    Between consecutive dispatches, clears and changes of service, the set of idle engines is fixed. In each such
    segment, tau(x) is the modeled travel to cell x from the nearest idle engine allowed there, with idle engines at
    their stations. For call types without automatic aid, only engines of the cell's own agency are allowed; for the
    other call types, engines of either agency are allowed. Readiness R is the fire-call-weighted mean of tau over the
    cells, in minutes; a lower R means better protection. R_full is the same with every in-service engine idle, so
    R - R_full is the protection lost to busy engines. Segments are also cut at every hour, and the yearly value
    weights each segment by its length and its hour's share of fire calls.
    A segment where some weighted cell has no idle allowed engine is left out (one or two a year in the record). With
    wait_for_busy, such a cell instead gets the least, over the allowed in-service engines, of the minutes until the
    engine clears (at the segment's middle) plus its travel from its station; a closer engine that clears later can
    give the least sum. Then the segment also gets waiting_weight, the weight of those cells.
    Each wait shrinks by one minute per minute, and no run clears inside a segment. R therefore falls by
    waiting_weight minutes per minute, and the value at the segment's middle is the segment's mean.
    environment.make_readiness_tools computes R for one set of idle engines. The two functions sum in a different
    order, so they are kept apart to keep every result unchanged.

    Input
        world         : the dict from build_world (engines, readiness_cells, travel_to_cells, fire_hour_share)
        runs_table    : the run to measure (DataFrame with engine_id, dispatched_at, cleared_at in seconds)
        wait_for_busy : whether a cell with no idle allowed engine gets a wait instead of its segment being left out
                        (bool)
        verbose       : whether to print the segments, the summary and the cells that lose the most (bool)
    Output
        Three tables. segments (DataFrame, one row per segment): start_s, end_s, idle_engines, R_min, R_full_min,
        loss_min, and waiting_weight when wait_for_busy is True. cell_tau (DataFrame, one row per cell): time-weighted
        mean tau and tau_full in minutes and their difference. summary (DataFrame): R, R_full and the loss for the year
        (hour-weighted and time-weighted), each month and each hour, plus quantiles of the loss.
    Example
        segments, cell_tau, summary = fire_readiness(world, protocol_run)
        summary.loc["year, hour-weighted", "loss_min"] -> about 0.21 (12.7 s)
        toy: one engine 5 min from the only cell, on a run from 0 to 600 s: the segment 0-600 s (middle 300 s) is left
        out; with wait_for_busy its R is (600 - 300) / 60 + 5 = 10 min and waiting_weight is 1 (R falls from 15 to 5 min)
    """
    engines, cells, travel_to_cells, hour_share = world["engines"], world["readiness_cells"], world["travel_to_cells"], world["fire_hour_share"]
    engine_ids = travel_to_cells.index
    year_end_s = (YEAR_END - DAY_ZERO).total_seconds()
    service = engines.set_index("engine_id").loc[engine_ids]
    service_from = (service["in_service_from"] - DAY_ZERO).dt.total_seconds().to_numpy()
    service_to = (service["in_service_to"] - DAY_ZERO).dt.total_seconds().to_numpy()
    sent = runs_table.dropna(subset=["engine_id", "dispatched_at", "cleared_at"])
    every_hour = np.arange(0.0, year_end_s, 3600.0)             # cut at every hour, so each segment lies in one hour and takes its weight; this also cuts at every month
    times = np.unique(np.clip(np.concatenate([[0.0, year_end_s], every_hour, sent["dispatched_at"], sent["cleared_at"], service_from,
                                              service_to]), 0.0, year_end_s))
    starts, ends = times[:-1], times[1:]
    middle, duration = (starts + ends) / 2, ends - starts
    in_service = (middle[:, None] >= service_from) & (middle[:, None] < service_to)
    holders = holding_runs(runs_table, engine_ids, middle)
    idle = in_service & (holders < 0)
    if wait_for_busy:                                            # minutes until each engine's holding run clears (0 when the engine is idle)
        clears = runs_table["cleared_at"].reindex(holders.ravel()).to_numpy(dtype=float).reshape(holders.shape)
        wait_min = np.where(holders >= 0, (clears - middle[:, None]) / 60, 0.0)
    travel_min = travel_to_cells.to_numpy() / 60
    own_agency = (service["engine_agency"] == "GFR").to_numpy()[:, None] == cells["cell_gfr"].to_numpy()[None, :]
    aid_cells, own_cells = (cells["weight_aid"] > 0).to_numpy(), (cells["weight_own"] > 0).to_numpy()
    weight_aid, weight_own = cells["weight_aid"].to_numpy()[aid_cells], cells["weight_own"].to_numpy()[own_cells]
    cell_weight = (cells["weight_aid"] + cells["weight_own"]).to_numpy()
    readiness, cell_minutes = {}, {}
    waiting_weight = np.zeros(len(starts))                       # weight of the cells whose tau includes a wait (wait_for_busy)
    for name, available in [("R", idle), ("R_full", in_service)]:
        values, cell_total, cell_time = np.empty(len(starts)), np.zeros(len(cells)), np.zeros(len(cells))
        for chunk in range(0, len(starts), READINESS_CHUNK):
            part = slice(chunk, chunk + READINESS_CHUNK)
            tau_any = np.where(available[part, :, None], travel_min[None], np.inf).min(axis=1)
            tau_own = np.where(available[part, :, None] & own_agency[None], travel_min[None], np.inf).min(axis=1)
            if wait_for_busy and name == "R":
                stuck = np.flatnonzero(~np.isfinite(tau_any[:, aid_cells]).all(axis=1) | ~np.isfinite(tau_own[:, own_cells]).all(axis=1))
                if len(stuck):                                   # a weighted cell with no idle allowed engine: least wait plus travel over busy engines
                    segment_rows = np.arange(len(starts))[part][stuck]
                    waited = np.where(in_service[segment_rows, :, None], wait_min[segment_rows, :, None] + travel_min[None], np.inf)
                    stuck_any, stuck_own = np.isinf(tau_any[stuck]), np.isinf(tau_own[stuck])
                    tau_any[stuck] = np.where(stuck_any, waited.min(axis=1), tau_any[stuck])
                    tau_own[stuck] = np.where(stuck_own, np.where(own_agency[None], waited, np.inf).min(axis=1), tau_own[stuck])
                    waiting_weight[segment_rows] = stuck_any[:, aid_cells] @ weight_aid + stuck_own[:, own_cells] @ weight_own
            values[part] = tau_any[:, aid_cells] @ weight_aid + tau_own[:, own_cells] @ weight_own
            tau_cell = (np.where(aid_cells, tau_any, 0) * cells["weight_aid"].to_numpy()
                        + np.where(own_cells, tau_own, 0) * cells["weight_own"].to_numpy()) / cell_weight
            reachable = np.isfinite(tau_cell)      # ponytail: without wait_for_busy, a segment with no allowed idle engine for a cell is left out (one or two segments a year)
            cell_total += duration[part] @ np.where(reachable, tau_cell, 0.0)
            cell_time += duration[part] @ reachable
        readiness[name], cell_minutes[name] = values, cell_total / cell_time
    segments = pd.DataFrame({"start_s": starts, "end_s": ends, "idle_engines": idle.sum(axis=1), "R_min": readiness["R"],
                             "R_full_min": readiness["R_full"]})
    segments["loss_min"] = segments["R_min"] - segments["R_full_min"]
    if wait_for_busy:                                            # R falls by this many minutes per minute inside the segment
        segments["waiting_weight"] = waiting_weight
    cell_tau = cells[["cell_x", "cell_y", "lon", "lat", "calls_aid", "calls_other", "cell_gfr"]].assign(
        tau_full_min=cell_minutes["R_full"], tau_min=cell_minutes["R"], loss_min=cell_minutes["R"] - cell_minutes["R_full"])
    hour = (starts // 3600 % 24).astype(int)
    month = (DAY_ZERO + pd.to_timedelta(starts, unit="s")).strftime("%Y-%m")
    finite = np.isfinite(segments["R_min"].to_numpy())
    weight = duration * hour_share.to_numpy()[hour] * finite

    def weighted(mask, column, weights):
        return float(np.sum(segments[column].to_numpy()[mask & finite] * weights[mask & finite]) / np.sum(weights[mask & finite]))

    every = np.ones(len(starts), dtype=bool)
    summary_rows = {"year, hour-weighted": {column: weighted(every, column, weight) for column in ["R_min", "R_full_min", "loss_min"]},
                    "year, time-weighted": {column: weighted(every, column, duration) for column in ["R_min", "R_full_min", "loss_min"]}}
    for label, mask in [*[(f"month {m}", month == m) for m in sorted(set(month))], *[(f"hour {h:02d}", hour == h) for h in range(24)]]:
        summary_rows[label] = {column: weighted(mask, column, duration) for column in ["R_min", "R_full_min", "loss_min"]}
    readiness_summary = pd.DataFrame(summary_rows).T
    for quantile in (0.5, 0.9, 0.99):
        readiness_summary.loc[f"loss, time-weighted P{int(quantile * 100)}", "loss_min"] = np.quantile(
            segments["loss_min"].to_numpy()[finite], quantile, weights=duration[finite], method="inverted_cdf")
    readiness_summary.loc["hours with loss over 1 min", "loss_min"] = duration[finite & (segments["loss_min"].to_numpy() > 1)].sum() / 3600
    readiness_summary.loc["segments with a cell no allowed engine can reach", "loss_min"] = int((~finite).sum())
    readiness_summary.loc["minutes in those segments (left out)", "loss_min"] = duration[~finite].sum() / 60
    if verbose:
        show_table("readiness segments (between dispatches and clears)", segments.round(3), verbose)
        print(f"segments: {len(segments):,}; hours: {duration.sum() / 3600:,.0f}; idle engines per segment: median "
              f"{np.median(segments['idle_engines']):.0f}, min {segments['idle_engines'].min()}\n"
              f"--- fire readiness (minutes; R: idle engines, R_full: every in-service engine idle)\n{readiness_summary.round(3).to_string()}\n"
              f"cells losing the most (time-weighted mean tau minus tau with every engine idle):\n"
              f"{cell_tau.nlargest(5, 'loss_min').round(2).to_string()}\n")
    return segments, cell_tau, readiness_summary


def readiness_loss_in_windows(segments, starts_s, ends_s):
    """Compute the time-weighted mean fire readiness loss (R - R_full, in seconds) over each time window.

    Inside a segment, the loss at a given time equals the loss at the segment's middle minus waiting_weight times
    (time - middle); waiting_weight is 0 without waits. Because the loss is linear inside a segment, a window that
    cuts a segment is integrated exactly.

    Input
        segments : the contiguous segments from fire_readiness (DataFrame)
        starts_s : the start of each window, in seconds since DAY_ZERO (array-like)
        ends_s   : the end of each window, in seconds since DAY_ZERO; same length as starts_s (array-like)
    Output
        The mean loss over each window, in seconds (array).
    Example
        readiness_loss_in_windows(segments, [0.0], [7200.0]) -> the loss over the first two hours, in seconds
        toys: segments 0-60 s with a loss of 1 min and 60-120 s with 3 min; window 30-90 s -> (30 x 60 + 30 x 180) / 60 = 120 s
              one segment 0-600 s, loss 5 min at its middle, waiting_weight 1 (it falls from 10 to 0 min); window 0-300 s -> 450 s
    """
    segment_start, segment_end = segments["start_s"].to_numpy(), segments["end_s"].to_numpy()
    middle, level = (segment_start + segment_end) / 2, 60 * segments["loss_min"].to_numpy()
    slope = segments["waiting_weight"].to_numpy() if "waiting_weight" in segments else np.zeros(len(segments))
    integral = np.concatenate([[0.0], np.cumsum(level * (segment_end - segment_start))])

    def integral_to(moments):
        i = np.clip(np.searchsorted(segment_start, moments, side="right") - 1, 0, len(segments) - 1)
        return (integral[i] + level[i] * (moments - segment_start[i])
                - slope[i] * ((moments - middle[i]) ** 2 - (segment_start[i] - middle[i]) ** 2) / 2)

    starts, ends = np.asarray(starts_s, dtype=float), np.asarray(ends_s, dtype=float)
    return (integral_to(ends) - integral_to(starts)) / (ends - starts)


def census_decisions(world, scenario, runs_table, action_windows_s, verbose=VERBOSE):
    """Count the room that the opportunity-cost rule or a learned policy has on low-acuity calls, without simulating.

    For every low-acuity run of runs_table, the census looks at the idle in-service engines at the run's dispatch
    moment; the run's own engine counts as idle. Among the allowed engines, it finds the protocol's engine: the engine
    with the least modeled travel from its station plus the quint penalty. For each action window, it counts the
    engines ranked within the window of the protocol's engine, picks the engine the rule would send (rule_pick), and
    records the opportunity cost that this engine saves and its extra modeled travel. Travel is from the stations, as
    in the opportunity cost.

    Input
        world            : the dict from build_world (engines, quint_indicator, opportunity_cost)
        scenario         : the scenario the run simulates (dict with dispatches, travel, quint_penalty, low_acuity)
        runs_table       : the protocol run to count on (DataFrame)
        action_windows_s : the action windows to count, in seconds (list), e.g. [30, 60, 120]
        verbose          : whether to print the summary and examples (bool)
    Output
        One row per low-acuity decision (DataFrame indexed by run row): dispatched_at, inside_city, protocol_engine,
        sent_engine, oc_protocol_min, and for each window: choices, rule_engine, oc_saving_min and extra_travel_s.
    Example
        census_decisions(world, world["record"], protocol_run, [30, 60, 120])["choices_60"].ge(2).mean() -> about 0.37
    """
    dispatches, estimated_travel, engines = scenario["dispatches"], scenario["travel"], world["engines"]
    quint_penalty_of_run, quint_indicator = scenario["quint_penalty"], world["quint_indicator"]
    low_acuity, opportunity_cost = scenario["low_acuity"], world["opportunity_cost"]
    engine_ids = estimated_travel.columns
    service = engines.set_index("engine_id").loc[engine_ids]
    rows = dispatches.index[low_acuity & dispatches["incident_lon"].notna()]
    rows = rows[runs_table.loc[rows, ["dispatched_at", "engine_id"]].notna().all(axis=1).to_numpy()]
    moments = runs_table.loc[rows, "dispatched_at"].to_numpy()
    when = (DAY_ZERO + pd.to_timedelta(moments, unit="s")).to_numpy()[:, None]
    holders = holding_runs(runs_table, engine_ids, moments)
    idle = ((when >= service["in_service_from"].to_numpy()[None, :]) & (when < service["in_service_to"].to_numpy()[None, :])
            & ((holders < 0) | (holders == rows.to_numpy()[:, None])))
    travel = estimated_travel.loc[rows].to_numpy()
    ranked = travel + quint_penalty_of_run.loc[rows].to_numpy()[:, None] * quint_indicator.reindex(engine_ids).to_numpy()[None, :]
    ranked = np.where(idle & np.isfinite(ranked), ranked, np.inf)
    inside_city = dispatches["focal"].fillna(False).astype(bool)
    records = []
    for i, row in enumerate(rows):
        protocol_column = int(np.argmin(ranked[i]))
        if not np.isfinite(ranked[i, protocol_column]):
            continue
        protocol_engine = engine_ids[protocol_column]
        widest = engine_ids[ranked[i] <= ranked[i, protocol_column] + max(action_windows_s)]
        costs = opportunity_cost(engine_ids[idle[i]], widest)
        record = {"row": row, "dispatched_at": DAY_ZERO + pd.Timedelta(seconds=moments[i]),
                  "inside_city": bool(inside_city.at[row]), "protocol_engine": protocol_engine,
                  "sent_engine": runs_table.at[row, "engine_id"], "oc_protocol_min": costs[protocol_engine]}
        for window in action_windows_s:
            near = engine_ids[ranked[i] <= ranked[i, protocol_column] + window]
            pick = rule_pick(costs[near], protocol_engine)
            with np.errstate(invalid="ignore"):                  # both costs infinite: the rule keeps the protocol's engine, and inf - inf makes the saving NaN
                saving = costs[protocol_engine] - costs[pick]
            record.update({f"choices_{window}": len(near), f"rule_engine_{window}": pick, f"oc_saving_min_{window}": saving,
                           f"extra_travel_s_{window}": travel[i, engine_ids.get_loc(pick)] - travel[i, protocol_column]})
        records.append(record)
    census = pd.DataFrame(records).set_index("row")
    if verbose:
        days = 365
        summary = pd.DataFrame({f"{window} s": {
            "low-acuity decisions": len(census), "with two or more engines in the window": (census[f"choices_{window}"] >= 2).mean(),
            "rule sends another engine": (census[f"rule_engine_{window}"] != census["protocol_engine"]).mean(),
            "  per day": (census[f"rule_engine_{window}"] != census["protocol_engine"]).sum() / days,
            "  readiness saved when it does (s), median": census.loc[census[f"rule_engine_{window}"] != census["protocol_engine"], f"oc_saving_min_{window}"].replace(np.inf, np.nan).median() * 60,
            "  extra modeled travel to the call (s), median": census.loc[census[f"rule_engine_{window}"] != census["protocol_engine"], f"extra_travel_s_{window}"].median(),
            "  saving infinite (protocol engine was the last one for a cell)": int(np.isinf(census[f"oc_saving_min_{window}"]).sum())}
            for window in action_windows_s})
        changed = census[census[f"rule_engine_{ACTION_WINDOW_S}"] != census["protocol_engine"]]
        print(f"--- low-acuity decisions on the protocol run: the room for another engine\n{summary.round(3).to_string()}\n"
              f"protocol engine (from the stations) equals the engine the run sent: {(census['protocol_engine'] == census['sent_engine']).mean():.1%}\n"
              f"examples where the rule sends another engine:\n{changed.head(5).round(3).to_string()}\n")
    return census


def compare_policies(world, scenario, protocol_table, policy_table, first_due_protocol, readiness_protocol, readiness_policy,
                     verbose=VERBOSE):
    """Compare a policy's simulated run with the protocol's run outcome by outcome; both runs use the same random draws.

    Each outcome gets the policy minus protocol difference in seconds, its weekly block interval, and its class against
    the margin (FIRE_MARGIN_S for fire, MEDICAL_MARGIN_S for the medical constraint). The outcomes are:
    - first-engine arrival at the study's fire calls: whole year; July to September; calls whose first-due station was
      committed under the protocol; calls whose first-due station is one of HOTSPOT_STATIONS; calls in DAYTIME_HOURS;
    - fire readiness: the hour-weighted mean R of each week;
    - the P90 first arrival at high-acuity medical calls (HIGH_ACUITY_LEVELS), against MEDICAL_MARGIN_S;
    - the mean first arrival at low-acuity calls, which is the cost the rule pays;
    - and the number of runs that got another engine than in the protocol run.

    Input
        world              : the dict from build_world (fire_hour_share)
        scenario           : the scenario both runs simulate (dict with dispatches and low_acuity)
        protocol_table     : the protocol run's dispatch table (DataFrame)
        policy_table       : the policy run's dispatch table (DataFrame)
        first_due_protocol : the protocol run's first-due check (DataFrame from check_first_due)
        readiness_protocol : the protocol run's readiness segments (DataFrame from fire_readiness)
        readiness_policy   : the policy run's readiness segments (DataFrame)
        verbose            : whether to print the comparison (bool)
    Output
        One row per outcome (DataFrame): calls, difference s, interval low s, interval high s, class.
    Example
        compare_policies(world, world["record"], protocol_table, rule_table, first_due_protocol, readiness_protocol,
                         readiness_rule).loc["fire calls, whole year", "class"] -> 'practical equivalence'
    """
    dispatches, hour_share, low_acuity = scenario["dispatches"], world["fire_hour_share"], scenario["low_acuity"]
    received_s = (dispatches["call_received_at"] - DAY_ZERO).dt.total_seconds()

    def first_arrival_s(table, rows):
        part = table.loc[rows].assign(incident_id=dispatches.loc[rows, "incident_id"]).dropna(subset=["arrived_at"])
        first = part.groupby("incident_id")["arrived_at"].min()
        return (first - DAY_ZERO).dt.total_seconds() - received_s.groupby(dispatches["incident_id"]).first().reindex(first.index)

    def paired(rows):
        both = pd.concat({"policy": first_arrival_s(policy_table, rows), "protocol": first_arrival_s(protocol_table, rows)}, axis=1).dropna()
        call_time = dispatches.loc[rows].groupby("incident_id")["call_received_at"].first().reindex(both.index)
        return both.assign(week=(call_time - DAY_ZERO).dt.days // 7, hour=call_time.dt.hour, received=call_time)

    fire_rows = dispatches.index[dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False)]
    fire = paired(fire_rows)
    committed = set(first_due_protocol.index[first_due_protocol["committed"]])
    hotspot = set(first_due_protocol.index[first_due_protocol["first_due_station"].isin(HOTSPOT_STATIONS)])
    groups = [("fire calls, whole year", fire.index), ("fire calls, Jul-Sep", fire.index[fire["received"] >= TRAVEL_TEST_FROM]),
              ("fire calls, first-due station committed (protocol)", fire.index[fire.index.isin(committed)]),
              ("fire calls, hotspot first-due stations", fire.index[fire.index.isin(hotspot)]),
              ("fire calls, daytime hours", fire.index[fire["hour"].isin(DAYTIME_HOURS)])]
    rows = {}
    for label, ids in groups:
        part = fire.loc[ids]
        difference = part["policy"] - part["protocol"]
        low, high = weekly_block_interval(difference, part["week"])
        rows[label] = {"calls": len(part), "difference s": difference.mean(), "interval low s": low, "interval high s": high,
                       "class": classify_against_margin(low, high, FIRE_MARGIN_S)}
    weekly = {}
    for name, segments in [("policy", readiness_policy), ("protocol", readiness_protocol)]:
        weight = (segments["end_s"] - segments["start_s"]) * hour_share.to_numpy()[(segments["start_s"] // 3600 % 24).astype(int)]
        weight = weight.where(np.isfinite(segments["R_min"]), 0.0)
        week = segments["start_s"] // (7 * 86400)
        weekly[name] = (segments["R_min"].where(weight > 0, 0.0) * weight).groupby(week).sum() / weight.groupby(week).sum()
    readiness_difference = (weekly["policy"] - weekly["protocol"]) * 60
    low, high = weekly_block_interval(readiness_difference, pd.Series(readiness_difference.index, index=readiness_difference.index))
    rows["fire readiness R, weekly hour-weighted mean"] = {"calls": len(readiness_difference), "difference s": readiness_difference.mean(),
                                                          "interval low s": low, "interval high s": high,
                                                          "class": classify_against_margin(low, high, FIRE_MARGIN_S)}
    high_rows = dispatches.index[dispatches["call_type"].eq("Medical").fillna(False)
                                 & dispatches["determinant_level"].isin(HIGH_ACUITY_LEVELS).fillna(False)]
    medical = paired(high_rows)
    point, low, high = weekly_block_quantile_difference(medical["policy"], medical["protocol"], medical["week"], 0.9)
    rows["high-acuity medical, P90 first arrival"] = {"calls": len(medical), "difference s": point, "interval low s": low, "interval high s": high,
                                                       "class": "within delta" if high <= MEDICAL_MARGIN_S else "beyond delta or unresolved"}
    low_rows = dispatches.index[low_acuity]
    cost = paired(low_rows)
    difference = cost["policy"] - cost["protocol"]
    low, high = weekly_block_interval(difference, cost["week"])
    rows["low-acuity calls, mean first arrival (cost)"] = {"calls": len(cost), "difference s": difference.mean(), "interval low s": low,
                                                          "interval high s": high, "class": ""}
    rows["runs sent another engine than the protocol run"] = {
        "calls": int((policy_table["engine_id"].astype(object).fillna("") != protocol_table["engine_id"].astype(object).fillna("")).sum())}
    comparison = pd.DataFrame(rows).T
    if verbose:
        print(f"--- policy minus protocol, same random draws (seconds; negative = faster; m = {FIRE_MARGIN_S} s, delta = "
              f"{MEDICAL_MARGIN_S} s; 95% weekly block-bootstrap interval)\n{comparison.to_string()}\n")
    return comparison


def summarize_over_seeds(policy_comparison, verbose=VERBOSE):
    """Summarize the rule minus protocol differences at ACTION_WINDOW_S over the travel seeds.

    Input
        policy_comparison : the comparisons stacked by seed, action window and outcome (DataFrame)
        verbose           : whether to print the summary (bool)
    Output
        One row per outcome (DataFrame): mean, lowest and highest difference over seeds, and the count of each class.
    Example
        summarize_over_seeds(policy_comparison).loc["fire calls, whole year", "classes over seeds"]
        -> {'practical equivalence': 5}
    """
    main_window = policy_comparison.xs(ACTION_WINDOW_S, level="action_window_s")
    over_seeds = pd.DataFrame({
        "mean difference s": main_window["difference s"].astype(float).groupby(level="outcome", sort=False).mean(),
        "lowest s": main_window["difference s"].astype(float).groupby(level="outcome", sort=False).min(),
        "highest s": main_window["difference s"].astype(float).groupby(level="outcome", sort=False).max(),
        "classes over seeds": main_window["class"].fillna("").groupby(level="outcome", sort=False).agg(lambda c: dict(c.value_counts()))})
    if verbose:
        print(f"--- opportunity-cost rule ({ACTION_WINDOW_S} s window) minus protocol over "
              f"{main_window.index.get_level_values('seed').nunique()} seeds\n{over_seeds.round(2).to_string()}\n")
    return over_seeds


def branch_decisions(world, record, protocol_run, census, verbose=VERBOSE):
    """Test whether the fire readiness that a single decision saves turns into later benefit, with paired branch simulations.

    The function draws low-acuity decisions where the census finds that the rule would send another engine (at
    ACTION_WINDOW_S). It draws them evenly over BRANCH_STRATA strata of the opportunity cost saved (seed BRANCH_SEED).
    Each decision is simulated twice from the same state. That state is rebuilt by simulating the calls first
    dispatched in the BRANCH_WARMUP_S seconds before the decision. Branch A follows the protocol. In that rebuilt state,
    branch B applies the rule at the decision and the protocol everywhere else. A decision counts only if two
    conditions hold: branch A reproduces the full-year protocol run over the BRANCH_HORIZON_S seconds after the
    decision (every run's engine and times, and the fire readiness), and the rule still changes the engine in that
    state. Drawing goes on until BRANCH_DECISIONS such decisions count; each stratum has an equal target and tries at
    most three times that target. Over the horizon, each branch gets its mean fire readiness, its first arrivals at
    later study fire calls and at the decision's own call, and the decision run's busy time.

    Input
        world        : the dict from build_world
        record       : the record as a scenario (world["record"])
        protocol_run : the record simulated under the protocol (DataFrame from simulate)
        census       : the census on that run (DataFrame from census_decisions)
        verbose      : whether to print the results, the slope on the saving and examples (bool)
    Output
        Two tables. branches (DataFrame, one row per decision tried): stratum, time, the engines of A and B, the
        saving, extra travel, workload, benefit_s (readiness gained over the horizon, A minus B, in seconds), later fire
        calls and their arrival difference, the own call's difference, and whether it counted. branch_cells (DataFrame,
        one row per cell): the readiness change summed over the counted decisions (minutes times hours, A minus B;
        positive where the rule's engine left better cover).
    Example
        branches, branch_cells = branch_decisions(world, world["record"], protocol_run, census)
        branches.loc[branches["counted"], "benefit_s"].mean() -> readiness the rule gains over two hours, in seconds
    """
    dispatches, runs, cells, readiness_of = record["dispatches"], record["runs"], world["readiness_cells"], world["readiness_of"]
    engine_ids = world["travel_to_cells"].index
    service = world["engines"].set_index("engine_id").loc[engine_ids]
    service_from = (service["in_service_from"] - DAY_ZERO).dt.total_seconds().to_numpy()
    service_to = (service["in_service_to"] - DAY_ZERO).dt.total_seconds().to_numpy()

    def simulate_window(window_runs, policy):
        return simulate(world, {**record, "runs": window_runs, "fates": call_fates(window_runs)}, policy, verbose=False)

    def horizon_readiness(table, start, end):
        """Return the mean fire readiness of a simulated run from start to end (minutes), and each cell's tau summed
        over time in that span (minutes times hours)."""
        sent = table.dropna(subset=["engine_id", "dispatched_at", "cleared_at"])
        sent = sent[(sent["dispatched_at"] < end) & (sent["cleared_at"] > start)]     # runs that can hold an engine in the horizon
        times = np.unique(np.clip(np.concatenate([[start, end], sent["dispatched_at"], sent["cleared_at"]]), start, end))
        middle, duration = (times[:-1] + times[1:]) / 2, np.diff(times)
        idle = (middle[:, None] >= service_from) & (middle[:, None] < service_to) & (holding_runs(sent, engine_ids, middle) < 0)
        readiness, tau = zip(*(readiness_of(mask)[:2] for mask in idle))
        readiness, tau = np.array(readiness), np.vstack(tau)
        finite = np.isfinite(readiness)
        return (np.sum(readiness[finite] * duration[finite]) / np.sum(duration[finite]),
                np.nansum(np.where(np.isfinite(tau), tau, np.nan) * duration[:, None], axis=0) / 3600)

    changed = census[(census[f"rule_engine_{ACTION_WINDOW_S}"] != census["protocol_engine"]) & np.isfinite(census[f"oc_saving_min_{ACTION_WINDOW_S}"])]
    stratum = pd.qcut(changed[f"oc_saving_min_{ACTION_WINDOW_S}"].rank(method="first"), BRANCH_STRATA, labels=False)
    first_dispatch = runs.groupby("incident_id")["dispatched_at"].transform("min")
    fire_incident = dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False)
    received_s = (dispatches["call_received_at"] - DAY_ZERO).dt.total_seconds()
    check_columns = TIME_COLUMNS + ["engine_id"]
    target = BRANCH_DECISIONS // BRANCH_STRATA
    records, cell_change = [], np.zeros(len(cells))
    for level in range(BRANCH_STRATA):
        candidates = changed[stratum == level].sample(frac=1.0, random_state=BRANCH_SEED + level).index[:3 * target]
        counted = 0
        for row in candidates:
            if counted == target:
                break
            moment = protocol_run.at[row, "dispatched_at"]
            end = moment + BRANCH_HORIZON_S
            window = runs[(first_dispatch >= moment - BRANCH_WARMUP_S) & (first_dispatch <= end)]
            decision = {}

            def rule_at_decision(world_, scenario_, run, available, travel, context=None):
                if run.name != row:
                    return closest_engine(world_, scenario_, run, available, travel)
                protocol_engine, chosen, costs = opportunity_cost_choice(world_, scenario_, run, available, travel, ACTION_WINDOW_S)
                decision.update(protocol_engine=protocol_engine, rule_engine=chosen,
                                oc_saving_s=(costs[protocol_engine] - costs[chosen]) * 60 if chosen != protocol_engine else 0.0,
                                extra_travel_s=travel[chosen] - travel[protocol_engine])
                return chosen

            branch_a, branch_b = simulate_window(window, closest_engine), simulate_window(window, rule_at_decision)
            horizon_rows = window.index[(protocol_run.loc[window.index, "dispatched_at"] >= moment).to_numpy()
                                        & (protocol_run.loc[window.index, "dispatched_at"] <= end).to_numpy()]
            a_part, full_part = branch_a.loc[horizon_rows, check_columns], protocol_run.loc[horizon_rows, check_columns]
            readiness_a, cells_a = horizon_readiness(branch_a, moment, end)
            readiness_full, _ = horizon_readiness(protocol_run, moment, end)
            reproduced = bool((a_part["engine_id"].astype(object).fillna("") == full_part["engine_id"].astype(object).fillna("")).all()
                              and np.allclose(a_part[TIME_COLUMNS].to_numpy(dtype=float), full_part[TIME_COLUMNS].to_numpy(dtype=float),
                                              rtol=0.0, atol=1e-6, equal_nan=True)
                              and abs(readiness_a - readiness_full) < 1e-9)
            readiness_b, cells_b = horizon_readiness(branch_b, moment, end)
            later_rows = window.index[fire_incident[window.index].to_numpy() & (received_s[window.index] > moment).to_numpy()]
            incident_of = dispatches.loc[later_rows, "incident_id"]
            later = pd.concat({"A": branch_a.loc[later_rows, "arrived_at"].groupby(incident_of).min(),
                               "B": branch_b.loc[later_rows, "arrived_at"].groupby(incident_of).min()}, axis=1).dropna()
            own_rows = window.index[(dispatches.loc[window.index, "incident_id"] == dispatches.at[row, "incident_id"]).to_numpy()]
            is_counted = reproduced and decision.get("rule_engine") not in (None, decision.get("protocol_engine"))
            counted += is_counted
            if is_counted:
                cell_change += cells_a - cells_b
            records.append({"row": row, "stratum": level, "dispatched_at": DAY_ZERO + pd.Timedelta(seconds=moment),
                            "week": int(moment // (7 * 86400)), "engine_A": decision.get("protocol_engine"), "engine_B": decision.get("rule_engine"),
                            "census_oc_saving_s": changed.at[row, f"oc_saving_min_{ACTION_WINDOW_S}"] * 60,
                            "oc_saving_s": decision.get("oc_saving_s"), "extra_travel_s": decision.get("extra_travel_s"),
                            "workload_min": (branch_a.at[row, "cleared_at"] - branch_a.at[row, "dispatched_at"]) / 60,
                            "benefit_s": (readiness_a - readiness_b) * 60, "later_fire_calls": len(later),
                            "later_fire_difference_s": (later["B"] - later["A"]).sum(),
                            "own_call_difference_s": branch_b.loc[own_rows, "arrived_at"].min() - branch_a.loc[own_rows, "arrived_at"].min(),
                            "branch_A_reproduces_protocol_run": reproduced, "counted": is_counted})
    branches = pd.DataFrame(records).set_index("row")
    branch_cells = cells[["cell_x", "cell_y", "lon", "lat"]].assign(readiness_change_min_h=cell_change)
    if verbose:
        used = branches[branches["counted"]]
        design = np.column_stack([np.ones(len(used)), used[["oc_saving_s", "extra_travel_s", "workload_min"]].to_numpy(dtype=float)])
        benefit = used["benefit_s"].to_numpy(dtype=float)
        slope = np.linalg.lstsq(design, benefit, rcond=None)[0][1]
        draws = [np.linalg.lstsq(design[pick], benefit[pick], rcond=None)[0][1] for pick in weekly_block_resamples(used["week"].to_numpy())]
        low, high = np.quantile(draws, BOOTSTRAP_INTERVAL)
        strata = used.groupby("stratum").agg(decisions=("benefit_s", "size"), oc_saving_s=("oc_saving_s", "mean"),
                                             benefit_s=("benefit_s", "mean"), extra_travel_s=("extra_travel_s", "mean"),
                                             workload_min=("workload_min", "mean"), own_call_difference_s=("own_call_difference_s", "mean"))
        print(f"--- paired branches: {len(branches)} decisions tried, {len(used)} counted; branch A reproduced the full-year run in "
              f"{branches['branch_A_reproduces_protocol_run'].mean():.1%}; the rule still changed the engine in the reconstructed state in "
              f"{(branches['engine_B'] != branches['engine_A']).mean():.1%}\n"
              f"mean readiness gained over {BRANCH_HORIZON_S / 3600:.0f} h by the rule's engine: {used['benefit_s'].mean():.2f} s; "
              f"slope on the opportunity cost saved (holding extra travel and workload): {slope:.3f} "
              f"(95% weekly block-bootstrap interval {low:.3f} to {high:.3f})\n"
              f"by stratum of opportunity cost saved (census):\n{strata.round(2).to_string()}\n"
              f"later study fire calls within the horizon: {int(used['later_fire_calls'].sum())}, first arrival B minus A summed: "
              f"{used['later_fire_difference_s'].sum():.1f} s\n"
              f"examples:\n{used.head(5).round(2).to_string()}\n")
    return branches, branch_cells


# ----------------------------------------------------------------------------------------------------------------------
# Measures of synthetic scenarios
# ----------------------------------------------------------------------------------------------------------------------

def measure_run(world, scenario, run):
    """Measure a scenario's run: its first-due check, its census at ACTION_WINDOW_S, and its fire readiness with waits.

    Input
        world    : the dict from build_world
        scenario : the scenario that was simulated (dict)
        run      : the simulated run (DataFrame from simulate)
    Output
        Three tables: the first-due check (check_first_due), the census (census_decisions) and the readiness segments
        (fire_readiness with wait_for_busy).
    Example
        first_due, census, segments = measure_run(world, scenario, protocol_run)
    """
    return (check_first_due(scenario["dispatches"], run, scenario["travel"], world["engines"], verbose=False),
            census_decisions(world, scenario, run, [ACTION_WINDOW_S], verbose=False),
            fire_readiness(world, run, wait_for_busy=True, verbose=False)[0])


def check_scenario_conditions(scenario, simulated, verbose=VERBOSE):
    """Check what a scenario's outages and large fires did in a simulated run.

    For outages, the function counts the runs of real calls sent to an engine during its scheduled outage, from the
    stand-in's scheduled start to its end; this count must be 0. It also counts the stand-ins that waited for their
    engine to come back from a call. For large fires, it reports for each fire the engines that reached it, the
    minutes from the call to the first and the last arrival, and the hours on scene.

    Input
        scenario  : the scenario (dict with dispatches, runs and large_fire)
        simulated : its simulated run (DataFrame)
        verbose   : whether to print the checks and three large fires (bool)
    Output
        Two tables: checks (Series) and large_fires (DataFrame, one row per large fire).
    Example
        checks, large_fires = check_scenario_conditions(scenario, protocol_run)
        checks["real runs sent to an engine out of service"] -> 0
    """
    table = scenario["dispatches"]
    stand_in = table["protocol"].eq(OUT_OF_SERVICE_PROTOCOL).fillna(False).to_numpy(dtype=bool)
    real_runs = simulated[~stand_in].dropna(subset=["engine_id", "dispatched_at"])
    sent_while_out = 0
    for row in np.flatnonzero(stand_in):
        scheduled_from, scheduled_to = scenario["runs"].at[row, "dispatched_at"], scenario["runs"].at[row, "cleared_at"]
        own = real_runs[real_runs["engine_id"] == scenario["runs"].at[row, "engine_id"]]
        sent_while_out += int(((own["dispatched_at"] >= scheduled_from) & (own["dispatched_at"] < scheduled_to)).sum())
    waited_min = (simulated.loc[stand_in, "dispatched_at"] - scenario["runs"].loc[stand_in, "dispatched_at"]) / 60
    large = scenario["large_fire"].to_numpy(dtype=bool)
    received_s = scenario["runs"].loc[large, "call_received_at"]
    large_fires = simulated[large].assign(incident_id=table.loc[large, "incident_id"].astype(object).to_numpy(), received_s=received_s).groupby("incident_id").agg(
        engines_on_scene=("arrived_at", "count"), engines=("engine_id", lambda engine: ", ".join(engine.dropna())),
        first_arrival_min=("arrived_at", "min"), last_arrival_min=("arrived_at", "max"), received_s=("received_s", "first"),
        on_scene_h=("cleared_at", "max"))
    if len(large_fires):
        large_fires["on_scene_h"] = (large_fires["on_scene_h"] - large_fires["last_arrival_min"]) / 3600
        for column in ["first_arrival_min", "last_arrival_min"]:
            large_fires[column] = (large_fires[column] - large_fires["received_s"]) / 60
    checks = pd.Series({"stand-in runs (engine-outages)": int(stand_in.sum()),
                        "real runs sent to an engine out of service": sent_while_out,
                        "stand-ins that waited for their engine (count)": int((waited_min > 0).sum()),
                        "stand-ins that waited: median minutes": waited_min[waited_min > 0].median(),
                        "large fires": len(large_fires),
                        "large fires with every engine on scene": int((large_fires["engines_on_scene"] == LARGE_FIRE_ENGINES).sum()) if len(large_fires) else 0})
    if verbose:
        print(f"--- what the outages and large fires did\n{checks.round(2).to_string()}\n"
              + (f"large fires (first three):\n{large_fires.head(3).round(2).to_string()}\n" if len(large_fires) else ""))
    return checks, large_fires


def weekly_calibration_parts(dispatches_table, runs_table, first_due, census, segments, top_cells, days):
    """Split each statistic that nominal scenarios must reproduce into a weekly numerator and denominator.

    The statistics are: other calls per day, study fire calls per week, runs per call, the share of other calls in the
    record's 20 busiest cells, engine busy hours per day, the room on low-acuity calls (decisions per day, the share with
    a choice at ACTION_WINDOW_S, the share where the rule would send another engine, and the readiness saved and the
    extra travel when the rule would send another engine), fire calls whose first-due station had no free engine,
    minutes to the first engine at fire calls, and the time-weighted fire readiness loss. Outage stand-ins are left
    out.

    Input
        dispatches_table : the dispatch table of the version (DataFrame; the record, or a scenario's table)
        runs_table       : its protocol run (DataFrame)
        first_due        : its first-due check (DataFrame)
        census           : its census (DataFrame)
        segments         : its readiness segments with waits (DataFrame)
        top_cells        : the record's 20 busiest cells for other calls (MultiIndex of (cell_x, cell_y))
        days             : the days to cover, as days since DAY_ZERO (range)
    Output
        One row per week (DataFrame). Each column is a pair: the statistic, then "numerator" or "denominator".
    Example
        parts = weekly_calibration_parts(dispatches, protocol_run, first_due_protocol, census, record_segments, top_cells, range(182))
        parts[("study fire calls per week", "numerator")].sum() -> 960
    """
    weeks = pd.Index(sorted({day // 7 for day in days}), name="week")
    days_in_week = pd.Series([day // 7 for day in days]).value_counts().reindex(weeks).astype(float)
    start_s, end_s = min(days) * 86400, (max(days) + 1) * 86400
    received_s = (dispatches_table["call_received_at"] - DAY_ZERO).dt.total_seconds()
    real = (~dispatches_table["protocol"].eq(OUT_OF_SERVICE_PROTOCOL).fillna(False) & dispatches_table["incident_lon"].notna()
            & (received_s >= start_s) & (received_s < end_s))
    calls = dispatches_table[real].groupby("incident_id").agg(received_s=("call_received_at", "first"), protocol=("protocol", "first"),
                                                              lon=("incident_lon", "first"), lat=("incident_lat", "first"), runs=("engine_id", "size"))
    calls["week"] = (calls["received_s"] - DAY_ZERO).dt.days // 7
    fire = calls["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False)
    x, y = pyproj.Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform(calls["lon"].to_numpy(), calls["lat"].to_numpy())
    in_top = pd.Series(pd.MultiIndex.from_arrays([np.floor(x / READINESS_CELL_M).astype(int), np.floor(y / READINESS_CELL_M).astype(int)]).isin(top_cells),
                       index=calls.index)
    sent = runs_table[real.to_numpy()].dropna(subset=["engine_id", "dispatched_at", "cleared_at"])
    census_week = (census["dispatched_at"] - DAY_ZERO).dt.days // 7
    census = census[(census_week >= weeks.min()) & (census_week <= weeks.max())]
    census_week = census_week[census.index]
    switched = census[f"rule_engine_{ACTION_WINDOW_S}"] != census["protocol_engine"]
    saving = census[f"oc_saving_min_{ACTION_WINDOW_S}"]
    counted_saving = switched & np.isfinite(saving)
    fire_calls = first_due[(first_due["week"] >= weeks.min()) & (first_due["week"] <= weeks.max())]
    reached = fire_calls.dropna(subset=["minutes_to_first_on_scene"])
    segments = segments[(segments["start_s"] >= start_s) & (segments["start_s"] < end_s) & np.isfinite(segments["loss_min"])]
    duration = segments["end_s"] - segments["start_s"]

    def per_week(values, week):
        return pd.Series(values, dtype=float).groupby(np.asarray(week)).sum().reindex(weeks, fill_value=0.0)

    one = pd.Series(1.0, index=weeks)
    statistics = {
        "other calls per day": (per_week((~fire).astype(float), calls["week"]), days_in_week),
        "study fire calls per week": (per_week(fire.astype(float), calls["week"]), one),
        "runs per call": (per_week(calls["runs"].astype(float), calls["week"]), per_week(np.ones(len(calls)), calls["week"])),
        "share of other calls in the record's 20 busiest cells": (per_week((~fire & in_top).astype(float), calls["week"]),
                                                                  per_week((~fire).astype(float), calls["week"])),
        "engine busy hours per day": (per_week(((sent["cleared_at"] - sent["dispatched_at"]) / 3600).to_numpy(), sent["dispatched_at"] // (7 * 86400)),
                                      days_in_week),
        "low-acuity decisions per day": (per_week(np.ones(len(census)), census_week), days_in_week),
        f"low-acuity decisions with a choice ({ACTION_WINDOW_S} s)": (per_week((census[f"choices_{ACTION_WINDOW_S}"] >= 2).astype(float), census_week),
                                                                       per_week(np.ones(len(census)), census_week)),
        "rule would send another engine (share)": (per_week(switched.astype(float), census_week), per_week(np.ones(len(census)), census_week)),
        "readiness saved when it would (s, mean)": (per_week(np.where(counted_saving, 60 * saving.replace(np.inf, 0.0), 0.0), census_week),
                                                    per_week(counted_saving.astype(float), census_week)),
        "extra travel when it would (s, mean)": (per_week(np.where(switched, census[f"extra_travel_s_{ACTION_WINDOW_S}"], 0.0), census_week),
                                                 per_week(switched.astype(float), census_week)),
        "fire calls whose first-due station had no free engine": (per_week(fire_calls["committed"].astype(float), fire_calls["week"]),
                                                                  per_week(np.ones(len(fire_calls)), fire_calls["week"])),
        "minutes to the first engine at fire calls (mean)": (per_week(reached["minutes_to_first_on_scene"].to_numpy(), reached["week"]),
                                                             per_week(np.ones(len(reached)), reached["week"])),
        "fire readiness loss (s, time-weighted)": (per_week((60 * segments["loss_min"] * duration).to_numpy(), segments["start_s"] // (7 * 86400)),
                                                   per_week(duration.to_numpy(), segments["start_s"] // (7 * 86400)))}
    return pd.concat({(name, part): series for name, pair in statistics.items() for part, series in zip(["numerator", "denominator"], pair)}, axis=1)


def compare_scenarios_with_record(record_parts, scenario_parts, verbose=VERBOSE):
    """Check the nominal scenarios against the record, statistic by statistic.

    For each statistic of weekly_calibration_parts, the record gives its value and weekly block interval. The nominal
    scenarios give the value pooled over all their weeks and the value per seed. A statistic passes when the pooled
    value falls inside the record's interval. The spread of the weekly values (standard deviation) is shown for both.

    Input
        record_parts   : the record's parts (DataFrame from weekly_calibration_parts)
        scenario_parts : the parts of each nominal scenario, keyed by seed (dict of DataFrames)
        verbose        : whether to print the table and the count inside the interval (bool)
    Output
        One row per statistic (DataFrame): record, interval low, interval high, scenarios, inside record interval, one
        column per seed, record weekly SD and scenario weekly SD.
    Example
        compare_scenarios_with_record(record_parts, scenario_parts).loc["study fire calls per week", "inside record interval"] -> True
    """
    rows = {}
    for name in record_parts.columns.get_level_values(0).unique():
        numerator, denominator = record_parts[(name, "numerator")], record_parts[(name, "denominator")]
        low, high = weekly_block_interval(numerator, pd.Series(record_parts.index, index=record_parts.index), denominator)
        pooled = (sum(parts[(name, "numerator")].sum() for parts in scenario_parts.values())
                  / sum(parts[(name, "denominator")].sum() for parts in scenario_parts.values()))
        weekly_scenarios = pd.concat([parts[(name, "numerator")] / parts[(name, "denominator")] for parts in scenario_parts.values()])
        rows[name] = {"record": numerator.sum() / denominator.sum(), "interval low": low, "interval high": high, "scenarios": pooled,
                      "inside record interval": bool(low <= pooled <= high),
                      **{f"seed {seed}": parts[(name, "numerator")].sum() / parts[(name, "denominator")].sum() for seed, parts in scenario_parts.items()},
                      "record weekly SD": (numerator / denominator).std(), "scenario weekly SD": weekly_scenarios.std()}
    calibration = pd.DataFrame(rows).T
    if verbose:
        print(f"--- nominal scenarios against the record, October-March (pooled over {len(scenario_parts)} seeds; "
              f"{int(100 * (BOOTSTRAP_INTERVAL[1] - BOOTSTRAP_INTERVAL[0]))}% weekly block-bootstrap interval of the record)\n"
              f"{calibration.to_string(float_format=lambda value: f'{value:.3f}')}\n"
              f"statistics inside the record's interval: {int(calibration['inside record interval'].sum())} of {len(calibration)}\n")
    return calibration


def describe_room(dispatches_table, census, first_due, segments, days):
    """Describe the room for another engine under one condition, from its protocol run.

    Input
        dispatches_table : the condition's dispatch table (DataFrame)
        census           : the census on its protocol run (DataFrame)
        first_due        : the first-due check on its protocol run (DataFrame)
        segments         : the readiness segments of its protocol run (DataFrame)
        days             : the days covered, as days since DAY_ZERO (range)
    Output
        One value per statistic (Series): other calls per day, study fire calls per week, low-acuity decisions per day,
        the share with a choice, how often the rule would send another engine and the readiness it would save, the
        count of infinite savings, the share of fire calls with no free first-due engine, the time-weighted readiness
        loss, and the hours with a loss over 1 minute.
    Example
        describe_room(scenario["dispatches"], census, first_due, segments, range(182))["rule would send another engine"]
        -> about 0.115 for a nominal October-March
    """
    start_s, end_s = min(days) * 86400, (max(days) + 1) * 86400
    received_s = (dispatches_table["call_received_at"] - DAY_ZERO).dt.total_seconds()
    real = (~dispatches_table["protocol"].eq(OUT_OF_SERVICE_PROTOCOL).fillna(False) & (received_s >= start_s) & (received_s < end_s))
    calls = dispatches_table[real].groupby("incident_id")["protocol"].first()
    fire = calls.isin(STUDY_FIRE_PROTOCOLS).fillna(False)
    switched = census[f"rule_engine_{ACTION_WINDOW_S}"] != census["protocol_engine"]
    saving = census[f"oc_saving_min_{ACTION_WINDOW_S}"]
    segments = segments[(segments["start_s"] >= start_s) & (segments["start_s"] < end_s)]
    duration = segments["end_s"] - segments["start_s"]
    return pd.Series({
        "other calls per day": (~fire).sum() / len(days), "study fire calls per week": fire.sum() / len(days) * 7,
        "low-acuity decisions per day": len(census) / len(days),
        f"with a choice ({ACTION_WINDOW_S} s)": (census[f"choices_{ACTION_WINDOW_S}"] >= 2).mean(),
        "rule would send another engine": switched.mean(), "  per day": switched.sum() / len(days),
        "  readiness saved (s, mean)": 60 * saving[switched & np.isfinite(saving)].mean(),
        "  saving infinite (the protocol's engine was the last for a cell)": int((switched & ~np.isfinite(saving)).sum()),
        "fire calls with no free first-due engine": first_due["committed"].mean(),
        "fire readiness loss (s, time-weighted)": 60 * np.average(segments["loss_min"], weights=duration),
        "hours with readiness loss over 1 min": duration[segments["loss_min"] > 1].sum() / 3600})


def summarize_rule_by_condition(condition_comparisons, verbose=VERBOSE):
    """Summarize the rule minus protocol differences and their classes under each condition in one table.

    Input
        condition_comparisons : the comparisons from compare_policies, keyed by condition name (dict of DataFrames)
        verbose               : whether to print the table (bool)
    Output
        One row per condition (DataFrame of strings): fire calls, first-due committed, readiness R, high-acuity P90 and
        low-acuity cost, each as "difference (class)"; the low-acuity cost has no class and shows the difference alone.
    Example
        summarize_rule_by_condition(condition_comparisons).loc["nominal", "fire calls"]
        -> a string of the form "<signed difference in seconds, two decimals> (<class>)"
    """
    rule_summary = pd.DataFrame({condition_name: {
        label: f"{comparison.at[outcome, 'difference s']:+.2f} ({comparison.at[outcome, 'class']})" if comparison.at[outcome, "class"] else
               f"{comparison.at[outcome, 'difference s']:+.2f}"
        for label, outcome in [("fire calls", "fire calls, whole year"), ("first-due committed", "fire calls, first-due station committed (protocol)"),
                               ("readiness R", "fire readiness R, weekly hour-weighted mean"), ("high-acuity P90", "high-acuity medical, P90 first arrival"),
                               ("low-acuity cost", "low-acuity calls, mean first arrival (cost)")]}
        for condition_name, comparison in condition_comparisons.items()}).T
    if verbose:
        print(f"--- the opportunity-cost rule ({ACTION_WINDOW_S} s) minus the protocol under each condition (seconds; class against m = "
              f"{FIRE_MARGIN_S} s, delta = {MEDICAL_MARGIN_S} s)\n{rule_summary.to_string()}\n")
    return rule_summary


def summarize_surges(scenario_surges, verbose=VERBOSE):
    """Summarize the fire readiness cost of short surges by surge size and first-due area, with weekly block intervals.

    Input
        scenario_surges : one row per surge (DataFrame): extra_calls, first_due_station, day, usual_other_calls_in_disk,
                          readiness_loss_without_surge_s, readiness_cost_s, rule_minus_protocol_s (NaN where the rule
                          was not run)
        verbose         : whether to print the totals and the areas where a middle-sized surge costs most (bool)
    Output
        One row per surge size and area, including "all areas" (DataFrame): surges, usual other calls in the disk, the
        loss without the surge, the readiness cost with its interval, and the rule minus protocol difference with its
        interval.
    Example
        summarize_surges(scenario_surges).loc[(12, "all areas"), "readiness cost (s)"] -> about 20.3
    """
    surge_area_rows = {}
    for count, count_part in scenario_surges.groupby("extra_calls"):
        for area, part in [("all areas", count_part), *count_part.groupby("first_due_station")]:
            rule_part = part.dropna(subset=["rule_minus_protocol_s"])
            cost_low, cost_high = weekly_block_interval(part["readiness_cost_s"], part["day"] // 7)
            rule_low, rule_high = weekly_block_interval(rule_part["rule_minus_protocol_s"], rule_part["day"] // 7)
            surge_area_rows[(count, area)] = {
                "surges": len(part), "usual other calls in the disk": part["usual_other_calls_in_disk"].mean(),
                "readiness loss without the surge (s)": part["readiness_loss_without_surge_s"].mean(), "readiness cost (s)": part["readiness_cost_s"].mean(),
                "cost interval low": cost_low, "cost interval high": cost_high, "rule minus protocol (s, first seed)": rule_part["rule_minus_protocol_s"].mean(),
                "rule interval low": rule_low, "rule interval high": rule_high}
    scenario_surge_areas = pd.DataFrame(surge_area_rows).T.rename_axis(["extra calls", "first-due area"])
    if verbose:
        counts = sorted(scenario_surges["extra_calls"].unique())
        middle_count = counts[len(counts) // 2]
        print(f"--- short surges: the fire readiness loss while a surge lasts, against the same hours without it, and the rule "
              f"minus the protocol then (first seed)\n"
              f"{scenario_surge_areas.xs('all areas', level='first-due area').round(2).to_string()}\n"
              f"areas where a surge of {middle_count} extra calls costs most:\n"
              f"{scenario_surge_areas.xs(middle_count, level='extra calls').drop(index='all areas').sort_values('readiness cost (s)', ascending=False).head(5).round(2).to_string()}\n")
    return scenario_surge_areas
