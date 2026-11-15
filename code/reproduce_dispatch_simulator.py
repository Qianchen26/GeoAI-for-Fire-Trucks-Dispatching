"""SimPy replay of the FY2025 engine dispatches in Gainesville and Alachua County, built on reproduce_simulator.py.

Lombok -> fire: a damaged house -> a call; a contractor -> an engine; pre-construction delay + construction ->
turnout + travel + on-scene work. Every call arrives at its recorded time; when an engine run starts, one function
lists the engines CAD may send and another picks one.

Section 1 (every engine available, the 2025 engine goes): each run keeps its recorded times (cancelled runs clear
without arriving), so the simulated dispatch table and the response curves equal the 2025 data.
Section 2 (the SOC 2024 rule): the available engine with the shortest estimated travel goes (GFR and ACFR pooled);
an engine other than 2025's gets a travel time from straight-line distance (a curve fitted on 2025 runs). Dispatch,
turnout and on-scene times stay as recorded. Its gains are measured with the same estimate the rule minimises, so
they are an optimistic bound.
Section 3 (road network): ranking and substituted travel come from free-flow times on the county OSM drive network,
calibrated on 2025 runs; the linear form (a + b x network time) beat Kolesar et al. (1975) out of sample.
Section 4 (agency line): an engine of the other agency (GFR vs everyone else) may go only to call types that got
automatic aid in FY2024 (SOC 2024, printed p. 31: "a limited set of emergent calls").
Section 5 (where idle engines are): 5.1 (cases 2 and 3) an engine that cleared a call it reached is on its way home,
a share of the drive given by return_progress; 5.1b fits that share's dwell and slowdown to the AVL positions CAD
logged when it sent engines soon after a call.
Section 6 (M0, research.md Step 1): the step-4 rule with returning engines (5.1b) where no run keeps its 2025 times: the
closest available engine always goes with modeled travel from where it is (ties by engines-table order, not by the 2025
engine); a run with no available engine waits for the next clear, by SOC priority, and no busy engine is ever taken.
E0 checks it against the 2025 record: state checks and first-on-scene curve RMSE for all, multi-engine, away-from-home
and other-agency calls.
Section 7 (M0.1): M0 with the spread of real travel. Engines are still ranked on modeled travel; the chosen engine then
takes modeled travel x a ratio recorded / modeled, drawn from the runs that left from home in Oct-Jun (by area and bin
of modeled travel) at a quantile drawn once per run, the same for every policy. The Jul-Sep runs test the spread; E0
is run again.
Section 8 (M0.2): M0.1, but in calls with two or more runs the call, not the record's row, decides which engines
arrive: the first engines to reach the scene arrive (up to the call's recorded number), and at each recorded
cancellation moment the call releases the engine that would arrive last. In 2025, 43.8% of multi-engine calls had their
first-dispatched run cancelled, and under row fates the closest engine went to that run. E0 is run again.
Sections 7 and 8 also print what the functions did: where each engine was when sent (station or on its way home), the
travel drawn for it, and which runs of a call arrived.
Section 9 (M0.3): M0.2 with engine types. A quint (Q: a pumper that also carries an aerial ladder) is sent only when its
modeled travel beats every available engine's by more than a penalty, which is fitted so that the rule sends CAD's engine
as often as possible on single-engine calls in Oct-Jun (checked on Jul-Sep). The engine sent still takes its own travel.
E0 is run again.
Section 10 (M0.4): M0.3 with two changes, so that the fire outcome research.md compares policies on is right in the
protocol run and every policy works with the same eligible engines. The quint penalty is fitted per call type: elevator
calls (F56, CAD sends the quint), brush fires (F82, CAD almost never sends one) and all other calls. Brush fires draw
their travel ratio from brush-fire runs only (engines that go out with their station's brush truck travel longer). E0
is run again, and the study's fire calls are checked against research.md's margin m with a weekly block bootstrap.
Section 11 (E1, research.md RQ1): how well the protocol protects fire calls, on the frozen M0.4 run beside the 2025
record. (a) For each study fire call: was every engine of its first-due station (the nearest by modeled travel, within
the agency line) committed to another call when it was first dispatched; which calls held them; how long the first
engine took. (b) Fire readiness: between consecutive dispatches and clears, the fire-call-weighted mean modeled travel
from the nearest idle allowed engine to 1 km cells (intensity from the Oct-Mar fire calls), against the same with every
engine idle. Idle engines are placed at their stations.
Section 12 (research.md 3.4): the opportunity-cost rule, on low-acuity calls only (SOC low-priority medical and service
calls): among the engines ranked within ACTION_WINDOW_S of the protocol's, send the one whose departure costs the least
fire readiness. (a) A census on the M0.4 run: how many low-acuity decisions have a choice, how often the rule changes
the engine, what it saves and costs. (b) The rule's year against the protocol's under the same travel draws, for five
seeds and three windows, on research.md's measures with weekly block-bootstrap intervals. (c) Paired branches: 300
decisions, each simulated with the protocol and with the rule from the same rebuilt state, two hours ahead.
Section 13 (research.md 3.3): the scenario generator. Roads, stations and engines stay fixed; calls come from the
October-March record. Other calls: each day copies a whole real day of the same kind (weekday or weekend) within two
weeks, so the calls keep how they fall against each other in time. Study fire calls: per day a gamma day factor (busy
days), per hour a Poisson number of calls at the hour's mean count on the same kind of day within two weeks, each
copying a real study fire call from the same block of hours within two weeks (type, location, fate). (a) Nominal scenarios (four seeds) are
simulated under the protocol and checked against the record on calls, engine time, the room for another engine, the
first-due check and fire readiness (with the wait research.md 3.3 includes when no engine is free). (b) Conditions:
other calls x0.8-1.5 with the extra calls placed by 2020 residents, short surges (6, 12 or 24 extra calls in 2 h
within 2.5 km, visiting every first-due area in turn), engines out of service, a closed station, a large fire each
week; each is shown on concrete cases, and (c) its room is measured and the opportunity-cost rule run against the
protocol. (d) For the surges, the readiness each one costs while it lasts, against the same hours without it, by
first-due area over four seeds, and what the rule changes then.
evaluation_matrix scores every version on the same rows (travel error, same-engine shares, curve RMSE, first on scene).

Inputs:
    data/derived/engine_dispatches_fy2025/engine_dispatches.parquet  one row per engine dispatch (FY2025)
    data/derived/engine_dispatches_fy2025/engines.csv                engine home positions and active span
    data/reports/engine_dispatches_fy2025/response_curves.csv        the 2025 curves the dashboard draws (byte check)
    data/raw/gnv_alachua/alachua_drive.graphml                       OSM drive network (osmnx, free-flow travel time)
    data/raw/gnv_alachua/acfr_gfr_automatic_aid_fy2024.parquet       GFR-ACFR automatic aid dispatches, FY2024
    data/raw/gnv_alachua/census2020_blocks_alachua.gpkg              2020 Census blocks: residents (Section 13)

Outputs (data/reports/reproduce_dispatch_simulator/), sections 2, 3, 4, 5.1, 5.1b, 6, 7, 8, 9, 10 start with protocol_,
network_, boundary_, returning_, returning_avl_, m0_, m01_, m02_, m03_, m04_:
    simulated_dispatches.parquet  the dispatch table rebuilt from the simulation (same 32 columns)
    response_curves.csv           response and completion curves of the simulated calls
    curve_check.csv               per curve: grid points, RMSE and largest |simulated - 2025| share, calls
    protocol_engine_check.csv     share of runs that got the 2025 engine (all, first engine per call, by area)
    protocol_first_on_scene.csv   median and P90 minutes to the first engine on scene, 2025 vs the rule
    network_travel_models.csv     test errors of the two calibration forms and the straight-line curve
    boundary_evaluation_matrix.csv  the evaluation matrix: 2025 record and steps 2, 3, 4
    returning_avl_return_progress.csv  test errors of step 5.1's and the AVL-fitted share of the drive home
    returning_avl_evaluation_matrix.csv  the evaluation matrix with steps 5.1 and 5.1b added
    m0_e0_states.csv              E0 state checks of the M0 run (never served, engine taken, double booking, kept 2025 times, waits)
    m0_e0_strata.csv              E0 first-on-scene curve RMSE by stratum, M0 vs 2025
    m0_evaluation_matrix.csv      the evaluation matrix with step 6 (M0) added
    m01_travel_spread.csv         M0.1: ratio recorded / modeled travel by area and bin of modeled travel (fitting runs)
    m01_travel_spread_check.csv   M0.1: held-out check (Jul-Sep), travel quantiles and shares outside the predicted P10-P90
    m01_e0_states.csv, m01_e0_strata.csv, m01_evaluation_matrix.csv  E0 for the M0.1 run; the matrix with step 7 added
    m02_call_fates.csv            M0.2: per run of a multi-run call, its fate in 2025 and in the simulation, engine, minutes
    m02_e0_states.csv, m02_e0_strata.csv, m02_evaluation_matrix.csv  E0 for the M0.2 run; the matrix with step 8 added
    m03_quint_penalty_fit.csv     M0.3: share of runs where the rule sends CAD's engine, by quint penalty (fit / test months)
    m03_e0_states.csv, m03_e0_strata.csv, m03_evaluation_matrix.csv  E0 for the M0.3 run; the matrix with step 9 added
    m04_quint_penalty_by_call_type.csv  M0.4: share of runs where the rule sends CAD's engine, by call type and penalty
    m04_brush_travel_spread.csv, m04_brush_travel_spread_check.csv  M0.4: brush fires' own travel spread and its check
    m04_fire_check.csv            M0.4: first-engine arrival at the study's fire calls vs 2025, by code, year and Jul-Sep
    m04_brush_quint_calls.csv     M0.4: share of multi-engine brush fires that got a quint, 2025 vs M0.3 and M0.4
    m04_e0_states.csv, m04_e0_strata.csv, m04_evaluation_matrix.csv  E0 for the M0.4 run; the matrix with step 10 added
    e1_first_due_calls.csv        E1: per study fire call and version (record, M0.4 run): first-due station, committed, held by
    e1_first_due_summary.csv, e1_first_due_by_station.csv  E1: shares with weekly block-bootstrap intervals; per station
    e1_readiness_summary.csv, e1_readiness_cells.csv  E1: fire readiness (year, month, hour) and per 1 km cell, per version
    e1_readiness_segments.parquet E1: readiness per segment between dispatches and clears, per version
    oc_census.csv                 Section 12: per low-acuity decision on the M0.4 run: choices, rule's engine, saving, extra travel
    oc_policy_comparison.csv, oc_over_seeds.csv  rule minus protocol per seed and window, and over the five seeds
    oc_rule_states.csv            state checks of the rule's year
    oc_branches.csv, oc_branch_cells.csv  paired branches per decision, and the readiness change per 1 km cell
    scenario_calibration.csv      Section 13: nominal scenarios against the record (October-March), statistic by statistic
    scenario_room.csv             Section 13: the room for another engine under each condition (protocol runs)
    scenario_policy_comparison.csv  Section 13: the opportunity-cost rule minus the protocol under each condition
    scenario_checks.csv           Section 13: state checks of every condition's runs and what its outages and large fires did
    scenario_surges.csv           Section 13.5: one row per short surge (size, seed, area, readiness with and without it, rule change)
    scenario_surge_areas.csv      Section 13.5: the readiness a surge costs by size and first-due area, with weekly block-bootstrap intervals

Run from the project root:
    python code/reproduce_dispatch_simulator.py
"""
import functools
import zlib
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import scipy.interpolate
import scipy.optimize
import scipy.sparse.csgraph
import scipy.spatial
import simpy
from sklearn.metrics import root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]                                     # [DESIGN] this file lives in code/
DISPATCHES_PARQUET = PROJECT_ROOT / "data" / "derived" / "engine_dispatches_fy2025" / "engine_dispatches.parquet"  # [SOURCE]
OBSERVED_CURVES_CSV = PROJECT_ROOT / "data" / "reports" / "engine_dispatches_fy2025" / "response_curves.csv"         # [SOURCE]
ENGINES_CSV = PROJECT_ROOT / "data" / "derived" / "engine_dispatches_fy2025" / "engines.csv"                         # [SOURCE]
ROAD_NETWORK_GRAPHML = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "alachua_drive.graphml"  # [SOURCE] OSM drive network, free-flow speed_kph and travel_time (osmnx)
AUTOMATIC_AID_FY2024 = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "acfr_gfr_automatic_aid_fy2024.parquet"  # [SOURCE] GFR-ACFR automatic aid dispatches, FY2024
TIME_COLUMNS = ["dispatched_at", "enroute_at", "arrived_at", "cleared_at"]  # [SOURCE] CAD times of one engine run, in order
DAY_ZERO = pd.Timestamp("2024-10-01")                                       # [SOURCE] FY2025 starts; simulation clock 0
YEAR_END = pd.Timestamp("2025-10-01")                                       # [SOURCE] FY2025 ends (exclusive)
SERVICE_EDGE_DAYS = 14                                                      # [DESIGN] first/last dispatch this close to the year's edge = in service all year
RESPONSE_GRID_MINUTES = np.round(np.arange(0, 20.0001, 0.1), 1)             # [SOURCE] grid of the 2025 response curves
COMPLETION_GRID_MINUTES = np.arange(0, 241, 1.0)                            # [SOURCE] grid of the 2025 completion curves
AREA_LABELS = {True: "inside city", False: "outside city"}                  # [SOURCE] focal = call inside Gainesville
PRIORITY_LABELS = ["Low", "Moderate", "High", "(missing)"]                  # [SOURCE] 2025 priority groups
OUTPUT_DIR = PROJECT_ROOT / "data" / "reports" / "reproduce_dispatch_simulator"  # [DESIGN]
GEOD = pyproj.Geod(ellps="WGS84")                                           # [STANDARD] distances on the WGS84 ellipsoid
DISTANCE_BANDS_KM = [0, 1, 2, 4, 8, 16, 60]                                 # [DESIGN] bands for checking the travel estimate
TRAVEL_CURVE_BINS = 40                                                      # [DESIGN] equal-count distance bins of the travel curve
QUICKEST_TIE_SECONDS = 10                                                   # [DESIGN] estimates this close to the quickest count as equally quick
SNAP_WARN_M = 500                                                           # [DESIGN] a call this far from the road network is reported
TRAVEL_TEST_FROM = pd.Timestamp("2025-07-01")                               # [DESIGN] travel models fit on Oct-Jun runs, tested on Jul-Sep
RETURN_CHECK_SECONDS = 3600                                                 # [DESIGN] an engine cleared longer ago than this is home (no drive is longer)
NODES_KEPT = 512                                                            # [DESIGN] shortest-path arrays kept in memory, one per scene node
RETURN_CORRIDOR_FACTOR = 1.5                                                # [DESIGN] scene -> engine -> home within this x scene -> home = on the way home
RETURN_MIN_KM = 0.3                                                         # [DESIGN] scenes this close to home give no usable share of the way
RETURN_MINUTE_BANDS = [0, 1, 2, 4, 8, 60]                                   # [DESIGN] minutes-since-clear bands for checking the return progress
PRIORITY_RANK = {"High": 0, "Moderate": 1, "Low": 2}                        # [SOURCE] SOC 2024 Appendix E risk levels; waiting runs are served in this order, missing last
E0_RMSE_LIMIT = 0.03                                                        # [DESIGN] research.md E0: first-on-scene curve RMSE at most this
TRAVEL_SPREAD_BINS = 8                                                      # [DESIGN] M0.1: equal-count bins of modeled travel per area (inside / outside city); about 1,000 fitting runs each
TRAVEL_SPREAD_SEED = 20241001                                               # [DESIGN] M0.1: seed of each run's travel quantile, drawn once, so every policy sees the same draw
QUINT_PENALTY_GRID_S = np.arange(0, 301, 10)                                # [DESIGN] M0.3: candidate extra seconds a quint must save over an engine before it is sent
STUDY_FIRE_PROTOCOLS = ["F67", "F69", "F71", "F82"]                         # [SOURCE] research.md fire calls: outside, structure, vehicle, brush fires (348 calls Jul-Sep, 317 reached)
QUINT_PENALTY_BY_TYPE_GRID_S = np.arange(-300, 301, 10)                     # [DESIGN] M0.4: candidate quint penalties per call type; negative = CAD prefers the quint
BRUSH_FIRE_PROTOCOL = "F82"                                                 # [SOURCE] MPDS fire protocol 82: vegetation, wildland, brush or grass fire
QUINT_RULE_CALL_TYPES = {"F56": "F56 elevator", BRUSH_FIRE_PROTOCOL: "F82 brush fire"}  # [SOURCE] 2025 record: CAD sent the quint to 98% of elevator calls at stations with both types, and a quint to 5-10% of multi-engine brush fires (M0.3: 24-36%); every other call type shares one penalty
BRUSH_SPREAD_BINS = 2                                                       # [DESIGN] M0.4: bins of modeled travel per area for brush fires' own travel spread (58-97 fitting runs each)
FIRE_MARGIN_S = 10                                                          # [SOURCE] research.md: threshold m (tentative 10 s) for a practical difference in fire arrival
BLOCK_BOOTSTRAP_DRAWS = 2000                                                # [DESIGN] weekly block-bootstrap draws for the interval of a mean arrival error
BLOCK_BOOTSTRAP_SEED = 20250701                                             # [DESIGN] seed of the weekly block bootstrap
BOOTSTRAP_INTERVAL = (0.025, 0.975)                                         # [STANDARD] two-sided 95% interval
QUINT_SHARE_TOLERANCE = 0.05                                                # [DESIGN] M0.4 done-criterion: share of multi-engine brush fires sent a quint within 5 points of 2025 (sampling SE about 2 points on 179 calls)
TRAINING_MONTHS_END = pd.Timestamp("2025-04-01")                           # [SOURCE] research.md 3.3: training uses October to March (fire-call intensity for readiness)
READINESS_CELL_M = 1000                                                     # [DESIGN] E1: side of the square cells of fire-call intensity and readiness, metres
PROJECTED_CRS = "EPSG:32617"                                                # [STANDARD] WGS 84 / UTM zone 17N: metres over Alachua County
READINESS_CHUNK = 2000                                                      # [DESIGN] E1: time segments per vectorized chunk of the readiness computation
LOW_ACUITY_SERVICE_PROTOCOLS = ["F53"]                                      # [SOURCE] research.md 3.2: low-acuity = SOC low-priority medical codes or a service call (F53 citizen assist)
ACTION_WINDOW_S = 60                                                        # [DESIGN] research.md 3.2 Delta-a (provisional): an engine ranked this close to the protocol's may be sent instead on a low-acuity call
ACTION_WINDOWS_S = [30, ACTION_WINDOW_S, 120]                               # [DESIGN] Section 12: action windows counted in the census and tried in the year (sensitivity)
MEDICAL_MARGIN_S = 10                                                       # [SOURCE] research.md: delta (provisional 10 s), the allowed rise in the high-acuity medical P90
HIGH_ACUITY_LEVELS = ["C", "D", "E"]                                        # [SOURCE] research.md 3.2: Charlie, Delta and Echo medical calls always get the closest engine
HOTSPOT_STATIONS = ["GFR_03", "CO_21", "CO_16"]                             # [SOURCE] E1: first-due stations with 48% of the committed fire calls (pre-specified subgroup)
DAYTIME_HOURS = range(10, 21)                                               # [SOURCE] E1: hours whose readiness loss exceeds the yearly mean (pre-specified subgroup)
POLICY_SEEDS = [TRAVEL_SPREAD_SEED, 1, 2, 3, 4]                             # [DESIGN] Section 12.3: travel-draw seeds shared by both policies (the M0.4 seed first)
BRANCH_DECISIONS = 300                                                      # [DESIGN] Section 12.4: sampled decisions for paired branches
BRANCH_STRATA = 5                                                           # [DESIGN] Section 12.4: equal-count strata of opportunity cost saved, sampled evenly
BRANCH_HORIZON_S = 7200                                                     # [DESIGN] Section 12.4: branch length after the decision (about one engine outing and return)
BRANCH_WARMUP_S = 24 * 3600                                                 # [DESIGN] Section 12.4: calls simulated before the decision to rebuild the engines' state (checked against the full run)
BRANCH_SEED = 20250101                                                      # [DESIGN] Section 12.4: seed of the decision sample
OC_TIE_S = 1.0                                                              # [DESIGN] Section 12: the rule changes the engine only if it saves more than this much fire readiness (s); co-located engines differ by a few metres
CENSUS_BLOCKS_GPKG = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "census2020_blocks_alachua.gpkg"  # [SOURCE] 2020 Census blocks: residents (POP100) and internal points
SCENARIO_WINDOW_DAYS = 14                                                   # [DESIGN] Section 13: a scenario day's call rates and drawn calls come from training days this close to it
SCENARIO_BAND_HOURS = 4                                                     # [DESIGN] Section 13: a new call copies a real call received in the same block of this many hours
SCENARIO_SEEDS = [101, 102, 103, 104]                                       # [DESIGN] Section 13.2: nominal scenarios, each a whole October-March, checked against the record
STRUCTURE_FIRE_PROTOCOL = "F69"                                             # [SOURCE] MPDS fire protocol 69: structure fire (where a large fire is placed)
DEMAND_FACTORS = [0.8, 1.1, 1.3, 1.5]                                       # [SOURCE] research.md 3.3 (non-fire demand x0.8-1.3) and the author (2026-11-14: up to x1.5, about ten years of 2012-22 growth)
SURGES_PER_DAY = 1                                                          # [DESIGN] Section 13: short surges per day in the surge condition
SURGE_RADIUS_M = 2500                                                       # [DESIGN] Section 13: a short surge's calls fall within this distance of its centre (the record's bursts were counted the same way)
SURGE_HOURS = 2                                                             # [DESIGN] Section 13: length of a short surge (the record's bursts were counted over 2 h)
SURGE_EXTRA_CALLS = [6, 12, 24]                                             # [DESIGN] the author (2026-11-14): a stress sweep; 6 + the usual ~1 = the record's largest local burst (7 other calls within 2.5 km in 2 h), then double and quadruple
OUTAGE_ENGINE_COUNTS = [1, 2, 3]                                            # [PLACEHOLDER] Section 13: engines out of service each day; the records hold no outages, so this is a stress sweep
LARGE_FIRES_PER_WEEK = 1                                                    # [DESIGN] Section 13: large fires per week in the large-fire condition
LARGE_FIRE_ENGINES = 6                                                      # [PLACEHOLDER] Section 13: engines a large fire holds: a first alarm's 3 engine-class units (ACFR response matrix), doubled for a second alarm; to confirm with GFR/ACFR
LARGE_FIRE_ON_SCENE_S = 4 * 3600                                            # [SOURCE] Section 13: the median, over the 8 largest Oct-Mar fires (engine-hours), of their longest single-engine dispatch-to-clear time (4.0 h)
OUT_OF_SERVICE_PROTOCOL = "OUT"                                             # [DESIGN] Section 13: protocol code of the stand-in run that holds an engine while it is out of service
OUT_OF_SERVICE_WAIT_RANK = -1                                               # [DESIGN] Section 13: a pending outage takes its engine at its next clear, before any waiting call (SOC ranks start at 0)
VERBOSE = True                                                             # [DESIGN] print every input and output table


def show_table(name, table, verbose=VERBOSE):
    """Print a table's shape and first five rows, every column.

    Example: show_table("demo", pd.DataFrame({"a": [1, 2]})) -> None; prints "--- demo (2, 1)" and the rows
    """
    if verbose:
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(f"--- {name} {table.shape}\n{table.head(5).to_string()}\n")


def load_inputs(verbose=VERBOSE):
    """The 2025 dispatch table, one row per engine run.

    Example: load_inputs() -> DataFrame (37194, 32)
    """
    dispatches = pd.read_parquet(DISPATCHES_PARQUET)
    show_table("2025 engine dispatches (input)", dispatches, verbose)
    if verbose:
        print(f"incidents {dispatches.incident_id.nunique():,}, engines {dispatches.engine_id.nunique()}, "
              f"time columns {dispatches[TIME_COLUMNS].dtypes.astype(str).to_dict()}")
    return dispatches


def build_runs(dispatches, verbose=VERBOSE):
    """Replay input: the call time and the CAD times of each engine run, in seconds since DAY_ZERO.

    Example: build_runs(dispatches).loc[0, ["engine_id", "call_received_at", "dispatched_at"]].tolist()
             -> ['E62', 6775.0, 6796.0]  # 2024-10-01 01:52:55 and 01:53:16
    """
    runs = dispatches[["incident_id", "engine_id"]].copy()
    for column in ["call_received_at"] + TIME_COLUMNS:
        runs[column] = (dispatches[column] - DAY_ZERO).dt.total_seconds()
    show_table(f"engine runs, seconds since {DAY_ZERO}", runs, verbose)
    if verbose:
        stamps = runs[TIME_COLUMNS]
        gaps = stamps.ffill(axis=1).diff(axis=1).iloc[:, 1:]                  # between consecutive recorded times
        order = runs.sort_values(["engine_id", "dispatched_at"])
        previous_clear = order.groupby("engine_id")["cleared_at"].shift()
        print("all times whole seconds:", bool(((stamps % 1 == 0) | stamps.isna()).all().all()))
        print("times recorded (1 = present, order dispatched/enroute/arrived/cleared):",
              stamps.notna().astype(int).astype(str).agg("".join, axis=1).value_counts().to_dict())
        print("a later time before an earlier one:", int((gaps < 0).any(axis=1).sum()), "runs")
        print("dispatched before the call:", int((runs.dispatched_at < runs.call_received_at).sum()), "runs")
        print("engine dispatched again before its previous run cleared:", int((order.dispatched_at < previous_clear).sum()), "runs")
    return runs


def choose_engines(call_runs):
    """Which engine runs a call gets. This version: the engines that went in 2025.

    Example: choose_engines([0]) -> [0]  # row labels in the runs table
    """
    return call_runs


def all_engines_available(engine_pool, now_seconds, engines):
    """Section 1 rule for availability: every engine counts as available, whatever it is doing.

    Example: all_engines_available({"E1": ..., "E62": ...}, 6796.0, None) -> ['E1', 'E62']
    """
    return list(engine_pool)


def dispatched_in_2025(run, available_engines, travel_for_run):
    """Section 1 rule for the choice: the engine that went in 2025, keeping its recorded times (travel None).

    Example: dispatched_in_2025(runs.loc[0], ["E1", "E62"], None) -> ('E62', None)
    """
    return run["engine_id"], None


def shift_arrival(times, travel_seconds):
    """A run's CAD times with arrival and clear moved so that travel takes travel_seconds; a run with no en-route
    or no arrival time (cancelled) keeps its times.

    Example: shift_arrival(pd.Series({"dispatched_at": 0.0, "enroute_at": 60.0, "arrived_at": 400.0, "cleared_at": 900.0}),
                           300.0).tolist() -> [0.0, 60.0, 360.0, 860.0]
    """
    if pd.isna(times["enroute_at"]) or pd.isna(times["arrived_at"]):
        return times
    shift = times["enroute_at"] + travel_seconds - times["arrived_at"]
    return times + pd.Series({"dispatched_at": 0.0, "enroute_at": 0.0, "arrived_at": shift, "cleared_at": shift})


def simulate(runs, available=all_engines_available, choose=dispatched_in_2025, engines=None, estimated_travel=None,
             where_engines_are=None, wait_rank=None, report=None, counterfactual=False, fates_by_arrival=None,
             verbose=VERBOSE):
    """Replay every call in SimPy; returns per run the simulated CAD times (seconds since day zero) and the engine used.

    Each engine is a one-crew simpy.PreemptiveResource. When a run starts (its first recorded time), `available`
    lists the engines CAD may send and `choose` picks one with a travel time (None keeps the recorded times). The run
    then passes its times in time order (two 2025 runs have them out of order). A newer run may take an engine that is
    still on an older run, as when a returning crew picks up a new call (three 2025 runs); the older run keeps its times.
    The defaults (every engine available, the 2025 engine goes) replay the 2025 log.
    Each engine's last clear (time, and the run whose scene it left; None for a cancelled run) is kept in engine_state;
    where_engines_are, if given, turns the travel table's row into travel from where the engines are now.
    If `choose` returns no engine (None), the run waits and never takes a busy engine: each clear offers the engines to
    the waiting runs in order of wait_rank (lower first; all equal if None), then of waiting time, and a run that gets
    an engine moves all its times by its wait. A new run looks for an engine after the clears of its second, except a
    run with a negative wait_rank (Section 13's scheduled outage), which looks before them, so a clear in its first
    second hands the engine to it ahead of every waiting call. counterfactual=True (M0) adds two rules: a request cancelled in the
    record stops waiting at its recorded release and gets no engine, and an engine released from a cancelled run it
    was driving to turns home from where it was (a share of the way to that call, kept in engine_state for
    where_engines_are).
    fates_by_arrival (M0.2; from call_fates): in the calls it lists (two or more runs), the call, not the record's row,
    decides which runs arrive. Each run is sent at its recorded moment with the chosen engine and its recorded turnout.
    The first engines to reach the scene arrive, up to the call's recorded number of arrivals, and take the recorded
    on-scene times in order. At each recorded cancellation moment the call releases one run: one still waiting for an
    engine, else the engine under way that would arrive last, else (none under way) its next run is not sent. An engine
    that reaches the scene after the call has its arrivals is released there.
    `report`, if a dict, receives the rows that kept their recorded times (kept_recorded), the waits (waited_seconds:
    row -> seconds), the rows whose engine a newer run took (taken_by_newer_run), the rows still waiting at the end
    (never_served), the cancelled requests that never got an engine (cancelled_while_waiting), and every dispatch
    (dispatches: run, time, engine, the engine's last clear in engine_state, its modeled travel from where it was, the
    travel it takes).

    Example: simulate(runs).loc[0].tolist() -> [6796.0, 6858.0, 7255.0, 7404.0, 'E62']  # 4 CAD times, engine
    """
    env = simpy.Environment()
    engine_ids = runs["engine_id"].unique() if engines is None else engines["engine_id"]   # every engine CAD could send
    engine_pool = {engine_id: simpy.PreemptiveResource(env, capacity=1) for engine_id in engine_ids}
    engine_state = {}                                            # engine id -> (last clear, seconds; run left, or None)
    simulated = pd.DataFrame(np.nan, index=runs.index, columns=TIME_COLUMNS)
    simulated["engine_id"] = pd.Series(index=runs.index, dtype="object")
    waiting = []                                                 # runs with no engine yet: (rank, since, row, wake-up event)
    kept_recorded, waited_seconds, taken_by_newer_run_rows, cancelled_while_waiting, dispatch_log = [], {}, [], [], []

    def pick(row):
        travel_for_run = None if estimated_travel is None else estimated_travel.loc[row]   # every engine -> this call, seconds
        if where_engines_are is not None:
            travel_for_run = where_engines_are(row, env.now, engine_state, travel_for_run)
        engine_id, travel_seconds = choose(runs.loc[row], available(engine_pool, env.now, engines), travel_for_run)
        if report is not None and engine_id is not None:         # callers send a chosen engine at once
            dispatch_log.append((row, env.now, engine_id, engine_state.get(engine_id),
                                 np.nan if travel_for_run is None else travel_for_run.get(engine_id, np.nan), travel_seconds))
        return engine_id, travel_seconds

    def serve_waiting():
        # ponytail: waiting runs are offered engines only at clears; an engine entering service mid-wait waits for the next clear
        for item in sorted(waiting, key=lambda item: item[:2]):
            engine_id, travel_seconds = pick(item[2])
            if engine_id is not None:
                waiting.remove(item)                             # the request is granted now, so the next run sees the engine busy
                item[3].succeed((engine_id, travel_seconds, engine_pool[engine_id].request(priority=-env.now)))

    def engine_run(row):
        times = runs.loc[row, TIME_COLUMNS]
        start = times.dropna().min()
        if wait_rank is None or wait_rank[row] >= 0:
            yield env.timeout(start - env.now)
            yield env.timeout(0)                                 # runs clearing in this same second free their engines first
        elif start > env.now:                                    # a scheduled outage (negative rank) starts when its call does, so it
            yield env.timeout(start - env.now)                   # looks at once: a new process runs before this second's clears
        engine_id, travel_seconds = pick(row)
        request = None
        if engine_id is None:                                    # no engine to send: wait for a clear
            wake = env.event()
            item = (0 if wait_rank is None else wait_rank[row], env.now, row, wake)
            waiting.append(item)
            cancelled = counterfactual and pd.isna(times["arrived_at"])   # M0: its release time is exogenous
            if cancelled:                                        # it stops waiting at its recorded release
                yield wake | env.timeout(max(0.0, times["cleared_at"] - env.now))
                if not wake.triggered:
                    waiting.remove(item)
                    cancelled_while_waiting.append(row)
                    simulated.at[row, "cleared_at"] = env.now    # the request ends here, with no engine
                    return
            engine_id, travel_seconds, request = yield wake
            waited_seconds[row] = env.now - start
            release = times["cleared_at"]
            times = times + waited_seconds[row]                  # the run starts when it gets its engine
            if cancelled:                                        # ...but a cancelled run still ends at its recorded release
                times["cleared_at"] = release
                if times["enroute_at"] > release:
                    times["enroute_at"] = np.nan
        if travel_seconds is None:
            kept_recorded.append(row)
        recorded = (times if travel_seconds is None else shift_arrival(times, travel_seconds)).dropna().sort_values()
        simulated.at[row, "engine_id"] = engine_id
        taken_by_newer_run = False
        if request is None:
            request = engine_pool[engine_id].request(priority=-recorded.iloc[0])   # newer run ranks first
        with request as engine:
            yield engine
            for column, at in recorded.items():
                try:
                    yield env.timeout(at - env.now)
                except simpy.Interrupt:                                  # engine taken by a newer run
                    taken_by_newer_run = True
                    yield env.timeout(at - env.now)
                simulated.at[row, column] = env.now
        enroute_s = simulated.at[row, "enroute_at"]
        if taken_by_newer_run:
            taken_by_newer_run_rows.append(row)
        elif counterfactual and "arrived_at" not in recorded.index and travel_seconds and not pd.isna(enroute_s):
            # ponytail: the share of the way out assumes the engine left from home; one that left another scene is placed on its home -> call line
            outbound_share = min(1.0, max(0.0, (env.now - enroute_s) / travel_seconds))
            engine_state[engine_id] = (env.now, row, outbound_share) if outbound_share > 0 else (env.now, None)
        else:
            engine_state[engine_id] = (env.now, row if "arrived_at" in recorded.index else None)
        serve_waiting()

    def call_run(row, call_state, under_way):
        # M0.2: one run of a call whose fates are decided by the call (fates_by_arrival), not by this row of the record
        times = runs.loc[row, TIME_COLUMNS]
        yield env.timeout(times["dispatched_at"] - env.now)
        yield env.timeout(0)                                     # runs clearing in this same second free their engines first
        if call_state["releases_pending"]:                       # a release found no run under way: this run is not sent
            call_state["releases_pending"] -= 1
            cancelled_while_waiting.append(row)
            simulated.loc[row, ["dispatched_at", "cleared_at"]] = env.now
            return
        engine_id, travel_seconds = pick(row)
        request = None
        if engine_id is None:                                    # no engine to send: wait; the call may release the wait
            wake = env.event()
            waiting.append((0 if wait_rank is None else wait_rank[row], env.now, row, wake))
            granted = yield wake
            if granted is None:
                cancelled_while_waiting.append(row)
                simulated.loc[row, ["dispatched_at", "cleared_at"]] = [times["dispatched_at"], env.now]
                return
            engine_id, travel_seconds, request = granted
            waited_seconds[row] = env.now - times["dispatched_at"]
        turnout_s = times["enroute_at"] - times["dispatched_at"]
        simulated.at[row, "engine_id"] = engine_id
        with (request or engine_pool[engine_id].request(priority=-env.now)) as engine:
            yield engine
            simulated.at[row, "dispatched_at"] = env.now
            arrival_s = env.now + turnout_s + travel_seconds
            under_way[row] = (arrival_s, env.active_process)
            try:
                yield env.timeout(turnout_s)
                simulated.at[row, "enroute_at"] = env.now
                yield env.timeout(arrival_s - env.now)
            except simpy.Interrupt as interrupt:                 # released by the call at a recorded cancellation moment
                assert interrupt.cause == "released by the call", interrupt.cause
                enroute_s = simulated.at[row, "enroute_at"]
                outbound_share = 0.0 if pd.isna(enroute_s) else min(1.0, (env.now - enroute_s) / travel_seconds)
                engine_state[engine_id] = (env.now, row, outbound_share) if outbound_share > 0 else (env.now, None)
            else:
                under_way.pop(row, None)
                if call_state["arrived"] < call_state["arrivals"]:   # the call still needs an engine on scene
                    simulated.at[row, "arrived_at"] = env.now
                    on_scene_s = call_state["on_scene_s"][call_state["arrived"]]
                    call_state["arrived"] += 1
                    yield env.timeout(on_scene_s)
                    engine_state[engine_id] = (env.now, row)
                else:                                            # the call has its arrivals: released at the scene
                    engine_state[engine_id] = (env.now, row, 1.0)
            simulated.at[row, "cleared_at"] = env.now
        serve_waiting()

    def release_extra_runs(incident, call_state, under_way):
        # M0.2: at each recorded cancellation moment the call releases one run: one still waiting for an engine, else the
        # engine under way that would arrive last, else (none under way) its next run is not sent
        for moment in call_state["release_moments"]:
            yield env.timeout(max(0.0, moment - env.now))
            queued = [item for item in waiting if runs.at[item[2], "incident_id"] == incident]
            if queued:
                waiting.remove(queued[-1])
                queued[-1][3].succeed(None)
            elif under_way:
                last = max(under_way, key=lambda row: under_way[row][0])
                under_way.pop(last)[1].interrupt("released by the call")
            else:
                call_state["releases_pending"] += 1

    def call(call_runs):
        yield env.timeout(runs.at[call_runs[0], "call_received_at"] - env.now)
        incident = runs.at[call_runs[0], "incident_id"]
        if fates_by_arrival is not None and incident in fates_by_arrival:
            arrivals, release_moments, on_scene_s = fates_by_arrival[incident]
            call_state = {"arrivals": arrivals, "release_moments": release_moments, "on_scene_s": on_scene_s,
                          "arrived": 0, "releases_pending": 0}
            under_way = {}                                       # run -> (when its engine would arrive, its process)
            env.process(release_extra_runs(incident, call_state, under_way))
            for row in choose_engines(call_runs):
                env.process(call_run(row, call_state, under_way))
            return
        for row in choose_engines(call_runs):
            env.process(engine_run(row))

    for call_runs in runs.groupby("incident_id", sort=False).groups.values():
        env.process(call(list(call_runs)))
    env.run()
    show_table("simulated CAD times (seconds since day zero) and engine used", simulated, verbose)
    if report is not None:
        report.update(kept_recorded=kept_recorded, waited_seconds=waited_seconds, taken_by_newer_run=taken_by_newer_run_rows,
                      never_served=[item[2] for item in waiting], cancelled_while_waiting=cancelled_while_waiting,
                      dispatches=pd.DataFrame(dispatch_log, columns=["run", "time_s", "engine_id", "last_clear", "modeled_s",
                                                                     "travel_s"]))
    return simulated


def compare_times(runs, simulated, verbose=VERBOSE):
    """Count runs whose simulated CAD times or engine differ from the recorded ones (missing equals missing).

    Example: compare_times(runs, simulated) -> 0  # int
    """
    columns = TIME_COLUMNS + ["engine_id"]
    same = (simulated[columns] == runs[columns]) | (simulated[columns].isna() & runs[columns].isna())
    differing = int((~same.all(axis=1)).sum())
    if verbose:
        side_by_side = pd.concat({"2025": runs[columns], "simulated": simulated[columns]}, axis=1)
        print(f"{side_by_side.head(5).to_string()}\nruns compared {len(runs):,}; runs with any time or engine different: {differing}\n")
    return differing


def seconds_between(later, earlier):
    """Whole seconds from one time column to another; missing when either is missing (as in the dataset build).

    Example: seconds_between(pd.Series(pd.to_datetime(["2024-10-01 01:54:18"])),
                             pd.Series(pd.to_datetime(["2024-10-01 01:53:16"]))).tolist() -> [62]
    """
    return (later - earlier).dt.total_seconds().round().astype("Int64")


def simulated_dispatch_table(dispatches, simulated, engines=None, verbose=VERBOSE):
    """The dispatch table rebuilt from the simulation: CAD times from SimPy, and from them the durations,
    cancelled flag, arrival order and engines per call, with the dataset's own formulas. Call attributes are inputs
    and are carried over. Where the simulation used another engine than 2025, the engine columns describe that engine
    as sent from its home station (auto_aid unknown).

    Example: simulated_dispatch_table(dispatches, simulated).shape -> (37194, 32)
    """
    table = dispatches.copy()
    for column in TIME_COLUMNS:
        seconds = simulated[column]                              # missing stays missing (NaT) without a pandas cast warning
        table[column] = (DAY_ZERO + pd.to_timedelta(seconds.fillna(0), unit="s").where(seconds.notna())).astype(dispatches[column].dtype)
    no_engine = simulated["engine_id"].isna()
    if no_engine.any():                                          # M0: a request cancelled before any engine was free
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
    table["n_engines_sent"] = table.groupby("incident_id")["engine_id"].transform("count").astype("Int64")  # engines, not requests
    show_table("simulated dispatch table", table, verbose)
    if verbose:
        differing = [column for column in dispatches.columns if not table[column].equals(dispatches[column])]
        print(f"columns that differ from the 2025 table: {differing}; whole table equal: {table.equals(dispatches)}\n")
    return table


def response_curves(table, verbose=VERBOSE):
    """Share of calls that reached each stage within t minutes of their own call, defined as the 2025 curves are.

    The denominator is every call in the group, so calls no engine reached keep a curve below 100%.

    Example: response_curves(table).query("curve == 'stages' and series == 'first_on_scene' and minute == 20.0")
             ["share"].tolist() -> [0.856...]
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
    """Per curve: grid points, points found in both, RMSE and largest |simulated - 2025| share, and the call counts.

    Example: compare_curves(simulated_curves, observed_curves)[["rmse", "max_abs_difference"]].max().tolist() -> [0.0, 0.0]
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


def write_outputs(table, curves, check, prefix="", extra_tables=None):
    """Save the simulated dispatch table, its response curves, the curve check and any extra tables, file names
    starting with `prefix`; say whether the curve file is byte-identical to the 2025 one the dashboard draws."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT_DIR / f"{prefix}simulated_dispatches.parquet", index=False)
    curves.to_csv(OUTPUT_DIR / f"{prefix}response_curves.csv", index=False)
    check.to_csv(OUTPUT_DIR / f"{prefix}curve_check.csv", index=False)
    for name, extra in (extra_tables or {}).items():
        extra.to_csv(OUTPUT_DIR / f"{prefix}{name}.csv")
    same_file = (OUTPUT_DIR / f"{prefix}response_curves.csv").read_bytes() == OBSERVED_CURVES_CSV.read_bytes()
    print(f"wrote {prefix}simulated_dispatches.parquet, {prefix}response_curves.csv, {prefix}curve_check.csv"
          f"{''.join(f', {prefix}{name}.csv' for name in (extra_tables or {}))} to {OUTPUT_DIR}\n"
          f"{prefix}response_curves.csv byte-identical to {OBSERVED_CURVES_CSV.relative_to(PROJECT_ROOT)}: {same_file}\n")


def load_engines(verbose=VERBOSE):
    """The engines: type, agency, home station position, and the span each one is in service. An engine whose first
    (last) FY2025 dispatch falls within SERVICE_EDGE_DAYS of the year's start (end) is in service from the start
    (to the end); otherwise its span starts (ends) at that dispatch.

    Example: load_engines().loc[0, ["engine_id", "home_station", "in_service_from"]].tolist()
             -> ['E1', 'GFR_01', Timestamp('2024-10-01 00:00:00')]
    """
    engines = pd.read_csv(ENGINES_CSV, parse_dates=["first_seen", "last_seen"])
    edge = pd.Timedelta(days=SERVICE_EDGE_DAYS)
    engines["in_service_from"] = engines["first_seen"].where(engines["first_seen"] > DAY_ZERO + edge, DAY_ZERO)
    engines["in_service_to"] = engines["last_seen"].where(engines["last_seen"] < YEAR_END - edge, YEAR_END)
    show_table("engines (input)", engines, verbose)
    if verbose:
        part_year = (engines["in_service_from"] > DAY_ZERO) | (engines["in_service_to"] < YEAR_END)
        print(f"engines {len(engines)}, agencies {engines.engine_agency.value_counts().to_dict()}, "
              f"missing home position {int(engines.home_lon.isna().sum())}, no home station {int(engines.home_station.isna().sum())}\n"
              f"in service for part of the year only:\n"
              f"{engines.loc[part_year, ['engine_id', 'in_service_from', 'in_service_to', 'n_dispatches']].to_string(index=False)}\n")
    return engines


def straight_line_km(lon_from, lat_from, lon_to, lat_to):
    """Geodesic (straight-line) distance in km on the WGS84 ellipsoid; arrays in, array out.

    Example: straight_line_km(-82.3249, 29.6468, -82.3550, 29.6355) -> 3.17 (km)
    """
    return GEOD.inv(lon_from, lat_from, lon_to, lat_to)[2] / 1000


def fit_travel_curve(dispatches, verbose=VERBOSE):
    """Travel time from straight-line distance, fitted on 2025 runs: runs are cut into equal-count distance bins and
    each bin gives (median km, median travel seconds); between bins it is linear, beyond them it extends the end
    slopes. Distance is from the engine's dispatch position (AVL) to the call.

    Example: fit_travel_curve(dispatches)(np.array([1.0, 5.0])) -> array([235., 507.])  # seconds
    """
    usable = dispatches[dispatches.travel_s.gt(0).fillna(False)
                        & ~dispatches.qc_note.str.contains("implausible|equals_incident")
                        & dispatches[["engine_lon_at_dispatch", "incident_lon"]].notna().all(axis=1)]
    distance_km = straight_line_km(usable.engine_lon_at_dispatch.to_numpy(), usable.engine_lat_at_dispatch.to_numpy(),
                                   usable.incident_lon.to_numpy(), usable.incident_lat.to_numpy())
    recorded_seconds = usable.travel_s.to_numpy(dtype=float)
    bins = pd.qcut(distance_km, TRAVEL_CURVE_BINS, labels=False)
    points = pd.DataFrame({"km": distance_km, "seconds": recorded_seconds}).groupby(bins).median()
    points["seconds"] = np.maximum.accumulate(points["seconds"].to_numpy())   # never shorter for a longer distance
    travel_curve = scipy.interpolate.interp1d(points["km"], points["seconds"], fill_value="extrapolate")
    if verbose:
        estimated_minutes = travel_curve(distance_km) / 60
        recorded_minutes = recorded_seconds / 60
        error = pd.DataFrame({"band_km": pd.cut(distance_km, DISTANCE_BANDS_KM), "recorded_min": recorded_minutes,
                              "estimated_min": estimated_minutes, "abs_error_min": np.abs(estimated_minutes - recorded_minutes)})
        by_band = error.groupby("band_km", observed=True)
        print(f"runs used {len(usable):,} of {len(dispatches):,}; curve through {len(points)} bin medians, "
              f"e.g. 1 km -> {travel_curve(1.0) / 60:.1f} min, 5 km -> {travel_curve(5.0) / 60:.1f} min, "
              f"20 km -> {travel_curve(20.0) / 60:.1f} min\n"
              f"estimated vs recorded travel, by straight-line distance (medians):\n"
              f"{by_band.median().assign(runs=by_band.size()).round(2).to_string()}\n"
              f"all runs: median |error| {np.median(error.abs_error_min):.2f} min\n")
    return travel_curve


def estimate_travel_seconds(dispatches, engines, travel_curve, verbose=VERBOSE):
    """Estimated travel seconds from every engine's home station to every run's call (rows: runs, columns: engines);
    missing when the engine is outside its FY2025 service span at the dispatch time or the call has no location.

    Example: estimate_travel_seconds(dispatches, engines, travel_curve).shape -> (37194, 29)
    """
    call_lon, call_lat = dispatches["incident_lon"].to_numpy(), dispatches["incident_lat"].to_numpy()
    located = ~np.isnan(call_lon)
    estimates = pd.DataFrame(np.nan, index=dispatches.index, columns=engines["engine_id"])
    for engine in engines.itertuples():
        km = np.full(len(dispatches), np.nan)
        km[located] = straight_line_km(np.full(located.sum(), engine.home_lon), np.full(located.sum(), engine.home_lat),
                                       call_lon[located], call_lat[located])
        in_service = dispatches["dispatched_at"].between(engine.in_service_from, engine.in_service_to).to_numpy()
        estimates[engine.engine_id] = np.where(in_service, travel_curve(km), np.nan)
    show_table("estimated travel seconds, home station -> call (rows: runs)", estimates.round(0), verbose)
    if verbose:
        rank = estimates.rank(axis=1, method="min")                          # 1 = shortest; same-station engines tie
        rank_of_2025_engine = rank.to_numpy()[np.arange(len(rank)), rank.columns.get_indexer(dispatches["engine_id"])]
        print("rank of the engine that went in 2025 among in-service engines (by estimated travel from home):",
              pd.Series(rank_of_2025_engine).clip(upper=4).map({1: "1st", 2: "2nd", 3: "3rd", 4: "4th or later"})
              .fillna("no estimate").value_counts(normalize=True).round(3).to_dict(), "\n")
    return estimates


def define_available_engines(engine_pool, now_seconds, engines):
    """Section 2 rule for availability: engines in service at this moment (load_engines spans) and not on a run
    in the simulation.

    Example: define_available_engines(engine_pool, 6796.0, engines) -> ['E1', 'E2', 'E4', ...]  # 25 of 29: Q1, Q2 not yet in service; E3, E9 on runs
    """
    now = DAY_ZERO + pd.Timedelta(seconds=now_seconds)
    in_service = engines.loc[engines["in_service_from"].le(now) & engines["in_service_to"].ge(now), "engine_id"]
    return [engine_id for engine_id in in_service if engine_pool[engine_id].count == 0]


def dispatch_by_protocol(run, available_engines, travel_for_run):
    """Section 2 rule for the choice (GFR SOC 2024, printed p. 70 and 146): CAD sends the available engine with the
    shortest estimated travel time, GFR and ACFR pooled. Engines within QUICKEST_TIE_SECONDS of the quickest count as
    equally quick (two engines of one station); if the 2025 engine is among them it goes with its 2025 times. Another
    engine goes with its estimated travel. With no available engine or no call location, the 2025 engine goes with its
    2025 times. travel_for_run: estimated seconds from every engine to this run's call (engine id -> seconds).

    Example: dispatch_by_protocol(runs.loc[0], ["E62", "E3"], estimated_travel.loc[0]) -> ('E62', None)
    """
    candidates = travel_for_run[available_engines].dropna()
    if candidates.empty or candidates.get(run["engine_id"], np.inf) <= candidates.min() + QUICKEST_TIE_SECONDS:
        return run["engine_id"], None
    return candidates.idxmin(), float(candidates.min())


def compare_engines(runs, simulated, dispatches, verbose=VERBOSE):
    """How often the simulation sent the 2025 engine (all runs, the first engine of each call, inside and outside
    the city), how travel changed where it sent another engine, and how many runs found their engine still on an
    earlier run (0 when every choice was an available engine).

    Example: compare_engines(runs, protocol, dispatches).loc["all runs", "same_engine_share"] -> 0.698
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
    """Minutes from the call to the first engine on scene, median and P90 of the calls an engine reached, for all
    calls and inside / outside the city; one column per table, plus the last column minus the first.

    Example: compare_first_on_scene({"2025": dispatches, "protocol": protocol_table}).loc["all calls P90"]
             -> 2025 13.22, protocol 12.30, difference -0.92  # minutes
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


def load_road_network(verbose=VERBOSE):
    """The county drive network, cut to its largest strongly connected part so that every node can reach every other.

    Example: load_road_network().number_of_nodes() -> 15048
    """
    full_graph = ox.load_graphml(ROAD_NETWORK_GRAPHML)
    graph = ox.truncate.largest_component(full_graph, strongly=True)
    if verbose:
        edges = ox.convert.graph_to_gdfs(graph, nodes=False)
        print(f"road network: {full_graph.number_of_nodes():,} nodes, {full_graph.number_of_edges():,} edges; largest strongly "
              f"connected part kept: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges\n"
              f"edge speed_kph (free flow, OSM): median {edges.speed_kph.median():.0f}, P10 {edges.speed_kph.quantile(0.1):.0f}, "
              f"P90 {edges.speed_kph.quantile(0.9):.0f}; edge travel_time missing: {int(edges.travel_time.isna().sum())}\n")
    return graph


def snap_to_network(graph, dispatches, engines, verbose=VERBOSE):
    """Nearest network node to each run's call and to each engine's home station, with the snap distance in metres
    (a call without coordinates gets no node).

    Example: snap_to_network(graph, dispatches, engines)[1].loc[0, ["engine_id", "snap_m"]].tolist() -> ['E1', 12.3...]
    """
    located = dispatches["incident_lon"].notna()
    nodes, meters = ox.distance.nearest_nodes(graph, X=dispatches.loc[located, "incident_lon"].to_numpy(),
                                              Y=dispatches.loc[located, "incident_lat"].to_numpy(), return_dist=True)
    call_nodes = pd.DataFrame({"node": pd.Series(nodes, index=dispatches.index[located], dtype="Int64"),
                               "snap_m": pd.Series(meters, index=dispatches.index[located])}).reindex(dispatches.index)
    nodes, meters = ox.distance.nearest_nodes(graph, X=engines["home_lon"].to_numpy(), Y=engines["home_lat"].to_numpy(),
                                              return_dist=True)
    engine_nodes = pd.DataFrame({"engine_id": engines["engine_id"], "node": pd.Series(nodes, dtype="Int64"), "snap_m": meters})
    show_table("calls snapped to the network (rows: runs)", call_nodes, verbose)
    show_table("engine home stations snapped to the network", engine_nodes, verbose)
    if verbose:
        print(f"call snap distance (m): median {call_nodes.snap_m.median():.0f}, P90 {call_nodes.snap_m.quantile(0.9):.0f}, "
              f"P99 {call_nodes.snap_m.quantile(0.99):.0f}; over {SNAP_WARN_M} m: {int((call_nodes.snap_m > SNAP_WARN_M).sum())} runs; "
              f"no coordinates: {int(call_nodes.node.isna().sum())} runs\n"
              f"engine snap distance (m): median {engine_nodes.snap_m.median():.0f}, max {engine_nodes.snap_m.max():.0f}; "
              f"distinct home nodes: {engine_nodes.node.nunique()} for {len(engine_nodes)} engines\n")
    return call_nodes, engine_nodes


def network_from_stations(graph, dispatches, call_nodes, engine_nodes, verbose=VERBOSE):
    """Free-flow network time (seconds, fastest path) and shortest road distance (km) from every engine's home node to
    every run's call node (rows: runs, columns: engines); missing for a call without coordinates.

    Example: network_from_stations(graph, dispatches, call_nodes, engine_nodes)[0].loc[0, "E62"] -> 287.0 (seconds), roughly
    """
    node_list = list(graph.nodes)
    position = pd.Series(np.arange(len(node_list)), index=node_list)
    sources = engine_nodes["node"].astype("int64").unique()
    from_sources = {}
    for weight in ["travel_time", "length"]:                     # parallel edges keep their smallest weight
        matrix = nx.to_scipy_sparse_array(ox.convert.to_digraph(graph, weight=weight), nodelist=node_list, weight=weight)
        from_sources[weight] = scipy.sparse.csgraph.dijkstra(matrix, directed=True, indices=position[sources].to_numpy())
    engine_rows = pd.Series(np.arange(len(sources)), index=sources)[engine_nodes["node"].astype("int64")].to_numpy()
    located = call_nodes["node"].notna().to_numpy()
    call_columns = position[call_nodes.loc[located, "node"].astype("int64")].to_numpy()
    tables = []
    for weight, scale in [("travel_time", 1.0), ("length", 1 / 1000)]:
        values = np.full((len(dispatches), len(engine_nodes)), np.nan)
        values[located] = from_sources[weight][engine_rows][:, call_columns].T * scale
        tables.append(pd.DataFrame(values, index=dispatches.index, columns=engine_nodes["engine_id"]))
    network_seconds, network_km = tables
    show_table("network free-flow seconds, home station -> call (rows: runs)", network_seconds.round(0), verbose)
    if verbose:
        own = network_km.columns.get_indexer(dispatches["engine_id"])
        own_km = network_km.to_numpy()[np.arange(len(dispatches)), own]
        line_km = straight_line_km(dispatches["engine_lon_at_dispatch"].to_numpy(), dispatches["engine_lat_at_dispatch"].to_numpy(),
                                   dispatches["incident_lon"].to_numpy(), dispatches["incident_lat"].to_numpy())
        at_home = dispatches["at_home"].astype(bool).to_numpy() & (line_km > 1)
        print(f"unreachable pairs: {int(np.isinf(network_seconds.to_numpy()).sum())}; "
              f"road km / straight-line km for the 2025 engine leaving home, trips over 1 km: "
              f"median {np.nanmedian(own_km[at_home] / line_km[at_home]):.2f}\n")
    return network_seconds, network_km


def linear_travel_model(parameters, network_seconds, network_km):
    """Travel seconds = a + b x free-flow network seconds (the form of our San Francisco calibration).

    Example: linear_travel_model([60.0, 1.2], np.array([300.0]), None) -> array([420.])
    """
    intercept, slope = parameters
    return intercept + slope * network_seconds


def kolesar_travel_model(parameters, network_seconds, network_km):
    """Kolesar, Walker & Hausner (1975): an engine accelerates, then cruises. Travel seconds = 2 sqrt(D / acceleration)
    up to D = speed^2 / acceleration, then speed / acceleration + D / speed, with D the shortest road distance in km.
    The parameters are log(acceleration, km/s^2) and log(speed, km/s), so both stay positive.

    Example: kolesar_travel_model(np.log([0.0001, 0.015]), None, np.array([1.0])) -> array([200.])
    """
    acceleration, speed = np.exp(parameters)
    breakpoint_km = speed ** 2 / acceleration
    return np.where(network_km <= breakpoint_km, 2 * np.sqrt(network_km / acceleration),
                    speed / acceleration + network_km / speed)


def compare_travel_models(dispatches, engines, network_seconds, network_km, travel_curve, verbose=VERBOSE):
    """Fit both forms by least mean |error| on runs that left from home before TRAVEL_TEST_FROM, test them and the
    step-2 straight-line curve on the later runs, and keep the network form with the smaller test mean |error|.

    Example: compare_travel_models(dispatches, engines, network_seconds, network_km, travel_curve)[0]
             -> 'linear on network time'  # chosen form; [1] is a function (seconds, km) -> travel seconds
    """
    rows = np.arange(len(dispatches))
    own_engine = network_seconds.columns.get_indexer(dispatches["engine_id"])
    seconds, km = network_seconds.to_numpy()[rows, own_engine], network_km.to_numpy()[rows, own_engine]
    homes = engines.set_index("engine_id").loc[dispatches["engine_id"], ["home_lon", "home_lat"]].to_numpy()
    line_km = straight_line_km(homes[:, 0], homes[:, 1], dispatches["incident_lon"].to_numpy(), dispatches["incident_lat"].to_numpy())
    observed = dispatches["travel_s"].to_numpy(dtype=float, na_value=np.nan)
    usable = (dispatches["at_home"].astype(bool).to_numpy() & (observed > 0) & np.isfinite(seconds)
              & ~dispatches["qc_note"].str.contains("implausible|equals_incident").to_numpy())
    train = usable & (dispatches["dispatched_at"] < TRAVEL_TEST_FROM).to_numpy()
    test = usable & ~train
    forms = {"linear on network time": (linear_travel_model, [60.0, 1.0]),
             "Kolesar 1975 on road distance": (kolesar_travel_model, np.log([1e-4, 0.015]))}
    fitted = {}
    for name, (model, start) in forms.items():
        result = scipy.optimize.minimize(lambda parameters: np.mean(np.abs(model(parameters, seconds[train], km[train]) - observed[train])),
                                         start, method="Nelder-Mead", options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-8})
        fitted[name] = (model, result.x)
    predictions = {name: model(parameters, seconds[test], km[test]) for name, (model, parameters) in fitted.items()}
    predictions["step 2: straight-line curve"] = travel_curve(line_km[test])
    errors_min = {name: (predicted - observed[test]) / 60 for name, predicted in predictions.items()}
    comparison = pd.DataFrame({name: {"test runs": len(error), "mean |error| min": np.mean(np.abs(error)),
                                      "median |error| min": np.median(np.abs(error)), "bias min": np.mean(error)}
                               for name, error in errors_min.items()}).T
    chosen = min(fitted, key=lambda name: comparison.loc[name, "mean |error| min"])
    model, parameters = fitted[chosen]
    if verbose:
        bands = pd.cut(km[test], DISTANCE_BANDS_KM)
        by_band = pd.DataFrame({name: pd.Series(np.abs(error)).groupby(bands, observed=True).median()
                                for name, error in errors_min.items()})
        by_band["runs"] = pd.Series(bands).value_counts().reindex(by_band.index)
        print(f"fit on {train.sum():,} runs before {TRAVEL_TEST_FROM.date()} that left from home, tested on {test.sum():,} later runs\n"
              f"parameters: linear a = {fitted['linear on network time'][1][0]:.1f} s, b = {fitted['linear on network time'][1][1]:.3f}; "
              f"Kolesar acceleration = {np.exp(fitted['Kolesar 1975 on road distance'][1][0]) * 1e6:.1f} mm/s^2, "
              f"speed = {np.exp(fitted['Kolesar 1975 on road distance'][1][1]) * 3600:.1f} km/h\n"
              f"{comparison.round(3).to_string()}\n"
              f"median |error| (min) by road distance (km):\n{by_band.round(2).to_string()}\n"
              f"chosen: {chosen}\n")
    return chosen, lambda network_seconds, network_km: model(parameters, network_seconds, network_km), comparison


def estimate_network_travel_seconds(dispatches, engines, network_seconds, network_km, travel_model, verbose=VERBOSE):
    """Estimated travel seconds from every engine's home station to every run's call along the road network (the chosen
    travel model); missing when the engine is outside its service span at the dispatch time or the call has no location.

    Example: estimate_network_travel_seconds(dispatches, engines, network_seconds, network_km, travel_model).shape -> (37194, 29)
    """
    estimates = pd.DataFrame(travel_model(network_seconds.to_numpy(), network_km.to_numpy()),
                             index=dispatches.index, columns=network_seconds.columns)
    for engine in engines.itertuples():
        out_of_service = ~dispatches["dispatched_at"].between(engine.in_service_from, engine.in_service_to)
        estimates.loc[out_of_service, engine.engine_id] = np.nan
    show_table("estimated travel seconds along the road network, home station -> call (rows: runs)", estimates.round(0), verbose)
    if verbose:
        rank = estimates.rank(axis=1, method="min")                          # 1 = shortest; same-node engines tie
        rank_of_2025_engine = rank.to_numpy()[np.arange(len(rank)), rank.columns.get_indexer(dispatches["engine_id"])]
        print("rank of the engine that went in 2025 among in-service engines (by estimated network travel from home):",
              pd.Series(rank_of_2025_engine).clip(upper=4).map({1: "1st", 2: "2nd", 3: "3rd", 4: "4th or later"})
              .fillna("no estimate").value_counts(normalize=True).round(3).to_dict(), "\n")
    return estimates


def control_table(runs, simulated, dispatches, engines, estimated_travel):
    """Control for a run of a rule: every run keeps its 2025 engine, but where the rule sent another engine the 2025
    engine also gets its estimated travel, so that rule minus control isolates the choice of engine.

    Example: control_table(runs, protocol, dispatches, engines, estimated_travel).shape -> (37194, 32)
    """
    other_engine = (simulated["engine_id"] != runs["engine_id"]).to_numpy()
    control_times = runs[TIME_COLUMNS].copy()
    for row in runs.index[other_engine]:
        control_times.loc[row] = shift_arrival(runs.loc[row, TIME_COLUMNS], estimated_travel.at[row, runs.at[row, "engine_id"]])
    return simulated_dispatch_table(dispatches, control_times.assign(engine_id=runs["engine_id"]), engines, verbose=False)


def evaluation_column(runs, simulated, dispatches, table, control, check, travel_test_error_min):
    """One version of the dispatch rule scored on the fixed rows of the evaluation matrix: travel estimate error,
    same-engine shares, curve RMSE, and first-on-scene minutes (with rule minus control at P90).

    Example: evaluation_column(runs, protocol, dispatches, protocol_table, control, protocol_check, 1.64)
             ["same engine: all runs"] -> 0.698
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


def load_aid_protocols(dispatches, verbose=VERBOSE):
    """Call types (protocol codes) that may get an engine from the other agency: those with at least one engine sent
    by automatic aid in FY2024 (SOC 2024, printed p. 31: aid covers "a limited set of emergent calls"; the list itself
    is not published). Learnt from FY2024 so that FY2025 can test it.

    Example: sorted(load_aid_protocols(dispatches))[:3] -> ['E02', 'E06', 'E09']
    """
    aid = pd.read_parquet(AUTOMATIC_AID_FY2024)
    engine_aid = aid[aid["Radio_Name"].str.match(r"^(E|Q)\d+$")]
    aid_protocols = set(engine_aid["Problem"].str.extract(r"^([EF]\d{2})", expand=False).dropna())
    show_table("FY2024 automatic aid dispatches (input)", aid, verbose)
    if verbose:
        cross = (dispatches["engine_agency"] == "GFR") != (dispatches["jurisdiction"] == "GFR")
        print(f"engine aid dispatches in FY2024: {len(engine_aid):,}; call types with any: {len(aid_protocols)} -> {sorted(aid_protocols)}\n"
              f"FY2025 runs from the other agency whose call type is in the list: {dispatches.loc[cross, 'protocol'].isin(aid_protocols).mean():.1%} "
              f"of {cross.sum():,}; FY2025 runs whose call type is in the list: {dispatches['protocol'].isin(aid_protocols).mean():.1%}\n")
    return aid_protocols


def apply_agency_boundaries(estimated_travel, dispatches, engines, aid_protocols, verbose=VERBOSE):
    """Estimated travel with the agency line drawn in: an engine of the other agency (GFR vs everyone else) is removed
    (missing) for a call unless the call type gets automatic aid. Calls keep their own agency's engines.

    Example: apply_agency_boundaries(estimated_travel, dispatches, engines, aid_protocols).loc[0].isna().sum() -> 13
    """
    engine_is_gfr = (engines.set_index("engine_id").loc[estimated_travel.columns, "engine_agency"] == "GFR").to_numpy()
    call_in_gfr = (dispatches["jurisdiction"] == "GFR").to_numpy()
    aid_call = dispatches["protocol"].isin(aid_protocols).to_numpy()
    other_agency = engine_is_gfr[None, :] != call_in_gfr[:, None]
    bounded = estimated_travel.mask(other_agency & ~aid_call[:, None])
    show_table("estimated travel with agency boundaries (rows: runs)", bounded.round(0), verbose)
    if verbose:
        own = bounded.columns.get_indexer(dispatches["engine_id"])
        engine_removed = bounded.isna().to_numpy()[np.arange(len(bounded)), own] & estimated_travel.notna().to_numpy()[np.arange(len(bounded)), own]
        print(f"run x engine pairs removed by the boundary: {(bounded.isna() & estimated_travel.notna()).to_numpy().mean():.1%}; "
              f"runs whose 2025 engine is now outside the rule: {int(engine_removed.sum())} ({engine_removed.mean():.2%})\n")
    return bounded


def make_seconds_from_node(graph):
    """Free-flow network seconds from one node to every node, computed once per node and kept (least recently used
    nodes are dropped after NODES_KEPT); returns that function and each node's position in its arrays.

    Example: seconds_from_node, node_position = make_seconds_from_node(graph)
             seconds_from_node(84918009)[node_position[84855238]] -> 41.0 (seconds), roughly
    """
    node_list = list(graph.nodes)
    node_position = pd.Series(np.arange(len(node_list)), index=node_list)
    matrix = nx.to_scipy_sparse_array(ox.convert.to_digraph(graph, weight="travel_time"), nodelist=node_list, weight="travel_time")

    @functools.lru_cache(maxsize=NODES_KEPT)
    def seconds_from_node(node):
        return scipy.sparse.csgraph.dijkstra(matrix, directed=True, indices=int(node_position[node]))

    return seconds_from_node, node_position


def return_progress(elapsed_s, drive_home_s, dwell_s=0.0, slowdown=1.0):
    """Share of the drive home done elapsed_s after clearing, or None once home: the engine stays dwell_s near the scene,
    then drives home slowdown times slower than the calibrated drive of drive_home_s. Step 5.1 used dwell 0, slowdown 1.

    Example: return_progress(120.0, 600.0) -> 0.2; return_progress(120.0, 600.0, dwell_s=60.0, slowdown=1.5) -> 0.0667
    """
    if elapsed_s >= dwell_s + slowdown * drive_home_s:
        return None
    return max(0.0, elapsed_s - dwell_s) / (slowdown * drive_home_s)


def fit_return_progress(dispatches, engines, call_nodes, home_nodes, seconds_from_node, node_position, travel_model,
                        verbose=VERBOSE):
    """Fit return_progress's dwell and slowdown to where engines really were (AVL position at dispatch, CAD fields
    Latitude/Longitude_At_Assign_Time) when sent soon after a call: runs whose engine had arrived at its previous call,
    cleared it under RETURN_CHECK_SECONDS before, and stood in the corridor between that scene and home (scene -> engine
    -> home at most RETURN_CORRIDOR_FACTOR x scene -> home, straight line), home included: keeping only engines still
    out would keep only the slow ones and bias the fit toward slow drives. The actual share of the
    way home is scene -> engine / (scene -> engine + engine -> home). Fit by least mean |share error| on runs before
    TRAVEL_TEST_FROM, tested on the later ones next to step 5.1 (dwell 0, slowdown 1).

    Example: fit_return_progress(...)[:2] -> (0.0, 1.18)  # dwell seconds, slowdown, roughly
    """
    order = dispatches.sort_values(["engine_id", "dispatched_at"])
    previous = order.groupby("engine_id")
    sample = pd.DataFrame({"engine_id": order["engine_id"], "dispatched_at": order["dispatched_at"],
                           "scene_row": pd.Series(order.index, index=order.index).groupby(order["engine_id"]).shift(),
                           "elapsed_s": (order["dispatched_at"] - previous["cleared_at"].shift()).dt.total_seconds(),
                           "previous_arrived": previous["arrived_at"].shift().notna(),
                           "at_home": order["at_home"].astype(bool)})
    sample = sample[sample.previous_arrived & sample.elapsed_s.between(0, RETURN_CHECK_SECONDS, inclusive="left")].copy()
    sample["scene_row"] = sample["scene_row"].astype(int)
    scene = dispatches.loc[sample["scene_row"], ["incident_lon", "incident_lat"]].to_numpy()
    home = engines.set_index("engine_id").loc[sample["engine_id"], ["home_lon", "home_lat"]].to_numpy()
    avl = dispatches.loc[sample.index, ["engine_lon_at_dispatch", "engine_lat_at_dispatch"]].to_numpy()
    scene_to_engine = straight_line_km(scene[:, 0], scene[:, 1], avl[:, 0], avl[:, 1])
    engine_to_home = straight_line_km(avl[:, 0], avl[:, 1], home[:, 0], home[:, 1])
    scene_to_home = straight_line_km(scene[:, 0], scene[:, 1], home[:, 0], home[:, 1])
    sample["actual_share"] = scene_to_engine / (scene_to_engine + engine_to_home)
    in_corridor = (scene_to_engine + engine_to_home <= RETURN_CORRIDOR_FACTOR * scene_to_home) & (scene_to_home >= RETURN_MIN_KM)
    scene_nodes = call_nodes.loc[sample["scene_row"], "node"].to_numpy()
    usable = in_corridor & ~pd.isna(scene_nodes)
    sample = sample[usable].copy()
    sample["drive_home_s"] = [float(travel_model(seconds_from_node(int(scene_node))[node_position[home_nodes[engine_id]]], np.nan))
                              for scene_node, engine_id in zip(scene_nodes[usable], sample["engine_id"])]
    train = (sample["dispatched_at"] < TRAVEL_TEST_FROM).to_numpy()

    def predicted_share(part, dwell_s, slowdown):
        return [1.0 if share is None else share for share in
                (return_progress(elapsed, drive, dwell_s, slowdown) for elapsed, drive in zip(part.elapsed_s, part.drive_home_s))]

    result = scipy.optimize.minimize(lambda parameters: np.mean(np.abs(np.array(predicted_share(sample[train], *np.exp(parameters)))
                                                                       - sample[train].actual_share)),
                                     np.log([60.0, 1.5]), method="Nelder-Mead", options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-8})
    dwell_s, slowdown = np.exp(result.x)
    test = sample[~train].copy()
    test["step 5.1 (no dwell, no slowdown)"] = predicted_share(test, 0.0, 1.0)
    test["AVL-fitted"] = predicted_share(test, dwell_s, slowdown)
    comparison = pd.DataFrame({name: {"test runs": len(test), "mean |share error|": np.mean(np.abs(test[name] - test.actual_share)),
                                      "median predicted share": np.median(test[name])}
                               for name in ["step 5.1 (no dwell, no slowdown)", "AVL-fitted"]}).T
    if verbose:
        minute_band = pd.cut(test.elapsed_s / 60, RETURN_MINUTE_BANDS)
        by_minute = test.groupby(minute_band, observed=True)[["actual_share", "step 5.1 (no dwell, no slowdown)", "AVL-fitted"]].median()
        by_minute["share at home"] = test.groupby(minute_band, observed=True).at_home.mean()
        by_minute["runs"] = test.groupby(minute_band, observed=True).size()
        print(f"runs sent within {RETURN_CHECK_SECONDS // 60} min of clearing a call they reached, in the scene -> home corridor: "
              f"{len(sample):,}, {sample.at_home.mean():.0%} at home (fit {train.sum():,} before {TRAVEL_TEST_FROM.date()}, "
              f"test {len(test):,}); median actual share of the way home {sample.actual_share.median():.2f}\n"
              f"fitted ({result.message}): dwell {dwell_s:.0f} s near the scene, then {slowdown:.2f} x the calibrated drive home\n"
              f"{comparison.round(3).to_string()}\nmedian share by minutes since clear (test runs):\n{by_minute.round(2).to_string()}\n")
    return dwell_s, slowdown, comparison


def share_of_way_home(engine_id, last_clear, now, call_nodes, home_nodes, seconds_from_node, node_position, travel_model,
                      dwell_s=0.0, slowdown=1.0):
    """Where an engine is at `now`, from its last clear in the simulation's engine_state: (share of the way home done,
    network seconds from the scene it left to every node) while it drives home, else (None, None): it is at its station
    because it has not run yet, its last run left no scene, it cleared RETURN_CHECK_SECONDS or more ago, that scene has no
    location, or it is home already. last_clear is (clear s, scene run) after a call it reached: the share is
    return_progress(elapsed, drive home, dwell_s, slowdown); or (clear s, scene run, f) after a cancelled run f of the way
    out (M0): it drives back without dwelling, so the share is 1 - f x (1 - its progress back).

    Example: an engine that cleared 300 s ago from a scene 600 s (calibrated) from home, slowdown 1.18, no dwell:
             share_of_way_home(engine, (cleared_s, scene_run), cleared_s + 300, ...)[0] -> 0.42  # 300 / (1.18 x 600)
             share_of_way_home(engine, None, now, ...) -> (None, None)  # has not run yet: at its station
    """
    if last_clear is None:
        return None, None
    cleared_s, scene_row, *outbound = last_clear
    elapsed = now - cleared_s
    if scene_row is None or elapsed >= RETURN_CHECK_SECONDS:
        return None, None
    scene_node = call_nodes.at[scene_row, "node"]
    if pd.isna(scene_node):
        return None, None
    from_scene = seconds_from_node(int(scene_node))
    drive_home = float(travel_model(from_scene[node_position[home_nodes[engine_id]]], np.nan))
    if outbound:                                                 # M0: released from a cancelled run, f of the way out
        progress_back = return_progress(elapsed, outbound[0] * drive_home, 0.0, slowdown)
        share_home = None if progress_back is None else 1 - outbound[0] * (1 - progress_back)
    else:
        share_home = return_progress(elapsed, drive_home, dwell_s, slowdown)
    return (None, None) if share_home is None else (share_home, from_scene)


def travel_while_returning(row, now, engine_state, travel_for_run, call_nodes, home_nodes, seconds_from_node, node_position,
                           travel_model, dwell_s=0.0, slowdown=1.0):
    """Cases 2 and 3: travel to this run's call for engines still driving home from their last call. An engine that
    cleared t seconds ago from scene S is a share p = return_progress(t, R, dwell_s, slowdown) of the way home, R being
    the calibrated drive S -> home; its travel is (1 - p) x travel(S -> call) + p x travel(home -> call). Other engines
    keep the travel from home. The travel model must be the linear one, which uses network seconds only.
    Section 6 (M0) also stores engines released from a cancelled run as (release, run, f): they were f of the way from
    home to that call and drive back from there without dwelling, so p = 1 - f x (1 - their progress back).

    Example: travel_while_returning(row, now, engine_state, network_travel.loc[row], call_nodes, home_nodes, seconds_from_node,
                                    node_position, network_travel_model)["E3"] -> 512.0 (seconds), roughly
    """
    call_node = call_nodes.at[row, "node"]
    if pd.isna(call_node):
        return travel_for_run
    adjusted = travel_for_run.copy()
    for engine_id, last_clear in engine_state.items():
        if pd.isna(adjusted.get(engine_id)):
            continue
        share_home, from_scene = share_of_way_home(engine_id, last_clear, now, call_nodes, home_nodes, seconds_from_node,
                                                   node_position, travel_model, dwell_s, slowdown)
        if share_home is None:
            continue
        from_scene_to_call = float(travel_model(from_scene[node_position[int(call_node)]], np.nan))
        adjusted[engine_id] = (1 - share_home) * from_scene_to_call + share_home * travel_for_run[engine_id]
    return adjusted


def evaluation_matrix(columns, verbose=VERBOSE):
    """The evaluation matrix: one column per version of the rule, the same rows for every version.

    Example: evaluation_matrix({"2025 record": column_2025, "step 2: straight line": column_2}).shape -> (15, 2)
    """
    matrix = pd.DataFrame(columns)
    if verbose:
        print(f"--- evaluation matrix\n{matrix.round(3).to_string()}\n")
    return matrix


def clean_runs_for_m0(runs, verbose=VERBOSE):
    """Runs with a usable timeline for M0 (Sections 1-5 keep the raw record): a negative turnout or on-scene time
    becomes 0, a run that arrived but never cleared stays the median on-scene time of complete runs, and a run that
    neither arrived nor cleared is released after the median dispatch-to-clear time of cancelled runs.

    Example: clean_runs_for_m0(runs).loc[14718, ["dispatched_at", "enroute_at"]] -> equal (was 1 s apart, reversed)
    """
    cleaned = runs.copy()
    complete = cleaned["arrived_at"].notna() & cleaned["cleared_at"].notna()
    cancelled = cleaned["arrived_at"].isna() & cleaned["cleared_at"].notna()
    median_on_scene = (cleaned["cleared_at"] - cleaned["arrived_at"])[complete].median()                 # [SOURCE] 730 s in FY2025
    median_cancelled = (cleaned["cleared_at"] - cleaned["dispatched_at"])[cancelled].median()            # [SOURCE] 333 s in FY2025
    cleaned["enroute_at"] = cleaned["enroute_at"].clip(lower=cleaned["dispatched_at"])
    cleaned["cleared_at"] = cleaned["cleared_at"].clip(lower=cleaned["arrived_at"])
    cleaned["cleared_at"] = cleaned["cleared_at"].fillna(cleaned["arrived_at"] + median_on_scene)
    cleaned["cleared_at"] = cleaned["cleared_at"].fillna(cleaned["dispatched_at"] + median_cancelled)
    if verbose:
        changed = (cleaned[TIME_COLUMNS] != runs[TIME_COLUMNS]) & ~(cleaned[TIME_COLUMNS].isna() & runs[TIME_COLUMNS].isna())
        rows = changed.any(axis=1)
        print(f"--- M0 timelines repaired: {int(rows.sum())} runs (median on-scene {median_on_scene:.0f} s, "
              f"median cancelled {median_cancelled:.0f} s)\n"
              f"{pd.concat({'2025': runs.loc[rows, TIME_COLUMNS], 'M0': cleaned.loc[rows, TIME_COLUMNS]}, axis=1).to_string()}\n")
    return cleaned


def dispatch_closest(run, available_engines, travel_for_run, rank_penalty=None):
    """Section 6 rule for the choice (M0): the available engine with the shortest modeled travel from where it is now
    goes, always with that travel, so no run keeps its 2025 times. Equal travel goes to the engine listed first (engines
    table order), never to the 2025 engine. With no available engine it returns (None, None) and the run waits
    (simulate). A call without a location has no travel for any engine: its 2025 engine goes with its 2025 times when
    free, otherwise the run waits for that engine. rank_penalty (M0.3; engine id -> seconds) is added when ranking only:
    the engine sent still takes its own modeled travel.

    Example: dispatch_closest(runs.loc[0], ["E62", "E3"], boundary_travel.loc[0]) -> ('E62', 257.05...)  # seconds
             dispatch_closest(runs.loc[0], [], boundary_travel.loc[0]) -> (None, None)
    """
    if travel_for_run.isna().all():                              # call without coordinates (4 runs)
        return (run["engine_id"], None) if run["engine_id"] in available_engines else (None, None)
    candidates = travel_for_run[available_engines].dropna()
    if candidates.empty:
        return None, None
    ranked = candidates if rank_penalty is None else candidates + rank_penalty.reindex(candidates.index, fill_value=0.0)
    best = ranked.idxmin()
    return best, float(candidates[best])


def share_first_arrivals_as_recorded(runs, simulated):
    """Share of calls whose simulated first arrival is the second recorded in 2025 (calls with no arrival count as not equal).

    Example: share_first_arrivals_as_recorded(runs, simulate(runs)) -> 0.877  # the replay: every call that had an arrival
             share_first_arrivals_as_recorded(runs, m0_protocol) -> 0.0001  # M0 (FY2025)
    """
    first_simulated = simulated["arrived_at"].groupby(runs["incident_id"]).min()
    first_2025 = runs["arrived_at"].groupby(runs["incident_id"]).min()
    return float((first_simulated == first_2025).mean())


def check_counterfactual_run(runs, simulated, dispatches, report, arrivals_by_call=False, verbose=VERBOSE):
    """E0 state checks of a run that should be counterfactual-consistent. The first six must be 0; kept 2025 times
    must not exceed the runs of calls without a location; the rest describe the run. With arrivals_by_call (M0.2, where
    the call decides which of its runs arrive) the arrival check counts calls whose number of arrivals differs from
    2025 instead of runs that lost their arrival.

    Example: check_counterfactual_run(runs, m0_protocol, dispatches, m0_report)["runs never served"] -> 0
    """
    served = simulated["engine_id"].notna()
    order = simulated[served].sort_values(["engine_id", "dispatched_at"])
    still_busy = order["dispatched_at"] < order.groupby("engine_id")["cleared_at"].shift()
    out_of_order = (simulated[TIME_COLUMNS].ffill(axis=1).diff(axis=1).iloc[:, 1:] < 0).any(axis=1)
    waits = pd.Series(report["waited_seconds"], dtype=float)
    if arrivals_by_call:
        arrivals = [frame["arrived_at"].notna().groupby(runs["incident_id"]).sum() for frame in (runs, simulated)]
        arrival_check = {"calls whose number of arrivals differs from 2025": int((arrivals[0] != arrivals[1]).sum())}
    else:
        arrival_check = {"runs with an arrival in 2025 but none simulated":
                         int((runs["arrived_at"].notna() & simulated["arrived_at"].isna()).sum())}
    checks = pd.Series({
        "runs never served": len(report["never_served"]),
        "runs whose engine was taken by a newer run": len(report["taken_by_newer_run"]),
        "runs sent while their engine was still on an earlier run": int(still_busy.sum()),
        **arrival_check,
        "runs with times out of order": int(out_of_order.sum()),
        "runs served without a clear": int((served & simulated["cleared_at"].isna()).sum()),
        "runs that kept their 2025 times": len(report["kept_recorded"]),
        "runs of calls without a location": int(dispatches["incident_lon"].isna().sum()),
        "cancelled requests that never got an engine": len(report["cancelled_while_waiting"]),
        "calls whose first arrival is the recorded second (share)": share_first_arrivals_as_recorded(runs, simulated),
        "runs that waited for an engine": len(waits),
        "wait seconds: median": waits.median(),
        "wait seconds: max": waits.max(),
    })
    if verbose:
        print(f"--- E0 state checks\n{checks.round(4).to_string()}\n")
        if len(waits):
            ranks = dispatches.loc[waits.index, "priority"].astype("string").fillna("(missing)")
            print(f"waits by priority (count, median s):\n{waits.groupby(ranks).agg(['size', 'median']).round(0).to_string()}\n")
    return checks


def compare_curves_by_stratum(table, dispatches, verbose=VERBOSE):
    """E0 strata: RMSE and largest |simulated - 2025| share of the first-on-scene curve, for all calls, multi-engine
    calls, calls whose first 2025 engine was sent away from home, and calls with an engine from the other agency.
    Strata come from the 2025 record, so both curves use the same calls.

    Example: compare_curves_by_stratum(m0_table, dispatches).columns.tolist() -> ['calls', 'rmse', 'max_abs_difference']
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
        print(f"--- E0 first-on-scene curve, simulated vs 2025, by stratum\n{by_stratum.round(4).to_string()}\n")
    return by_stratum


def decide_e0(states, by_stratum, verbose=VERBOSE):
    """E0 decision (research.md): first-on-scene curve RMSE for all calls at most E0_RMSE_LIMIT, no state violation,
    and only calls without a location keep their 2025 times.

    Example: decide_e0(e0_states, e0_strata) -> False  # FY2025: no state violation, but RMSE 0.0377 > 0.03
    """
    violation_checks = ["runs never served", "runs whose engine was taken by a newer run",
                        "runs sent while their engine was still on an earlier run",
                        "runs with an arrival in 2025 but none simulated", "calls whose number of arrivals differs from 2025",
                        "runs with times out of order", "runs served without a clear"]
    violations = states[states.index.intersection(violation_checks)].sum()   # one of the two arrival checks is present
    passed = bool(by_stratum.loc["all calls", "rmse"] <= E0_RMSE_LIMIT and violations == 0
                  and states["runs that kept their 2025 times"] <= states["runs of calls without a location"])
    if verbose:
        print(f"E0: RMSE all calls {by_stratum.loc['all calls', 'rmse']:.4f} (limit {E0_RMSE_LIMIT}), state violations "
              f"{int(violations)}, kept 2025 times {int(states['runs that kept their 2025 times'])} "
              f"(calls without a location {int(states['runs of calls without a location'])}) -> passed: {passed}\n")
    return passed


def draw_travel_ratio(spread, inside_city, modeled_seconds, quantile):
    """M0.1: recorded / modeled travel at `quantile` of the ratios observed on trips like this one: calls in the same
    area (inside or outside the city) whose modeled travel falls in the same bin. spread: {inside_city: (inner bin
    edges in seconds, one array of observed ratios per bin)}, from fit_travel_spread. Bins are right-closed like
    pd.qcut; a trip shorter (longer) than every fitting trip uses the first (last) bin.

    Example: toy = {True: (np.array([300.0]), [np.array([0.5, 1.0, 2.0]), np.array([0.9, 1.0, 1.1])])}
             draw_travel_ratio(toy, True, 200.0, 0.5) -> 1.0   # 200 s is in bin 0; its median
             draw_travel_ratio(toy, True, 200.0, 0.99) -> 2.0; draw_travel_ratio(toy, True, 400.0, 0.99) -> 1.1
    """
    inner_edges, ratios_by_bin = spread[inside_city]
    ratios = ratios_by_bin[int(np.searchsorted(inner_edges, modeled_seconds, side="left"))]
    return float(np.quantile(ratios, quantile, method="inverted_cdf"))


def fit_travel_spread(dispatches, network_seconds, network_km, travel_model, spread_quantile, bin_count=TRAVEL_SPREAD_BINS,
                      verbose=VERBOSE):
    """M0.1: how real travel spreads around the modeled travel. On the travel model's own fitting runs (left from home,
    dispatched before TRAVEL_TEST_FROM; the filter of compare_travel_models), ratio = recorded / modeled travel, kept
    per area (call inside or outside the city) and per `bin_count` equal-count bins of modeled travel. The later
    runs (Jul-Sep; held out from the ratios, though Section 3 chose the travel model's form on their errors) test it:
    each gets modeled travel x the ratio at its own quantile (spread_quantile: run row -> quantile in (0, 1), as in the
    simulation), and recorded travel should fall below the predicted P10 and above the predicted P90 about 10% of the
    time each, per area (the check) and per bin (in the bin table, so short trips are seen on their own). Returns the
    spread (for draw_travel_ratio), the bin table and the check. M0.4 fits brush fires on their own: pass the brush-fire
    rows of dispatches, network_seconds and network_km, with bin_count=BRUSH_SPREAD_BINS.

    Example: travel_spread, spread_table, spread_check = fit_travel_spread(dispatches, network_seconds, network_km,
                                                                           network_travel_model, spread_quantile)
             spread_table.loc[(False, 0), "ratio P50"] -> 1.097  # the shortest trips outside the city take ~10% longer
    """
    rows = np.arange(len(dispatches))
    own_engine = network_seconds.columns.get_indexer(dispatches["engine_id"])
    seconds, km = network_seconds.to_numpy()[rows, own_engine], network_km.to_numpy()[rows, own_engine]
    recorded = dispatches["travel_s"].to_numpy(dtype=float, na_value=np.nan)
    usable = (dispatches["at_home"].astype(bool).to_numpy() & (recorded > 0) & np.isfinite(seconds)
              & ~dispatches["qc_note"].str.contains("implausible|equals_incident").to_numpy())
    trips = pd.DataFrame({"inside_city": dispatches["focal"].fillna(False).astype(bool).to_numpy(),
                          "modeled_s": travel_model(seconds, km), "recorded_s": recorded,
                          "fitting_run": (dispatches["dispatched_at"] < TRAVEL_TEST_FROM).to_numpy()},
                         index=dispatches.index)[usable]
    trips["ratio"] = trips["recorded_s"] / trips["modeled_s"]
    # ponytail: ratios are kept as recorded, CAD button slips included (P1 down to 0.02 on the shortest trips); trim if a use needs physical travel only
    show_table("trips for the travel spread (runs that left from home)", trips, verbose)
    spread, bins = {}, []
    for inside_city, fit in trips[trips["fitting_run"]].groupby("inside_city"):
        bin_of_trip, edges = pd.qcut(fit["modeled_s"], bin_count, labels=False, retbins=True)
        ratios_by_bin = [np.sort(fit["ratio"][bin_of_trip == b].to_numpy()) for b in range(bin_count)]
        spread[inside_city] = (edges[1:-1], ratios_by_bin)
        for b, ratios in enumerate(ratios_by_bin):
            bins.append({"inside_city": inside_city, "bin": b, "modeled from (min)": edges[b] / 60,
                         "modeled to (min)": edges[b + 1] / 60, "fitting runs": len(ratios),
                         **{f"ratio P{q}": np.quantile(ratios, q / 100) for q in (1, 10, 50, 90, 99)}})
    spread_table = pd.DataFrame(bins).set_index(["inside_city", "bin"])
    test = trips[~trips["fitting_run"]].copy()
    quantile_of_run = {"drawn_s": spread_quantile[test.index], "predicted_p10_s": pd.Series(0.1, index=test.index),
                       "predicted_p90_s": pd.Series(0.9, index=test.index)}
    for name, quantile in quantile_of_run.items():
        test[name] = [modeled * draw_travel_ratio(spread, inside, modeled, quantile[row])
                      for row, inside, modeled in zip(test.index, test["inside_city"], test["modeled_s"])]
    test["bin"] = [int(np.searchsorted(spread[inside][0], modeled, side="left"))      # the bin draw_travel_ratio uses
                   for inside, modeled in zip(test["inside_city"], test["modeled_s"])]
    test["below_p10"], test["above_p90"] = test["recorded_s"] < test["predicted_p10_s"], test["recorded_s"] > test["predicted_p90_s"]
    spread_table = spread_table.join(test.groupby(["inside_city", "bin"]).agg(
        **{"held-out runs": ("below_p10", "size"), "held-out below P10": ("below_p10", "mean"),
           "held-out above P90": ("above_p90", "mean")}))
    checks = {}
    for area, part in [("all", test), *[(AREA_LABELS[flag], test[test["inside_city"] == flag]) for flag in (True, False)]]:
        checks[area] = {"test runs": len(part),
                        **{f"{source} P{q} (min)": part[column].quantile(q / 100) / 60
                           for source, column in [("recorded", "recorded_s"), ("modeled", "modeled_s"),
                                                  ("modeled x drawn ratio", "drawn_s")] for q in (10, 50, 90)},
                        "recorded below predicted P10": part["below_p10"].mean(),
                        "recorded above predicted P90": part["above_p90"].mean()}
    spread_check = pd.DataFrame(checks).T
    if verbose:
        print(f"travel spread fitted on {int(trips['fitting_run'].sum()):,} runs before {TRAVEL_TEST_FROM.date()} that left "
              f"from home, tested on {len(test):,} later runs (held out from the ratios; Section 3 chose the travel "
              f"model's form on their errors)\n"
              f"--- ratio recorded / modeled travel by area and bin of modeled travel; held-out shares outside the "
              f"bin's P10-P90 (about 0.10 each if calibrated)\n{spread_table.round(3).to_string()}\n"
              f"--- held-out check (Jul-Sep): travel minutes, and how often recorded travel falls outside the predicted "
              f"P10-P90 (about 0.10 each if calibrated)\n{spread_check.round(3).T.to_string()}\n")
    return spread, spread_table, spread_check


def dispatch_closest_with_spread(run, available_engines, travel_for_run, spread, inside_city, spread_quantile,
                                 rank_penalty=None):
    """Section 7 rule (M0.1): the M0 choice (dispatch_closest, on modeled travel, as CAD ranks on its estimate), but
    the chosen engine then takes modeled travel x a ratio drawn at this run's own quantile (spread_quantile: run row ->
    quantile, the same for every policy, so two policies that send different engines to a run share its luck).
    inside_city: run row -> whether the call is inside the city. Runs that keep their 2025 times (calls without a
    location) and runs that must wait pass through unchanged. rank_penalty (M0.3) goes to dispatch_closest.

    Example: dispatch_closest_with_spread(runs.loc[0], ["E62", "E3"], boundary_travel.loc[0], travel_spread,
                                          call_inside_city, spread_quantile) -> ('E62', 392.51...)  # 257.05 s x 1.527
    """
    engine_id, travel_seconds = dispatch_closest(run, available_engines, travel_for_run, rank_penalty)
    if travel_seconds is None:
        return engine_id, None
    ratio = draw_travel_ratio(spread, bool(inside_city[run.name]), travel_seconds, spread_quantile[run.name])
    return engine_id, travel_seconds * ratio


def describe_engine_origins(dispatch_log, dispatches, estimated_travel, call_nodes, home_nodes, seconds_from_node,
                            node_position, travel_model, dwell_s=0.0, slowdown=1.0, verbose=VERBOSE):
    """Where each engine was when the simulation sent it, from its dispatch log (report["dispatches"]): at its station,
    driving home from a call it reached, or driving home after a cancelled run, placed by share_of_way_home exactly as
    the simulation placed it. Per dispatch: minutes since the engine cleared, share of the way home done, its modeled
    travel from where it was, the travel it took, and the travel from home (estimated_travel), in minutes.
    Prints the counts beside the 2025 record (AVL position at assignment), the share of the way home by minutes since
    clearing, three examples of each kind, and three checks: every share is in [0, 1); nobody counts as on the way home
    RETURN_CHECK_SECONDS or more after clearing; every engine counted at its station got the travel from home.

    Example: describe_engine_origins(m01_report["dispatches"], dispatches, boundary_travel, call_nodes, ...)["origin"]
             .value_counts() -> at station / driving home from a call / driving home after a cancelled run (counts)
    """
    records = []
    for run, time_s, engine_id, last_clear, modeled_s, travel_s in dispatch_log.itertuples(index=False):
        share_home, _ = share_of_way_home(engine_id, last_clear, time_s, call_nodes, home_nodes, seconds_from_node,
                                          node_position, travel_model, dwell_s, slowdown)
        origin = ("at station" if share_home is None else
                  "driving home from a call" if len(last_clear) == 2 else "driving home after a cancelled run")
        records.append({"run": run, "time": DAY_ZERO + pd.Timedelta(seconds=time_s), "engine_id": engine_id, "origin": origin,
                        "minutes since clear": np.nan if last_clear is None else (time_s - last_clear[0]) / 60,
                        "share of way home": np.nan if share_home is None else share_home,
                        "modeled min": modeled_s / 60, "taken min": np.nan if travel_s is None else travel_s / 60,
                        "from home min": estimated_travel.at[run, engine_id] / 60})
    origins = pd.DataFrame(records)
    if verbose:
        recorded = dispatches["at_home"].astype(bool).map({True: "at its station", False: "away from its station"})
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
    """What the travel draw did in a run (M0.1 on): per dispatch (describe_engine_origins), the ratio of the travel taken
    to the modeled travel, with the call's area and the bin of the modeled travel. Prints five examples and, per area
    and bin, the drawn ratios' P10, P50 and P90 beside the fitted ones: they should match, because every run draws its
    own quantile from its bin's fitted ratios.

    Example: describe_travel_draws(m01_origins, travel_spread, call_inside_city).loc[(False, 0), "drawn P50"]
             -> about 1.10, beside fitted P50 1.097
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


def call_fates(runs):
    """M0.2: what the record fixes for each call with two or more runs: how many of its runs arrived, the moments its
    other runs were released (their clears, sorted), and the on-scene seconds of the runs that arrived, in arrival order.

    Example: call_fates(m0_runs)["100124-0344"] -> (1, [38904.0], [121.0])  # 1 arrival; 1 release at 38,904 s; 121 s on scene
             len(call_fates(m0_runs)) -> 2941
    """
    fates = {}
    for incident, call in runs.groupby("incident_id"):
        if len(call) < 2:
            continue
        arrived = call[call["arrived_at"].notna()].sort_values("arrived_at")
        fates[incident] = (len(arrived), sorted(call.loc[call["arrived_at"].isna(), "cleared_at"].tolist()),
                           (arrived["cleared_at"] - arrived["arrived_at"]).tolist())
    return fates


def describe_call_fates(runs, simulated, dispatch_log, fates_by_arrival, verbose=VERBOSE):
    """M0.2: what the call-level fates did. Per run of a call in fates_by_arrival: its fate in the 2025 record (arrived
    or cancelled) and in the simulation (arrived, released, or not sent), its engine, the travel its engine took (from
    the dispatch log), and minutes from the call to its dispatch and arrival in both. Prints the fates side by side,
    two checks (every call keeps its recorded number of arrivals; a released run has no arrival), and three calls whose
    first-dispatched run was cancelled in 2025, run by run.

    Example: describe_call_fates(runs, m02_protocol, m02_report["dispatches"], fates_by_arrival)["simulated"]
             .value_counts() -> arrived / released / not sent (counts)
    """
    rows = runs.index[runs["incident_id"].isin(fates_by_arrival.keys())]
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
        print(f"--- M0.2 call fates: {len(fates_by_arrival):,} calls with two or more runs, {len(table):,} runs\n"
              f"{pd.crosstab(table['2025'], table['simulated']).to_string()}\n"
              f"checks: calls whose number of arrivals differs from 2025: {int((arrivals['2025'] != arrivals['simulated']).sum())}; "
              f"released runs with an arrival: {int(((table['simulated'] != 'arrived') & table['simulated arrived min'].notna()).sum())}\n")
        for incident in examples:
            print(f"call {incident} (first-dispatched run cancelled in 2025):\n"
                  f"{table[table['incident_id'] == incident].drop(columns='incident_id').round(2).to_string()}\n")
    return table


def fit_quint_penalty(dispatches, runs, engines, estimated_travel, grid=QUINT_PENALTY_GRID_S, verbose=VERBOSE):
    """M0.3: how many seconds a quint (Q: a pumper that also carries an aerial ladder) must save over an engine before
    CAD sends it. Fitted on single-engine calls whose 2025 engine left from its station (one choice, a known origin):
    the engines CAD could send are those with an estimate in estimated_travel (in service, allowed by the agency line)
    and not on a 2025 run at that moment (the record's availability; runs gives the repaired 2025 times). The rule
    sends the one with the least modeled travel from home, plus the penalty if it is a quint. The penalty in `grid`
    that sends CAD's engine most often on the fitting months (before TRAVEL_TEST_FROM) is kept; the test months
    (Jul-Sep) check it. Returns the penalty (seconds) and the agreement table (penalty x months x calls); a group with
    no runs gets NaN. M0.4 fits one call type at a time: pass its rows of dispatches, and a grid with negative penalties.

    Example: quint_penalty_s, penalty_fit = fit_quint_penalty(dispatches, m0_runs, engines, boundary_travel)
             penalty_fit.loc[0, "fit: all"] -> share of fitting runs where the plain closest rule sends CAD's engine
    """
    engine_ids = estimated_travel.columns
    is_quint = (engines.set_index("engine_id").loc[engine_ids, "engine_type"] == "Q").to_numpy()
    rows = dispatches.index[(dispatches["n_engines_sent"] == 1) & dispatches["at_home"].astype(bool)
                            & dispatches["incident_lon"].notna()]
    dispatched_s = runs.loc[rows, "dispatched_at"].to_numpy()
    busy = np.zeros((len(rows), len(engine_ids)), dtype=bool)
    for column, engine_id in enumerate(engine_ids):              # on a 2025 run that started before and clears after
        own = runs[runs["engine_id"] == engine_id].sort_values("dispatched_at")
        starts, ends = own["dispatched_at"].to_numpy(), own["cleared_at"].to_numpy()
        # ponytail: only the engine's last run started before is checked; overlapping runs of one engine are rare (3 in 2025)
        last = np.searchsorted(starts, dispatched_s, side="left") - 1
        busy[:, column] = (last >= 0) & (ends[np.maximum(last, 0)] > dispatched_s)
    travel = estimated_travel.loc[rows, engine_ids].to_numpy()
    travel = np.where(busy | np.isnan(travel), np.inf, travel)
    cad = engine_ids.get_indexer(dispatches.loc[rows, "engine_id"])
    fitting = (dispatches.loc[rows, "dispatched_at"] < TRAVEL_TEST_FROM).to_numpy()
    kind = dispatches.loc[rows, "call_type"].astype(object).fillna("(missing)").to_numpy()
    study_fire = dispatches.loc[rows, "protocol"].isin(STUDY_FIRE_PROTOCOLS).to_numpy()
    groups = [("all", np.ones(len(rows), dtype=bool)), ("CAD Fire category", kind == "Fire"), ("study fires", study_fire),
              ("medical", kind == "Medical")]
    agreement = {}
    for penalty in grid:
        same = np.argmin(travel + penalty * is_quint, axis=1) == cad
        agreement[int(penalty)] = {f"{months}: {group}": same[in_months & in_group].mean() if (in_months & in_group).any() else np.nan
                                   for months, in_months in [("fit", fitting), ("test", ~fitting)] for group, in_group in groups}
    penalty_fit = pd.DataFrame(agreement).T.rename_axis("quint penalty s")
    quint_penalty_s = float(penalty_fit["fit: all"].idxmax())
    if verbose:
        closest = np.argmin(travel, axis=1)
        chosen = np.argmin(travel + quint_penalty_s * is_quint, axis=1)
        changed = np.flatnonzero(closest != chosen)
        examples = pd.DataFrame({
            "run": rows[changed], "call type": kind[changed], "closest (a quint)": engine_ids[closest[changed]],
            "closest travel min": travel[changed, closest[changed]] / 60, "sent instead": engine_ids[chosen[changed]],
            "sent travel min": travel[changed, chosen[changed]] / 60, "CAD sent": dispatches.loc[rows[changed], "engine_id"].to_numpy()})
        shown = penalty_fit.loc[[penalty for penalty in penalty_fit.index if penalty % 30 == 0 or penalty == quint_penalty_s]]
        print(f"--- quint penalty fit: {int(fitting.sum()):,} fitting and {int((~fitting).sum()):,} test runs (single-engine calls, "
              f"engine from its station); CAD's engine among the engines the rule could send: {np.isfinite(travel[np.arange(len(rows)), cad]).mean():.1%}\n"
              f"share of runs where the rule sends CAD's engine, by penalty:\n{shown.round(3).to_string()}\n"
              f"chosen penalty: {quint_penalty_s:.0f} s; it changes the closest engine on {len(changed):,} runs "
              f"({len(changed) / len(rows):.1%}), of which CAD sent the engine the penalty picks: "
              f"{(examples['sent instead'] == examples['CAD sent']).mean():.1%}, the quint: {(examples['closest (a quint)'] == examples['CAD sent']).mean():.1%}\n"
              f"examples:\n{examples.head(5).round(2).to_string(index=False)}\n")
    return quint_penalty_s, penalty_fit


def fit_quint_penalty_by_call_type(dispatches, runs, engines, estimated_travel, call_type_of_run, single_penalty_s,
                                   verbose=VERBOSE):
    """M0.4: the quint penalty fitted separately per call type. call_type_of_run: run row -> label (QUINT_RULE_CALL_TYPES,
    every other call one label). Each label runs fit_quint_penalty on its own rows over QUINT_PENALTY_BY_TYPE_GRID_S, so
    the same runs, availability and months as M0.3 (single-engine calls, engine from its station; fit Oct-Jun, test
    Jul-Sep). A negative penalty means CAD prefers the quint unless an engine is that much closer. single_penalty_s
    (M0.3's one penalty for all calls) is printed beside. Returns {label: penalty seconds} and the agreement table (rows:
    label x penalty; columns: fit and test months).

    Example: quint_penalty_by_type, type_penalty_fit = fit_quint_penalty_by_call_type(
                 dispatches, m0_runs, engines, boundary_travel, call_type_for_quint, quint_penalty_s)
             quint_penalty_by_type -> {'F56 elevator': -260.0, 'F82 brush fire': 260.0, 'all other calls': 10.0}
    """
    quint_penalty_by_type, agreement_by_type = {}, {}
    for label, rows in call_type_of_run.groupby(call_type_of_run).groups.items():
        quint_penalty_by_type[label], fit = fit_quint_penalty(dispatches.loc[rows], runs, engines, estimated_travel,
                                                              grid=QUINT_PENALTY_BY_TYPE_GRID_S, verbose=False)
        agreement_by_type[label] = fit[["fit: all", "test: all"]]
    type_penalty_fit = pd.concat(agreement_by_type, names=["call type"])
    if verbose:
        fitting_rows = dispatches.index[(dispatches["n_engines_sent"] == 1) & dispatches["at_home"].astype(bool)
                                        & dispatches["incident_lon"].notna()]
        months = np.where(dispatches.loc[fitting_rows, "dispatched_at"] < TRAVEL_TEST_FROM, "fitting runs", "test runs")
        summary = pd.crosstab(call_type_of_run[fitting_rows], months).rename_axis(index="call type", columns=None)
        summary["chosen penalty s"] = pd.Series(quint_penalty_by_type)
        for months_column in ["fit: all", "test: all"]:
            summary[f"{months_column[:4]} agreement at chosen"] = [type_penalty_fit.loc[(label, int(penalty)), months_column]
                                                                  for label, penalty in quint_penalty_by_type.items()]
            summary[f"{months_column[:4]} agreement at {single_penalty_s:.0f} s (M0.3)"] = [
                type_penalty_fit.loc[(label, int(single_penalty_s)), months_column] for label in quint_penalty_by_type]
        shown = type_penalty_fit[type_penalty_fit.index.get_level_values(1) % 60 == 0].unstack(0)
        print(f"--- quint penalty by call type (single-engine calls, engine from its station; negative = the quint "
              f"preferred)\n{summary.round(3).to_string()}\n"
              f"share of runs where the rule sends CAD's engine, by penalty:\n{shown.round(3).to_string()}\n")
    return quint_penalty_by_type, type_penalty_fit


def describe_type_choices(dispatches, versions, groups=None, verbose=VERBOSE):
    """M0.3: what the quint penalty changed in a full run. Per version (name -> simulated run table, simulate's output)
    and per call group: the share of runs sent a quint (2025 beside), the share that got CAD's engine, split by the type
    CAD sent; and five runs where the first two versions sent different types.

    Call groups (default): all, the CAD Fire category (call_type Fire: alarms, crashes and fires), the study's fire calls
    (STUDY_FIRE_PROTOCOLS) and medical calls; `groups` (a list of (name, mask over dispatches rows)) replaces them (M0.4).

    Example: describe_type_choices(dispatches, {"M0.2": m02_protocol, "M0.3": m03_protocol}).loc[("M0.3", "study fires")]
             -> runs sent a quint, same engine as CAD, ...
    """
    kind = dispatches["call_type"].astype(object).fillna("(missing)")
    cad_engine = dispatches["engine_id"].astype(object)
    cad_type = cad_engine.str[0]
    if groups is None:
        groups = [("all", kind.notna()), ("CAD Fire category", kind == "Fire"),
                  ("study fires", dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS)), ("medical", kind == "Medical")]
    shares = {("2025", group): {"runs sent a quint": (cad_type[mask] == "Q").mean()} for group, mask in groups}
    for name, simulated in versions.items():
        sent = simulated["engine_id"].astype(object)
        same = sent == cad_engine
        for group, mask in groups:
            shares[(name, group)] = {"runs sent a quint": (sent[mask].str[0] == "Q").mean(), "same engine as CAD": same[mask].mean(),
                                     "same engine, CAD sent an engine": same[mask & (cad_type == "E")].mean(),
                                     "same engine, CAD sent a quint": same[mask & (cad_type == "Q")].mean()}
    table = pd.DataFrame(shares).T
    if verbose:
        first, second = list(versions)[:2]
        sent_first, sent_second = (versions[name]["engine_id"].astype(object) for name in (first, second))
        differs = sent_first.str[0].ne(sent_second.str[0]) & sent_first.notna() & sent_second.notna()
        examples = pd.DataFrame({"call type": kind, "CAD sent": cad_engine, first: sent_first, second: sent_second})[differs]
        print(f"--- engine types sent, by version\n{table.round(3).to_string()}\n"
              f"runs where {first} and {second} sent different types: {int(differs.sum()):,}; examples:\n{examples.head(5).to_string()}\n")
    return table


def weekly_block_interval(errors_s, weeks):
    """Two-sided interval (BOOTSTRAP_INTERVAL) of the mean of errors_s, from BLOCK_BOOTSTRAP_DRAWS weekly block-bootstrap
    draws: the weeks that have calls are resampled with replacement (seed BLOCK_BOOTSTRAP_SEED), so calls of one busy
    week stay together, as research.md specifies for its comparisons. errors_s and weeks share one index; no errors
    give (nan, nan).

    Example: weekly_block_interval(pd.Series([10.0, 30.0, -20.0]), pd.Series([0, 0, 1])) -> (-20.0, 20.0)
             # each draw takes two weeks: means -20, 6.67 or 20 s
    """
    if errors_s.empty:
        return np.nan, np.nan
    sums, counts = errors_s.groupby(weeks).sum().to_numpy(), errors_s.groupby(weeks).size().to_numpy()
    picks = np.random.default_rng(BLOCK_BOOTSTRAP_SEED).integers(0, len(sums), size=(BLOCK_BOOTSTRAP_DRAWS, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(means, BOOTSTRAP_INTERVAL)
    return float(low), float(high)


def compare_fire_arrivals(dispatches, tables, verbose=VERBOSE):
    """M0.4: first-engine arrival at the study's fire calls (STUDY_FIRE_PROTOCOLS), the outcome research.md compares
    policies on. Per version (tables: name -> simulated dispatch table, from simulated_dispatch_table), per code and for
    all four codes together, over the whole year and over the held-out months (from TRAVEL_TEST_FROM, research.md's
    July-September): calls reached in both, mean simulated minus recorded first arrival (seconds) with its weekly
    block-bootstrap interval, whether that mean is within FIRE_MARGIN_S (research.md's m), whether the interval
    contains 0 (no detectable bias), where the interval lies against +-m (within +-m, early or late beyond m, or
    undetermined when it crosses a bound: the precision), and P90 minutes from the call to the first engine (2025 beside).

    Example: fire_check = compare_fire_arrivals(dispatches, {"M0.3": m03_table, "M0.4": m04_table})
             fire_check.loc[("M0.3", "Jul-Sep", "F82"), "mean error s"] -> -183.4
    """
    study_rows = dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False).to_numpy()

    def first_arrivals(table):                                   # per call: code, call time, first engine on scene
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
    """M0.4: share of one code's multi-engine calls (two or more engine runs in 2025) that got at least one quint, in
    2025 and per version (tables: name -> simulated dispatch table), by area and in all, on the same calls; a version
    meets the done-criterion where its share is within QUINT_SHARE_TOLERANCE of 2025's.

    Example: compare_quint_calls(dispatches, {"M0.3": m03_table}).loc["all", ["2025", "M0.3"]].tolist() -> about [0.07, 0.30]
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


def holding_runs(runs_table, engine_ids, moments):
    """E1: per moment (rows) and engine (columns), the row of the run holding the engine then (dispatched at or before
    the moment, cleared after it), or -1 when it is free. runs_table: simulate's output or the repaired 2025 runs
    (engine_id, dispatched_at, cleared_at in seconds since DAY_ZERO).

    Example: holding_runs(m0_runs, ["E62"], np.array([7000.0]))[0, 0] -> 0  # 2025 run 0 holds E62 at 7,000 s
    """
    holders = np.full((len(moments), len(engine_ids)), -1)
    sent = runs_table.dropna(subset=["engine_id", "dispatched_at", "cleared_at"]).sort_values("dispatched_at")
    for column, engine_id in enumerate(engine_ids):
        own = sent[sent["engine_id"] == engine_id]
        if own.empty:                                            # never sent: free at every moment
            continue
        starts, ends, rows = own["dispatched_at"].to_numpy(), own["cleared_at"].to_numpy(), own.index.to_numpy()
        # ponytail: only the engine's last run started by the moment is checked; overlapping runs of one engine are rare (3 in 2025, none simulated)
        last = np.searchsorted(starts, moments, side="right") - 1
        held = (last >= 0) & (ends[np.maximum(last, 0)] > moments)
        holders[:, column] = np.where(held, rows[np.maximum(last, 0)], -1)
    return holders


def check_first_due(dispatches, runs_table, estimated_travel, engines, verbose=VERBOSE):
    """E1 (research.md RQ1): for each study fire call (STUDY_FIRE_PROTOCOLS, with a location), whether every engine of
    its first-due station was committed to another call when the call was first dispatched. First-due station: the
    station of the engine with the least modeled travel from home (estimated_travel on the call's first run: in service
    and allowed by the agency line), as the protocol ranks engines. Committed: on a run of another call in runs_table
    (simulate's output; or the repaired 2025 runs for the record). Per call: code, area, call time, first-due station and
    its engines, committed or not, the calls holding them (code, call type, priority), minutes from the call to the
    first engine on scene, and how much farther from home (modeled minutes) that engine was than the first-due engine.

    Example: first_due_m04 = check_first_due(dispatches, m04_protocol, boundary_travel, engines)
             first_due_m04["committed"].mean() -> share of fire calls whose first-due station had no free engine
    """
    station_key = engines.set_index("engine_id")["home_station"].astype(object)
    station_key = station_key.where(station_key.notna(), station_key.index.to_series())   # no station id: its own station
    engine_ids = estimated_travel.columns
    station_of_column = station_key.reindex(engine_ids).to_numpy()
    inside_city = dispatches["focal"].fillna(False).astype(bool)
    call_type, protocol, priority = (dispatches[column].astype(object).fillna("(missing)") for column in ["call_type", "protocol", "priority"])
    rows = dispatches.index[dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False) & dispatches["incident_lon"].notna()]
    sent = runs_table.loc[rows, ["dispatched_at", "arrived_at", "engine_id"]].assign(incident_id=dispatches.loc[rows, "incident_id"])
    first_sent = sent.dropna(subset=["dispatched_at"]).sort_values("dispatched_at").groupby("incident_id").head(1)
    first_on_scene = sent.dropna(subset=["arrived_at"]).sort_values("arrived_at").groupby("incident_id").head(1).set_index("incident_id")
    travel = estimated_travel.loc[first_sent.index].to_numpy()                # every engine, from home, to the call (seconds)
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
    """E1: per version (name -> check_first_due output) and group of fire calls (all four codes, each code, inside and
    outside the city, Jul-Sep): fire calls; the share whose first-due station had no free engine, with its weekly
    block-bootstrap interval; among those, the share held by a medical call and by a CAD Fire-category call; minutes to
    the first engine on scene (median, P90) when the station had a free engine and when it had none; and the extra
    modeled minutes of the engine that came. Also, per first-due station (where), fire calls and the committed share.

    Example: first_due_summary, first_due_by_station = summarize_first_due({"2025 record": first_due_record,
                                                                            "M0.4 protocol run": first_due_m04})
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


def build_readiness_cells(dispatches, graph, engines, home_nodes, aid_protocols, seconds_from_node, node_position,
                          travel_model, verbose=VERBOSE):
    """E1 (research.md 3.3, fire readiness): the fire-call intensity on READINESS_CELL_M square cells, from the study's
    fire calls in the training months (before TRAINING_MONTHS_END), and the modeled travel from every engine's home
    station to every cell. A cell keeps its share of those calls, split into call types with automatic aid (an engine
    of either agency may come) and the rest (only the cell's own agency: GFR, or everyone else, by the majority of its
    calls), and the mean position of its calls, snapped to the road network. Returns the cells (one row per cell with a
    training fire call), the travel table (rows: engines, columns: cells; seconds) and the share of training fire calls
    by hour of day (lambda_F(x, t) = cell share x hour share).

    Example: cells, travel_to_cells, hour_share = build_readiness_cells(dispatches, graph, engines, home_nodes,
                 aid_protocols, seconds_from_node, node_position, network_travel_model)
             cells[["weight_aid", "weight_own"]].to_numpy().sum() -> 1.0
    """
    calls = dispatches[dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False) & dispatches["incident_lon"].notna()
                       & (dispatches["call_received_at"] < TRAINING_MONTHS_END)].drop_duplicates("incident_id")
    x, y = pyproj.Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform(
        calls["incident_lon"].to_numpy(), calls["incident_lat"].to_numpy())
    calls = calls.assign(cell_x=np.floor(x / READINESS_CELL_M).astype(int), cell_y=np.floor(y / READINESS_CELL_M).astype(int),
                         aid=calls["protocol"].isin(aid_protocols).to_numpy(), in_gfr=calls["jurisdiction"].eq("GFR").fillna(False).to_numpy())
    cells = calls.groupby(["cell_x", "cell_y"]).agg(
        calls_aid=("aid", "sum"), calls_other=("aid", lambda aid: int((~aid).sum())), gfr_share=("in_gfr", "mean"),
        lon=("incident_lon", "mean"), lat=("incident_lat", "mean")).reset_index()
    cells["cell_gfr"] = cells["gfr_share"] > 0.5
    cells["weight_aid"] = cells["calls_aid"] / len(calls)
    cells["weight_own"] = cells["calls_other"] / len(calls)
    cells["node"], cells["snap_m"] = ox.distance.nearest_nodes(graph, X=cells["lon"].to_numpy(), Y=cells["lat"].to_numpy(), return_dist=True)
    positions = node_position[cells["node"]].to_numpy()
    travel_to_cells = pd.DataFrame([travel_model(seconds_from_node(home_nodes[engine_id])[positions], np.nan)
                                    for engine_id in engines["engine_id"]], index=engines["engine_id"], columns=cells.index)
    hour_share = calls["call_received_at"].dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0.0)
    if verbose:
        show_table("readiness cells (training fire calls per cell)", cells, verbose)
        nearest = travel_to_cells.idxmin().to_numpy()
        print(f"training fire calls: {len(calls):,} in {len(cells):,} cells of {READINESS_CELL_M} m; with automatic aid "
              f"{int(cells['calls_aid'].sum()):,}, own agency only {int(cells['calls_other'].sum()):,}; cells mostly GFR "
              f"{int(cells['cell_gfr'].sum())}; snap distance median {cells['snap_m'].median():.0f} m, P90 {cells['snap_m'].quantile(0.9):.0f} m\n"
              f"busiest cells and their nearest engine (all idle):\n"
              f"{cells.assign(nearest_engine=nearest, nearest_min=travel_to_cells.min().to_numpy() / 60).nlargest(5, 'calls_aid')[['cell_x', 'cell_y', 'calls_aid', 'calls_other', 'cell_gfr', 'nearest_engine', 'nearest_min']].round(2).to_string()}\n"
              f"share of training fire calls by hour (0-23): {hour_share.round(3).tolist()}\n")
    return cells, travel_to_cells, hour_share


def fire_readiness(runs_table, engines, cells, travel_to_cells, hour_share, wait_for_busy=False, verbose=VERBOSE):
    """E1 (research.md 3.3): fire readiness over FY2025. Between consecutive dispatches, clears and in-service changes
    the set of idle engines is fixed; in each such segment tau(x) is the modeled travel to cell x from the nearest idle
    engine allowed there (the cell's own agency for call types without automatic aid, either agency for the rest), and
    readiness R is the fire-call-weighted mean of tau over the cells, in minutes. R_full is the same with every
    in-service engine idle, so R - R_full is the fire protection lost to engines being busy. Idle engines are placed at
    their home stations. Segments are also cut at every hour, and the yearly value weights each segment by its length
    and its hour's share of training fire calls. Returns the segments (start, end, idle engines, R and R_full), the cells with their time-weighted mean tau and
    tau_full (minutes), and the summary (year; by month; by hour; time-weighted quantiles of R - R_full).
    A segment where some weighted cell has no idle allowed engine is left out (E1: 1-2 a year). With wait_for_busy
    (Section 13; research.md 3.3: tau includes the wait when no engine is free) such a cell instead gets the least
    (minutes until an allowed in-service engine clears, at the segment's middle, + its travel from its station), and the
    segments get waiting_weight, the weight of those cells: every candidate's wait shrinks a minute per minute and no
    run clears inside a segment, so R falls by waiting_weight minutes per minute inside it and its middle value is its mean.

    Example: segments, cell_tau, readiness_summary = fire_readiness(m04_protocol, engines, cells, travel_to_cells, hour_share)
             readiness_summary.loc["year, hour-weighted", "R min"] -> readiness in minutes
             toy: one engine 5 min from the only cell, on a run from 0 to 600 s: the segment 0-600 s (middle 300 s) is
             left out; with wait_for_busy its R = (600 - 300) / 60 + 5 = 10 min and waiting_weight 1 (R runs from 15 to 5 min)
    """
    engine_ids = travel_to_cells.index
    year_end_s = (YEAR_END - DAY_ZERO).total_seconds()
    service = engines.set_index("engine_id").loc[engine_ids]
    service_from = (service["in_service_from"] - DAY_ZERO).dt.total_seconds().to_numpy()
    service_to = (service["in_service_to"] - DAY_ZERO).dt.total_seconds().to_numpy()
    sent = runs_table.dropna(subset=["engine_id", "dispatched_at", "cleared_at"])
    every_hour = np.arange(0.0, year_end_s, 3600.0)             # cut at every hour (so at every month too): a segment takes one hour's weight
    times = np.unique(np.clip(np.concatenate([[0.0, year_end_s], every_hour, sent["dispatched_at"], sent["cleared_at"], service_from,
                                              service_to]), 0.0, year_end_s))
    starts, ends = times[:-1], times[1:]
    middle, duration = (starts + ends) / 2, ends - starts
    in_service = (middle[:, None] >= service_from) & (middle[:, None] < service_to)
    holders = holding_runs(runs_table, engine_ids, middle)
    idle = in_service & (holders < 0)
    if wait_for_busy:                                            # minutes until each engine's holding run clears (0 when idle)
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
                if len(stuck):                                   # a weighted cell with no idle allowed engine: wait for the first to clear
                    segment_rows = np.arange(len(starts))[part][stuck]
                    waited = np.where(in_service[segment_rows, :, None], wait_min[segment_rows, :, None] + travel_min[None], np.inf)
                    stuck_any, stuck_own = np.isinf(tau_any[stuck]), np.isinf(tau_own[stuck])
                    tau_any[stuck] = np.where(stuck_any, waited.min(axis=1), tau_any[stuck])
                    tau_own[stuck] = np.where(stuck_own, np.where(own_agency[None], waited, np.inf).min(axis=1), tau_own[stuck])
                    waiting_weight[segment_rows] = stuck_any[:, aid_cells] @ weight_aid + stuck_own[:, own_cells] @ weight_own
            values[part] = tau_any[:, aid_cells] @ weight_aid + tau_own[:, own_cells] @ weight_own
            tau_cell = (np.where(aid_cells, tau_any, 0) * cells["weight_aid"].to_numpy()
                        + np.where(own_cells, tau_own, 0) * cells["weight_own"].to_numpy()) / cell_weight
            reachable = np.isfinite(tau_cell)      # ponytail: a segment with no allowed idle engine for a cell is left out, not given a wait (1-2 segments a year)
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


def make_readiness_tools(cells, travel_to_cells, engines):
    """Section 12 (research.md 3.3-3.4): E1's fire readiness for one set of idle engines, and the opportunity cost of
    sending an engine. readiness_of(idle mask over travel_to_cells' engines) -> (R in minutes, tau per cell in minutes):
    the fire-call-weighted mean travel from the nearest idle allowed engine at its station (the formula of
    fire_readiness, for one set). opportunity_cost(idle engine ids, candidate ids) -> Series candidate -> minutes: how
    much R rises if that engine leaves now, OC(e) = R(idle without e) - R(idle); inf when it is the only idle engine
    allowed in some weighted cell.

    Example: readiness_of, opportunity_cost = make_readiness_tools(readiness_cells, travel_to_cells, engines)
             opportunity_cost(["E1", "E3", "E5"], ["E1", "E3"]) -> E1 0.21, E3 0.08 (minutes), roughly
    """
    engine_ids = travel_to_cells.index
    position = pd.Series(np.arange(len(engine_ids)), index=engine_ids)
    travel_min = travel_to_cells.to_numpy() / 60
    own_agency = (engines.set_index("engine_id").loc[engine_ids, "engine_agency"] == "GFR").to_numpy()[:, None] == cells["cell_gfr"].to_numpy()[None, :]
    weight_aid, weight_own = cells["weight_aid"].to_numpy(), cells["weight_own"].to_numpy()

    def readiness_of(idle):
        tau_any = np.where(idle[:, None], travel_min, np.inf).min(axis=0)
        tau_own = np.where(idle[:, None] & own_agency, travel_min, np.inf).min(axis=0)
        weighted = np.where(weight_aid > 0, tau_any, 0.0) * weight_aid + np.where(weight_own > 0, tau_own, 0.0) * weight_own
        return float(weighted.sum()), weighted / (weight_aid + weight_own)

    def opportunity_cost(idle_engine_ids, candidate_ids):
        idle = np.zeros(len(engine_ids), dtype=bool)
        idle[position[list(idle_engine_ids)].to_numpy()] = True
        now, _ = readiness_of(idle)
        costs = {}
        for engine_id in candidate_ids:
            without = idle.copy()
            without[position[engine_id]] = False
            costs[engine_id] = readiness_of(without)[0] - now
        return pd.Series(costs, dtype=float)

    return readiness_of, opportunity_cost


def choose_by_opportunity_cost(run, available_engines, travel_for_run, rank_penalty, low_acuity, action_window_s, opportunity_cost):
    """Section 12 rule (research.md 3.2 and 3.4), the choice alone: the M0.4 protocol's engine, except on a low-acuity
    call (low_acuity: run row -> bool). There, among the available engines whose ranking (modeled travel from where
    each is, plus rank_penalty) is within action_window_s of the protocol's engine, the one whose departure costs the
    least fire readiness (opportunity_cost); a saving of OC_TIE_S or less keeps the protocol's engine. Returns the
    protocol's engine, the engine to send, and the opportunity costs of the engines in the window (empty when the call
    is not low-acuity or has one engine in it).

    Example: choose_by_opportunity_cost(m0_runs.loc[25], available, travel_for_run, rank_penalty, low_acuity, 60, opportunity_cost)
             -> ('E21', 'E28', E21 0.179 / E24 0.108 / E28 0.002 minutes)
    """
    protocol_engine, protocol_travel = dispatch_closest(run, available_engines, travel_for_run, rank_penalty)
    costs = pd.Series(dtype=float)
    if protocol_travel is not None and low_acuity[run.name]:
        ranked = (travel_for_run[available_engines] + rank_penalty.reindex(available_engines, fill_value=0.0)).dropna()
        near = ranked.index[ranked <= ranked[protocol_engine] + action_window_s]
        if len(near) > 1:
            costs = opportunity_cost(available_engines, near)
    chosen = costs.idxmin() if len(costs) and costs[protocol_engine] - costs.min() > OC_TIE_S / 60 else protocol_engine
    return protocol_engine, chosen, costs


def dispatch_by_opportunity_cost(run, available_engines, travel_for_run, spread, inside_city, spread_quantile, rank_penalty,
                                 low_acuity, action_window_s, opportunity_cost):
    """Section 12 rule as a dispatch function for simulate: choose_by_opportunity_cost picks the engine, which then takes
    its own modeled travel x the drawn ratio, as in dispatch_closest_with_spread.

    Example: dispatch_by_opportunity_cost(m0_runs.loc[25], available, travel_for_run, travel_spread, call_inside_city,
                                          spread_quantile, rank_penalty, low_acuity, ACTION_WINDOW_S, opportunity_cost)
             -> ('E28', 830.4)  # E21 was 49 s closer but leaves more fire readiness uncovered
    """
    _, chosen, _ = choose_by_opportunity_cost(run, available_engines, travel_for_run, rank_penalty, low_acuity, action_window_s,
                                              opportunity_cost)
    return dispatch_closest_with_spread(run, available_engines if chosen is None else [chosen], travel_for_run, spread, inside_city,
                                        spread_quantile, rank_penalty)


def census_decisions(dispatches, runs_table, estimated_travel, engines, quint_penalty_of_run, quint_indicator, low_acuity,
                     opportunity_cost, action_windows_s, verbose=VERBOSE):
    """Section 12.2: the room the rule (and an RL agent) has, counted on a protocol run without simulating. For every
    low-acuity run, at its dispatch moment in runs_table: the idle in-service engines (holding_runs; the run's own
    engine counts as idle); the protocol's engine (least modeled travel from home plus M0.4's quint penalty, among the
    allowed ones; checked against the engine the run sent); and, per action window in action_windows_s, how many engines
    rank within it of the protocol's engine, the engine the rule would send, the opportunity cost it saves (minutes of
    fire readiness) and its extra modeled travel (seconds). Travel is from the stations, as in the opportunity cost.

    Example: census = census_decisions(dispatches, m04_protocol, boundary_travel, engines, quint_penalty_of_run,
                                       quint_indicator, low_acuity, opportunity_cost, [30, 60, 120])
             census["choices_60"].ge(2).mean() -> share of low-acuity decisions with a choice at a 60 s window
    """
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
            pick = costs[near].idxmin() if costs[protocol_engine] - costs[near].min() > OC_TIE_S / 60 else protocol_engine
            record.update({f"choices_{window}": len(near), f"rule_engine_{window}": pick,
                           f"oc_saving_min_{window}": costs[protocol_engine] - costs[pick],
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


def weekly_block_quantile_difference(policy_values, protocol_values, weeks, quantile):
    """Section 12.3: the difference of a quantile (policy minus protocol) over paired calls, with its two-sided interval
    (BOOTSTRAP_INTERVAL) from BLOCK_BOOTSTRAP_DRAWS weekly block-bootstrap draws (weeks resampled with replacement, seed
    BLOCK_BOOTSTRAP_SEED). The three Series share one index.

    Example: weekly_block_quantile_difference(policy_s, protocol_s, weeks, 0.9) -> (4.0, -6.5, 15.1)  # seconds
    """
    week_ids = np.unique(weeks.to_numpy())
    members = [np.flatnonzero(weeks.to_numpy() == week) for week in week_ids]
    policy, protocol = policy_values.to_numpy(), protocol_values.to_numpy()
    rng = np.random.default_rng(BLOCK_BOOTSTRAP_SEED)
    draws = []
    for _ in range(BLOCK_BOOTSTRAP_DRAWS):
        picked = np.concatenate([members[k] for k in rng.integers(0, len(week_ids), len(week_ids))])
        draws.append(np.quantile(policy[picked], quantile) - np.quantile(protocol[picked], quantile))
    low, high = np.quantile(draws, BOOTSTRAP_INTERVAL)
    return float(np.quantile(policy, quantile) - np.quantile(protocol, quantile)), float(low), float(high)


def classify_against_margin(low, high, margin):
    """research.md E3: the class of a difference (policy minus protocol; negative = faster) from its interval and the
    margin: meaningful improvement (the interval is below -margin), practical equivalence (within +-margin), meaningful
    harm (above +margin), otherwise unresolved.

    Example: classify_against_margin(-30.0, -12.0, 10) -> 'meaningful improvement'
    """
    if high < -margin:
        return "meaningful improvement"
    if low > margin:
        return "meaningful harm"
    if -margin <= low and high <= margin:
        return "practical equivalence"
    return "unresolved"


def compare_policies(dispatches, protocol_table, policy_table, first_due_protocol, readiness_protocol, readiness_policy,
                     hour_share, low_acuity, verbose=VERBOSE):
    """Section 12.3: a policy's simulated year against the protocol's under the same random draws, on research.md's
    measures. Per outcome: the policy minus protocol difference (seconds), its weekly block-bootstrap interval and
    research.md's class against the margin (m for fire, delta for the medical constraint):
    - first-engine arrival at the study's fire calls: whole year; Jul-Sep; calls whose first-due station was committed
      under the protocol (first_due_protocol); calls whose first-due station is one of HOTSPOT_STATIONS; calls received
      in DAYTIME_HOURS (the hours whose E1 readiness loss exceeds the yearly mean);
    - fire readiness: the hour-weighted mean R of each week (readiness_*: fire_readiness segments);
    - P90 first arrival at high-acuity medical calls (determinant levels HIGH_ACUITY_LEVELS), against delta;
    - mean first arrival at low-acuity calls, their cost.

    Example: comparison = compare_policies(dispatches, m04_table, rule_table, first_due_m04, readiness_m04, readiness_rule,
                                           fire_hour_share, low_acuity)
             comparison.loc["fire calls, whole year", "class"] -> 'practical equivalence'
    """
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


def branch_decisions(dispatches, runs, protocol_run, census, simulate_window, protocol_choose, rule_choice, readiness_of, cells,
                     travel_to_cells, engines, verbose=VERBOSE):
    """Section 12.4 (research.md 3.4, paired branch simulations): does the fire readiness a decision saves turn into
    later benefit? Low-acuity decisions where the census finds the rule would send another engine (ACTION_WINDOW_S) are
    drawn evenly over BRANCH_STRATA strata of the opportunity cost saved (seed BRANCH_SEED). Each decision is simulated
    twice from the same state, rebuilt by simulating the calls first dispatched in the BRANCH_WARMUP_S before it
    (simulate_window on the repaired 2025 runs): branch A follows the protocol (protocol_choose); branch B applies the
    rule (rule_choice: run, available, travel -> protocol's engine, rule's engine, opportunity costs) at the decision,
    in that reconstructed state, and the protocol everywhere else. A decision counts only if branch A reproduces the
    full-year protocol run over the BRANCH_HORIZON_S after the decision (every run's engine and times, rtol 0, and the
    fire readiness) and the rule still changes the engine in that state; drawing goes on, up to three times the target
    per stratum, until BRANCH_DECISIONS such decisions. Over the horizon: each branch's mean fire readiness (readiness_of
    on the idle engines between dispatches and clears), first arrivals at later study fire calls and at the decision's
    own call, and the decision run's busy time (workload). Returns the per-decision table (counted and not) and the cells
    with the readiness change summed over the counted decisions (minutes x hours, A minus B: positive where the rule's
    engine left better cover).

    Example: branches, branch_cells = branch_decisions(dispatches, m0_runs, m04_protocol, census, simulate_window,
                                                       protocol_choose, rule_choice, readiness_of, readiness_cells,
                                                       travel_to_cells, engines)
             branches.loc[branches["counted"], "benefit_s"].mean() -> readiness gained over two hours by the rule (s)
    """
    engine_ids = travel_to_cells.index
    service = engines.set_index("engine_id").loc[engine_ids]
    service_from = (service["in_service_from"] - DAY_ZERO).dt.total_seconds().to_numpy()
    service_to = (service["in_service_to"] - DAY_ZERO).dt.total_seconds().to_numpy()

    def horizon_readiness(table, start, end):
        """Mean fire readiness (minutes) and per-cell tau x time (minutes x hours) over [start, end] for a run table."""
        sent = table.dropna(subset=["engine_id", "dispatched_at", "cleared_at"])
        sent = sent[(sent["dispatched_at"] < end) & (sent["cleared_at"] > start)]     # runs that can hold an engine in the horizon
        times = np.unique(np.clip(np.concatenate([[start, end], sent["dispatched_at"], sent["cleared_at"]]), start, end))
        middle, duration = (times[:-1] + times[1:]) / 2, np.diff(times)
        idle = (middle[:, None] >= service_from) & (middle[:, None] < service_to) & (holding_runs(sent, engine_ids, middle) < 0)
        readiness, tau = zip(*(readiness_of(mask) for mask in idle))
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

            def rule_at_decision(run, available, travel):
                if run.name != row:
                    return protocol_choose(run, available, travel)
                protocol_engine, chosen, costs = rule_choice(run, available, travel)
                decision.update(protocol_engine=protocol_engine, rule_engine=chosen,
                                oc_saving_s=(costs[protocol_engine] - costs[chosen]) * 60 if chosen != protocol_engine else 0.0,
                                extra_travel_s=travel[chosen] - travel[protocol_engine])
                return protocol_choose(run, [chosen], travel)

            branch_a, branch_b = simulate_window(window, protocol_choose), simulate_window(window, rule_at_decision)
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
        weeks = used["week"].to_numpy()
        week_ids = np.unique(weeks)
        members = [np.flatnonzero(weeks == week) for week in week_ids]
        rng = np.random.default_rng(BLOCK_BOOTSTRAP_SEED)
        draws = []
        for _ in range(BLOCK_BOOTSTRAP_DRAWS):
            pick = np.concatenate([members[k] for k in rng.integers(0, len(week_ids), len(week_ids))])
            draws.append(np.linalg.lstsq(design[pick], benefit[pick], rcond=None)[0][1])
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

def build_call_pool(dispatches, verbose=VERBOSE):
    """Section 13: the real calls a scenario draws from. One row per incident received in the training months (before
    TRAINING_MONTHS_END) with a location: call time (seconds since DAY_ZERO), day (days since DAY_ZERO), hour, block of
    SCENARIO_BAND_HOURS hours, weekend or not, study fire call (STUDY_FIRE_PROTOCOLS) or not, protocol, agency side of
    the line (jurisdiction) and city flag (focal), position in PROJECTED_CRS and its READINESS_CELL_M cell, the record
    rows of its runs, and the row of its first run to arrive (-1 when none arrived).

    Example: pool = build_call_pool(dispatches)
             pool.loc["100124-0043", ["day", "hour", "fire", "runs"]].tolist() -> [0, 1, False, 1]
    """
    training = dispatches[(dispatches["call_received_at"] < TRAINING_MONTHS_END) & dispatches["incident_lon"].notna()]
    groups = training.groupby("incident_id", sort=False)
    first = groups.head(1)
    first_arrival = training.dropna(subset=["arrived_at"]).sort_values("arrived_at").groupby("incident_id").head(1)
    x, y = pyproj.Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform(
        first["incident_lon"].to_numpy(), first["incident_lat"].to_numpy())
    received_s = (first["call_received_at"] - DAY_ZERO).dt.total_seconds().to_numpy()
    pool = pd.DataFrame({
        "received_s": received_s, "day": (received_s // 86400).astype(int), "hour": first["call_received_at"].dt.hour.to_numpy(),
        "weekend": (first["call_received_at"].dt.dayofweek >= 5).to_numpy(),
        "fire": first["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False).to_numpy(dtype=bool),
        "protocol": first["protocol"].astype(object).to_numpy(), "jurisdiction": first["jurisdiction"].astype(object).to_numpy(),
        "focal": first["focal"].fillna(False).to_numpy(dtype=bool), "x": x, "y": y,
        "cell_x": np.floor(x / READINESS_CELL_M).astype(int), "cell_y": np.floor(y / READINESS_CELL_M).astype(int)},
        index=pd.Index(first["incident_id"].astype(object).to_numpy(), name="incident_id"))
    pool["band"] = pool["hour"] // SCENARIO_BAND_HOURS
    rows = groups.groups
    pool["run_rows"] = [np.asarray(rows[incident]) for incident in pool.index]
    pool["runs"] = pool["run_rows"].map(len)
    pool["arrival_row"] = (pd.Series(first_arrival.index.to_numpy(), index=first_arrival["incident_id"].astype(object).to_numpy())
                           .reindex(pool.index).fillna(-1).astype(int))
    show_table("call pool: training-month incidents a scenario draws from", pool, verbose)
    if verbose:
        print(f"incidents {len(pool):,} (study fire calls {int(pool['fire'].sum()):,}), runs {int(pool['runs'].sum()):,}, "
              f"days {pool['day'].nunique()}; incidents per block of {SCENARIO_BAND_HOURS} h: "
              f"{pool.groupby('band').size().to_dict()}\n")
    return pool


def build_resident_locations(pool, graph, home_nodes, seconds_from_node, node_position, engines, verbose=VERBOSE):
    """Section 13 (where growth and surges put new calls: more residents, more calls): every 2020 Census block with
    residents, at its internal point snapped to the road network: residents, position (lon/lat and PROJECTED_CRS), its
    READINESS_CELL_M cell, node and snap distance, the agency side of the line and city flag of the nearest
    training-month call (jurisdiction, focal), its first-due station (the home station of the engine of its own agency
    side with the fewest free-flow network seconds to it; an engine without a station id is its own station, as in
    check_first_due), and the free-flow network seconds from every engine's home station to its node (rows: blocks;
    columns: engines, as network_from_stations gives for a call).

    Example: locations, location_seconds = build_resident_locations(pool, graph, home_nodes, seconds_from_node, node_position, engines)
             locations["residents"].sum() -> 278468; location_seconds.shape -> (blocks with residents, 29)
    """
    blocks = gpd.read_file(CENSUS_BLOCKS_GPKG, ignore_geometry=True)
    show_table("2020 Census blocks (input)", blocks, verbose)
    blocks = blocks[blocks["POP100"] > 0].reset_index(drop=True)
    lon, lat = blocks["INTPTLON"].astype(float).to_numpy(), blocks["INTPTLAT"].astype(float).to_numpy()
    x, y = pyproj.Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform(lon, lat)
    nodes, snap_m = ox.distance.nearest_nodes(graph, X=lon, Y=lat, return_dist=True)
    nearest_call = scipy.spatial.cKDTree(pool[["x", "y"]].to_numpy()).query(np.column_stack([x, y]))[1]
    locations = pd.DataFrame({"residents": blocks["POP100"].to_numpy(), "lon": lon, "lat": lat, "x": x, "y": y,
                              "cell_x": np.floor(x / READINESS_CELL_M).astype(int), "cell_y": np.floor(y / READINESS_CELL_M).astype(int),
                              "node": nodes, "snap_m": snap_m, "jurisdiction": pool["jurisdiction"].to_numpy()[nearest_call],
                              "focal": pool["focal"].to_numpy()[nearest_call]})
    positions = node_position[locations["node"]].to_numpy()
    location_seconds = pd.DataFrame({engine_id: seconds_from_node(int(node))[positions] for engine_id, node in home_nodes.items()},
                                    index=locations.index)
    engine_table = engines.set_index("engine_id").loc[location_seconds.columns]
    station_of_engine = engine_table["home_station"].astype(object).where(engine_table["home_station"].notna(), engine_table.index.to_series())
    other_side = (engine_table["engine_agency"] == "GFR").to_numpy()[None, :] != (locations["jurisdiction"] == "GFR").to_numpy()[:, None]
    locations["first_due_station"] = station_of_engine.to_numpy()[np.argmin(np.where(other_side, np.inf, location_seconds.to_numpy()), axis=1)]
    if verbose:
        show_table("residential locations (2020 blocks with residents)", locations, verbose)
        area_residents = locations.groupby("first_due_station")["residents"].sum().sort_values()
        print(f"first-due station areas with residents: {len(area_residents)}; residents per area: fewest {area_residents.index[0]} "
              f"{int(area_residents.iloc[0]):,}, median {int(area_residents.median()):,}, most {area_residents.index[-1]} {int(area_residents.iloc[-1]):,}")
        print(f"blocks with residents {len(locations):,}, residents {int(locations['residents'].sum()):,} in "
              f"{locations.groupby(['cell_x', 'cell_y']).ngroups:,} cells of {READINESS_CELL_M} m; snap distance median "
              f"{locations['snap_m'].median():.0f} m, P90 {locations['snap_m'].quantile(0.9):.0f} m; residents on the GFR side "
              f"{locations.loc[locations['jurisdiction'] == 'GFR', 'residents'].sum() / locations['residents'].sum():.1%}; "
              f"unreachable from some station: {int(np.isinf(location_seconds.to_numpy()).any(axis=1).sum())} blocks\n")
    return locations, location_seconds


def draw_source_calls(pool, day, fire, band, size, rng, candidates):
    """Section 13: `size` real calls (incident ids, drawn with replacement) for a new call on day `day`: same group (fire),
    same SCENARIO_BAND_HOURS block, received within SCENARIO_WINDOW_DAYS of the day (any kind of day). candidates: a dict
    kept by the caller, (day, fire, band) -> pool positions.

    Example: draw_source_calls(pool, 0, False, 3, 2, np.random.default_rng(1), {}) -> two incident ids received 12:00-15:59
             within 14 days of 1 October, e.g. Index(['100824-0213', '101124-0287'])
    """
    key = (day, fire, band)
    if key not in candidates:
        candidates[key] = np.flatnonzero((np.abs(pool["day"].to_numpy() - day) <= SCENARIO_WINDOW_DAYS) & (pool["fire"].to_numpy() == fire)
                                         & (pool["band"].to_numpy() == band))
    return pool.index[rng.choice(candidates[key], size=size)]


def fit_call_rates(pool, verbose=VERBOSE):
    """Section 13: how many calls arrive, from the pool. counts: calls on each training day (rows: days since DAY_ZERO)
    by group and hour (columns: fire False = other calls, True = study fire calls; hours 0-23). dispersion: per group,
    the day-to-day spread beyond chance, phi of a negative binomial (variance = mean + phi x mean^2), fitted by moments on
    each day's total against the mean total of the other days of the same kind (weekday or weekend) within
    SCENARIO_WINDOW_DAYS; 0 when the spread is no wider than chance.

    Example: counts, dispersion = fit_call_rates(pool)
             counts.shape -> (182, 48); dispersion -> {False: 0.01, True: 0.06}, roughly
    """
    days = np.arange((TRAINING_MONTHS_END - DAY_ZERO).days)
    columns = pd.MultiIndex.from_product([[False, True], range(24)], names=["fire", "hour"])
    counts = (pool.groupby(["day", "fire", "hour"]).size().unstack(["fire", "hour"])
              .reindex(index=days, columns=columns).fillna(0).astype(int))
    weekend = (DAY_ZERO + pd.to_timedelta(days, unit="D")).dayofweek >= 5
    dispersion, check = {}, {}
    for fire in (False, True):
        totals = counts.xs(fire, axis=1, level="fire").sum(axis=1).to_numpy(dtype=float)
        means = np.array([totals[(np.abs(days - day) <= SCENARIO_WINDOW_DAYS) & (weekend == weekend[day]) & (days != day)].mean()
                          for day in days])
        dispersion[fire] = max(0.0, float(np.sum((totals - means) ** 2 - means) / np.sum(means ** 2)))
        check["study fire calls" if fire else "other calls"] = {
            "mean per day": totals.mean(), "variance of the day's total around its window mean": np.mean((totals - means) ** 2),
            "chance alone (mean)": means.mean(), "phi": dispersion[fire]}
    if verbose:
        show_table("calls per training day by (fire, hour)", counts, verbose)
        print(f"day-to-day spread (negative binomial phi; variance = mean + phi x mean^2):\n{pd.DataFrame(check).T.round(4).to_string()}\n")
    return counts, dispersion


def expected_hourly_counts(counts, day):
    """Section 13: the mean calls per hour on scenario day `day` (days since DAY_ZERO): each (fire, hour) count averaged
    over the training days of the same kind (weekday or weekend) within SCENARIO_WINDOW_DAYS of it.

    Example: expected_hourly_counts(counts, 0)[(False, 12)] -> 4.9  # other calls, 12:00-13:00 on Tuesday 1 October, roughly
    """
    days = counts.index.to_numpy()
    weekend = (DAY_ZERO + pd.to_timedelta(days, unit="D")).dayofweek >= 5
    near = (np.abs(days - day) <= SCENARIO_WINDOW_DAYS) & (weekend == ((DAY_ZERO + pd.Timedelta(days=int(day))).dayofweek >= 5))
    return counts[near].mean()


def generate_calls(pool, counts, dispersion, days, seed, verbose=VERBOSE):
    """Section 13 (the nominal scenario): calls for `days` (days since DAY_ZERO, in the training months).
    - Other (non-fire) calls: each day copies every other call of one real training day of the same kind (weekday or
      weekend) within SCENARIO_WINDOW_DAYS, drawn at random: each call at its time of day, with its type, location and
      fate. A whole real day keeps how its calls fall against each other in time, which calls drawn one by one lose
      (Section 13.2).
    - Study fire calls: each day draws a day factor from a gamma with mean 1 and variance phi (busy days; dispersion from
      fit_call_rates). Each hour then gets a Poisson number of calls with mean = day factor x expected_hourly_counts, at
      uniform whole seconds of the hour. Each copies a real study fire call received in the same SCENARIO_BAND_HOURS
      block within SCENARIO_WINDOW_DAYS (any kind of day): its type, location and fate.
    Returns the calls (incident_id, received_s, source_incident, fire, large_fire, kind, location: -1 = the source's own
    location), sorted by time, and the day factors (rows: days; columns: fire): for other calls, the copied day's total
    over the day's expected total; for study fire calls, the gamma draw.

    Example: calls, day_factors = generate_calls(pool, counts, dispersion, range(7), seed=101)
             calls.iloc[0].tolist() -> ['syn101-000000', 1873.0, '100424-0016', False, False, 'nominal', -1]  # day 0 copies day 3
    """
    rng = np.random.default_rng(seed)
    training_days = counts.index.to_numpy()
    training_weekend = (DAY_ZERO + pd.to_timedelta(training_days, unit="D")).dayofweek >= 5
    other_pool = pool[~pool["fire"]]
    candidates, moments, sources, groups, factors, copied_days = {}, [], [], [], {}, {}
    for day in days:
        mean_counts = expected_hourly_counts(counts, day)
        same_kind = training_weekend == ((DAY_ZERO + pd.Timedelta(days=int(day))).dayofweek >= 5)
        copied_days[day] = int(rng.choice(training_days[(np.abs(training_days - day) <= SCENARIO_WINDOW_DAYS) & same_kind]))
        copied = other_pool[other_pool["day"] == copied_days[day]]
        sources.extend(copied.index)
        moments.extend(copied["received_s"].to_numpy() + (day - copied_days[day]) * 86400)
        groups.extend([False] * len(copied))
        factors[(day, False)] = len(copied) / mean_counts.xs(False).sum()
        phi = dispersion[True]
        factors[(day, True)] = rng.gamma(1 / phi, phi) if phi > 0 else 1.0
        numbers = rng.poisson(factors[(day, True)] * mean_counts.xs(True).to_numpy())
        for hour in np.flatnonzero(numbers):
            sources.extend(draw_source_calls(pool, day, True, hour // SCENARIO_BAND_HOURS, numbers[hour], rng, candidates))
            moments.extend(day * 86400 + hour * 3600 + rng.integers(0, 3600, size=numbers[hour]))
            groups.extend([True] * numbers[hour])
    calls = pd.DataFrame({"received_s": np.array(moments, dtype=float), "source_incident": sources, "fire": np.array(groups, dtype=bool)})
    calls = calls.sort_values("received_s", kind="stable").reset_index(drop=True)
    calls.insert(0, "incident_id", [f"syn{seed}-{n:06d}" for n in range(len(calls))])
    calls["large_fire"], calls["kind"], calls["location"] = False, "nominal", -1
    day_factors = pd.Series(factors).unstack()
    if verbose:
        show_table(f"generated calls (seed {seed})", calls, verbose)
        first_week = list(days)[:7]
        week = pd.DataFrame({"day": first_week, "copied day": [copied_days[day] for day in first_week]})
        week.insert(1, "weekday", (DAY_ZERO + pd.to_timedelta(week["day"], unit="D")).dt.day_name())
        week["copied weekday"] = (DAY_ZERO + pd.to_timedelta(week["copied day"], unit="D")).dt.day_name()
        week["other calls copied"] = [int((other_pool["day"] == copied_days[day]).sum()) for day in first_week]
        week["expected other calls"] = [expected_hourly_counts(counts, day).xs(False).sum() for day in first_week]
        week["fire day factor"] = [day_factors.at[day, True] for day in first_week]
        first_day = first_week[0]
        expected = expected_hourly_counts(counts, first_day)
        made = calls[calls["received_s"] // 86400 == first_day].assign(hour=lambda c: (c["received_s"] % 86400 // 3600).astype(int))
        by_hour = pd.DataFrame({"expected other calls": expected.xs(False), "copied other calls": made[~made["fire"]].groupby("hour").size(),
                                "expected fire calls": expected.xs(True), "generated fire calls": made[made["fire"]].groupby("hour").size()}).fillna(0)
        source = pool.loc[calls["source_incident"]]
        examples = calls.head(5).assign(source_day=source["day"].to_numpy()[:5], source_hour=source["hour"].to_numpy()[:5],
                                        source_protocol=source["protocol"].to_numpy()[:5], source_runs=source["runs"].to_numpy()[:5],
                                        cell=list(zip(source["cell_x"].to_numpy()[:5], source["cell_y"].to_numpy()[:5])))
        other_calls, call_day = ~calls["fire"].to_numpy(), calls["received_s"].to_numpy() // 86400
        call_weekend = (DAY_ZERO + pd.to_timedelta(call_day, unit="D")).dayofweek >= 5
        print(f"the first week's copied days (other calls) and fire day factors:\n{week.round(3).to_string(index=False)}\n"
              f"day {first_day}, calls by hour (fire: expected before the day factor):\n{by_hour.round(2).T.to_string()}\n"
              f"first calls and the real calls they copy:\n{examples.to_string()}\n"
              f"calls {len(calls):,} over {len(day_factors)} days ({len(calls) / len(day_factors):.1f} a day; study fire calls "
              f"{int(calls['fire'].sum()):,}); largest gap between a call's day and its source's day: "
              f"{int(np.abs(call_day - source['day'].to_numpy()).max())} days\n"
              f"other calls: one copied day per day {bool(pd.Series(source['day'].to_numpy()[other_calls]).groupby(call_day[other_calls]).nunique().max() == 1)}, "
              f"same kind of day {bool((call_weekend == source['weekend'].to_numpy())[other_calls].all())}, same time of day "
              f"{bool(((calls['received_s'].to_numpy() - source['received_s'].to_numpy()) % 86400 == 0)[other_calls].all())}; day factor (copied total / "
              f"expected) mean {day_factors[False].mean():.3f}, SD {day_factors[False].std():.3f}, distinct real days used "
              f"{len(set(copied_days.values()))} of {len(training_days)}\n"
              f"study fire calls: same block of hours as the source "
              f"{bool(((calls['received_s'] % 86400 // 3600) // SCENARIO_BAND_HOURS == source['band'].to_numpy())[~other_calls].all())}; "
              f"day factor mean {day_factors[True].mean():.3f}, SD {day_factors[True].std():.3f} (gamma SD sqrt(phi) {np.sqrt(dispersion[True]):.3f})\n")
    return calls, day_factors


def add_demand_growth(calls, day_factors, pool, counts, locations, factor, seed, verbose=VERBOSE):
    """Section 13 (more or fewer calls): other (non-fire) calls times `factor`; study fire calls are unchanged. Below 1,
    each other call is dropped with probability 1 - factor. Above 1, each day and hour gets a Poisson number of extra
    other calls with mean (factor - 1) x the day's other-call factor (generate_calls: the copied day's total over the
    expected total) x the hour's expected count, at uniform whole seconds of the
    hour. Each extra call is placed where people live: a residential block (build_resident_locations) drawn in proportion
    to its 2020 residents, every populated block eligible. Its type and fate come from a real other call drawn as for a
    nominal call (draw_source_calls: same block of hours, within SCENARIO_WINDOW_DAYS).

    Example: grown = add_demand_growth(calls, day_factors, pool, counts, locations, 1.3, seed=101)
             (~grown["fire"]).sum() / (~calls["fire"]).sum() -> 1.30, roughly; grown["location"] >= 0 for the extra calls
    """
    rng = np.random.default_rng([seed, 1])
    other = ~calls["fire"].to_numpy()
    if factor <= 1:
        changed = calls[~other | (rng.random(len(calls)) < factor)].reset_index(drop=True)
        extra = changed.iloc[:0]
    else:
        weight = (locations["residents"] / locations["residents"].sum()).to_numpy()
        candidates, moments, sources, places = {}, [], [], []
        for day in day_factors.index:
            numbers = rng.poisson((factor - 1) * day_factors.at[day, False] * expected_hourly_counts(counts, day).xs(False).to_numpy())
            for hour in np.flatnonzero(numbers):
                sources.extend(draw_source_calls(pool, day, False, hour // SCENARIO_BAND_HOURS, numbers[hour], rng, candidates))
                places.extend(rng.choice(len(locations), size=numbers[hour], p=weight))
                moments.extend(day * 86400 + hour * 3600 + rng.integers(0, 3600, size=numbers[hour]))
        seed_label = calls["incident_id"].iloc[0].split("-")[0]
        extra = pd.DataFrame({"incident_id": [f"{seed_label}-g{n:06d}" for n in range(len(moments))], "received_s": np.array(moments, dtype=float),
                              "source_incident": sources, "fire": False, "large_fire": False, "kind": "growth",
                              "location": np.array(places, dtype=int)})
        changed = pd.concat([calls, extra], ignore_index=True).sort_values("received_s", kind="stable").reset_index(drop=True)
    if verbose:
        busiest = pool[~pool["fire"]].groupby(["cell_x", "cell_y"]).size().nlargest(20).index
        source_cells = pd.MultiIndex.from_arrays([pool.loc[calls.loc[other, "source_incident"], "cell_x"], pool.loc[calls.loc[other, "source_incident"], "cell_y"]])
        before, after = int(other.sum()), int((~changed["fire"]).sum())
        message = (f"--- other calls x{factor}: {before:,} -> {after:,} ({after / before:.3f}); study fire calls {int(calls['fire'].sum()):,} -> "
                   f"{int(changed['fire'].sum()):,}\nshare in the record's 20 busiest cells: all other calls {source_cells.isin(busiest).mean():.3f}")
        if len(extra):
            placed = locations.loc[extra["location"]]
            source = pool.loc[extra["source_incident"]]
            message += (f", the extra calls {pd.MultiIndex.from_arrays([placed['cell_x'], placed['cell_y']]).isin(busiest).mean():.3f} (placed by residents)\n"
                        f"extra calls' sources: within {SCENARIO_WINDOW_DAYS} days {bool((np.abs(extra['received_s'].to_numpy() // 86400 - source['day'].to_numpy()) <= SCENARIO_WINDOW_DAYS).all())}, "
                        f"same block of hours {bool(((extra['received_s'] % 86400 // 3600) // SCENARIO_BAND_HOURS == source['band'].to_numpy()).all())}; "
                        f"blocks used {placed.index.nunique():,} of {len(locations):,}\nfirst extra calls (type from the source, place from a block):\n"
                        f"{extra.head(3).assign(source_protocol=source['protocol'].to_numpy()[:3], block_residents=placed['residents'].to_numpy()[:3], block_cell=list(zip(placed['cell_x'].to_numpy()[:3], placed['cell_y'].to_numpy()[:3]))).to_string()}")
        print(message + "\n")
    return changed


def add_short_surges(calls, pool, locations, days, extra_calls, seed, verbose=VERBOSE):
    """Section 13 (a short surge; a stress test, since the record holds no local burst beyond chance, at most 7 other
    calls within SURGE_RADIUS_M in SURGE_HOURS): SURGES_PER_DAY times a day, extra_calls other (non-fire) calls are
    added within SURGE_RADIUS_M of a centre over SURGE_HOURS from a uniform whole second of the day, each at a uniform
    whole second of the surge. Surges visit the first-due station areas in turn, in an order drawn per seed, so every
    area is tested equally often (the risk map of Usanov et al. 2019). The centre is a residential block of that area
    drawn in proportion to its residents; each call is placed at a block inside the disk drawn the same way, with its
    type and fate drawn as for a nominal call (draw_source_calls). Each surge draws from its own random stream ([seed,
    2, surge number]) for max(SURGE_EXTRA_CALLS) calls and keeps the first extra_calls, so a smaller surge is the start
    of a larger one at the same place and time; a call's incident id names its surge and its rank in it, so it also
    keeps its travel draw at every size. Returns the calls with the extra ones (kind 'surge') and one row per surge (with
    the disk's usual other calls over SURGE_HOURS in the training months, for scale).

    Example: surged, surges = add_short_surges(calls, pool, locations, range(182), 12, seed=101)
             surges.iloc[0] -> day 0, start, end, first-due station, centre block, cell, city flag, extra calls 12, usual calls
    """
    areas = np.sort(locations["first_due_station"].unique())
    order = np.random.default_rng([seed, 2]).permutation(len(areas))
    station = locations["first_due_station"].to_numpy()
    block_x, block_y, residents = locations["x"].to_numpy(), locations["y"].to_numpy(), locations["residents"].to_numpy(dtype=float)
    other = pool[~pool["fire"]]
    other_x, other_y = other["x"].to_numpy(), other["y"].to_numpy()
    training_hours = (TRAINING_MONTHS_END - DAY_ZERO).days * 24
    last_day, most = max(days), max(SURGE_EXTRA_CALLS)
    seed_label = calls["incident_id"].iloc[0].split("-")[0]
    candidates, surge_rows, incident_ids, moments, sources, places = {}, [], [], [], [], []
    for position, day in enumerate(days):
        for slot in range(SURGES_PER_DAY):
            number = position * SURGES_PER_DAY + slot
            area = areas[order[number % len(areas)]]
            rng = np.random.default_rng([seed, 2, number])
            in_area = np.flatnonzero(station == area)
            centre = int(rng.choice(in_area, p=residents[in_area] / residents[in_area].sum()))
            inside = np.flatnonzero(np.hypot(block_x - block_x[centre], block_y - block_y[centre]) <= SURGE_RADIUS_M)
            start = day * 86400 + int(rng.integers(0, 86400))
            surge_moments = start + rng.integers(0, SURGE_HOURS * 3600, size=most)
            surge_places = rng.choice(inside, size=most, p=residents[inside] / residents[inside].sum())
            surge_sources = [draw_source_calls(pool, int(min(moment // 86400, last_day)), False, int(moment % 86400 // 3600) // SCENARIO_BAND_HOURS,
                                               1, rng, candidates)[0] for moment in surge_moments]
            incident_ids.extend(f"{seed_label}-s{number:05d}-{rank:02d}" for rank in range(extra_calls))   # same id (so same travel draws) at every size
            moments.extend(surge_moments[:extra_calls])
            places.extend(surge_places[:extra_calls])
            sources.extend(surge_sources[:extra_calls])
            surge_rows.append({"day": day, "start_s": start, "end_s": start + SURGE_HOURS * 3600, "first_due_station": area, "centre_block": centre,
                               "cell_x": int(locations.at[centre, "cell_x"]), "cell_y": int(locations.at[centre, "cell_y"]),
                               "focal": bool(locations.at[centre, "focal"]), "extra_calls": extra_calls,
                               "usual_other_calls_in_disk": int(np.sum(np.hypot(other_x - block_x[centre], other_y - block_y[centre]) <= SURGE_RADIUS_M))
                                                            / training_hours * SURGE_HOURS})
    extra = pd.DataFrame({"incident_id": incident_ids, "received_s": np.array(moments, dtype=float), "source_incident": sources, "fire": False,
                          "large_fire": False, "kind": "surge", "location": np.array(places, dtype=int)})
    surges = pd.DataFrame(surge_rows)
    if verbose:
        show_table(f"short surges ({extra_calls} extra calls each)", surges, verbose)
        surge_of_call = np.repeat(np.arange(len(surges)), extra_calls)
        centre_of_call = surges["centre_block"].to_numpy()[surge_of_call]
        placed = locations.loc[extra["location"]]
        distance = np.hypot(placed["x"].to_numpy() - block_x[centre_of_call], placed["y"].to_numpy() - block_y[centre_of_call])
        source = pool.loc[extra["source_incident"]]
        first = extra.head(extra_calls).assign(minutes_after_start=(extra["received_s"].head(extra_calls) - surges.at[0, "start_s"]) / 60,
                                               source_protocol=source["protocol"].to_numpy()[:extra_calls],
                                               metres_from_centre=distance[:extra_calls].round(0))
        per_area = surges.groupby("first_due_station").size()
        print(f"surges {len(surges)} over {len(per_area)} first-due areas ({per_area.min()}-{per_area.max()} each); extra calls {len(extra):,}; usual other "
              f"calls in the disk over {SURGE_HOURS} h: median {surges['usual_other_calls_in_disk'].median():.2f}, so a surge is "
              f"{np.median(extra_calls / surges['usual_other_calls_in_disk'].clip(lower=0.01)):.0f} times the usual (median)\n"
              f"first surge ({surges.at[0, 'first_due_station']}, day {surges.at[0, 'day']}, from second {surges.at[0, 'start_s']:,} of the period):\n"
              f"{first[['incident_id', 'minutes_after_start', 'source_incident', 'source_protocol', 'location', 'metres_from_centre']].round(1).to_string()}\n"
              f"checks: every call within {SURGE_RADIUS_M} m of its centre {bool((distance <= SURGE_RADIUS_M).all())}, inside its surge's "
              f"{SURGE_HOURS} h {bool(((extra['received_s'].to_numpy() >= surges['start_s'].to_numpy()[surge_of_call]) & (extra['received_s'].to_numpy() < surges['end_s'].to_numpy()[surge_of_call])).all())}, "
              f"source in the same block of hours {bool(((extra['received_s'] % 86400 // 3600) // SCENARIO_BAND_HOURS == source['band'].to_numpy()).all())} "
              f"and within {SCENARIO_WINDOW_DAYS} days {bool((np.abs(np.minimum(extra['received_s'].to_numpy() // 86400, last_day) - source['day'].to_numpy()) <= SCENARIO_WINDOW_DAYS).all())}\n")
    return pd.concat([calls, extra], ignore_index=True).sort_values("received_s", kind="stable").reset_index(drop=True), surges


def readiness_loss_in_windows(segments, starts_s, ends_s):
    """Section 13.5: the time-weighted fire readiness loss (R - R_full, seconds) over each window [start, end), from
    fire_readiness's segments (contiguous). Inside a segment the loss is its middle value minus waiting_weight x (time -
    middle) (waiting_weight 0 without waits), so a window that cuts a segment is integrated exactly.

    Example: readiness_loss_in_windows(segments, [0.0], [7200.0]) -> the loss over the first two hours, in seconds
             toys: segments 0-60 s with a loss of 1 min and 60-120 s with 3 min; window 30-90 s -> (30 x 60 + 30 x 180) / 60 = 120 s
                   one segment 0-600 s, loss 5 min at its middle, waiting_weight 1 (falls from 10 to 0 min); window 0-300 s -> 450 s
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


def add_outages(engines, days, engine_count, seed, verbose=VERBOSE):
    """Section 13 (fewer engines): each day, engine_count engines drawn at random among those in service all that day are
    out of service from midnight to midnight. Returns the outages (engine_id, start_s, end_s; seconds since DAY_ZERO).

    Example: add_outages(engines, range(2), 2, seed=101) -> 4 rows, e.g. E24 0-86,400 s, E5 0-86,400 s, then day 1
    """
    rng = np.random.default_rng([seed, 3, engine_count])
    rows = []
    for day in days:
        start, end = DAY_ZERO + pd.Timedelta(days=day), DAY_ZERO + pd.Timedelta(days=day + 1)
        in_service = engines.loc[(engines["in_service_from"] <= start) & (engines["in_service_to"] >= end), "engine_id"].to_numpy()
        rows += [(engine_id, day * 86400.0, (day + 1) * 86400.0) for engine_id in rng.choice(in_service, size=engine_count, replace=False)]
    outages = pd.DataFrame(rows, columns=["engine_id", "start_s", "end_s"])
    if verbose:
        print(f"--- {engine_count} engines out each day: {len(outages):,} engine-days; first days:\n{outages.head(2 * engine_count).to_string()}\n"
              f"engine-days per engine (top five): {outages['engine_id'].value_counts().head(5).to_dict()}\n")
    return outages


def close_station(engines, station, days, verbose=VERBOSE):
    """Section 13 (a closed station): every engine of `station` out of service from the first of `days` to the end of the
    last. Returns the outages (engine_id, start_s, end_s).

    Example: close_station(engines, "GFR_03", range(182)) -> one row: E3, 0 to 15,724,800 s
    """
    outages = pd.DataFrame({"engine_id": engines.loc[engines["home_station"].eq(station), "engine_id"].to_numpy(),
                            "start_s": min(days) * 86400.0, "end_s": (max(days) + 1) * 86400.0})
    if verbose:
        print(f"--- station {station} closed:\n{outages.to_string()}\n")
    return outages


def add_large_fires(calls, pool, days, seed, verbose=VERBOSE):
    """Section 13 (a large fire): LARGE_FIRES_PER_WEEK in each week (days since DAY_ZERO // 7), at a uniform whole second
    of the week, at the location of a real training-month structure fire (STRUCTURE_FIRE_PROTOCOL, with an engine on
    scene) drawn at random; build_scenario gives it LARGE_FIRE_ENGINES runs that all arrive and stay LARGE_FIRE_ON_SCENE_S.
    Returns the calls with the large fires (kind 'large fire').

    Example: add_large_fires(calls, pool, range(182), seed=101)["large_fire"].sum() -> 26
    """
    rng = np.random.default_rng([seed, 4])
    sites = pool.index[pool["protocol"].eq(STRUCTURE_FIRE_PROTOCOL).fillna(False).to_numpy(dtype=bool) & (pool["arrival_row"] >= 0).to_numpy()]
    moments, sources = [], []
    for week in sorted({day // 7 for day in days}):
        week_days = [day for day in days if day // 7 == week]
        for _ in range(LARGE_FIRES_PER_WEEK):
            moments.append(min(week_days) * 86400 + int(rng.integers(0, len(week_days) * 86400)))
            sources.append(sites[rng.integers(len(sites))])
    seed_label = calls["incident_id"].iloc[0].split("-")[0]
    extra = pd.DataFrame({"incident_id": [f"{seed_label}-L{n:04d}" for n in range(len(moments))], "received_s": np.array(moments, dtype=float),
                          "source_incident": sources, "fire": True, "large_fire": True, "kind": "large fire", "location": -1})
    if verbose:
        print(f"--- large fires: {len(extra)} ({LARGE_FIRE_ENGINES} engines for {LARGE_FIRE_ON_SCENE_S / 3600:.1f} h each), at real "
              f"structure fire locations ({len(sites)} candidates); first three:\n"
              f"{extra.head(3).assign(cell=list(zip(pool.loc[extra['source_incident'].head(3), 'cell_x'], pool.loc[extra['source_incident'].head(3), 'cell_y']))).to_string()}\n")
    return pd.concat([calls, extra], ignore_index=True).sort_values("received_s", kind="stable").reset_index(drop=True)


def build_scenario(calls, outages, pool, locations, location_seconds, dispatches, m0_runs, engines, network_seconds, network_km, call_nodes,
                   travel_model, aid_protocols, quint_penalty_by_type, seed, verbose=VERBOSE):
    """Section 13: a scenario as the inputs simulate needs, built from real runs so that every rule of the M0.4 setting
    applies unchanged. Each new call copies the runs of its real incident (the record's 32 columns and the M0-repaired
    times), every time moved by the same whole-second shift (new call time minus the real one), under its new incident
    id. A call placed at a residential block (calls["location"] >= 0, from build_resident_locations) takes the block's
    position, agency side, city flag, network node and network seconds instead of its source's. A large fire copies its
    structure fire's first run to arrive LARGE_FIRE_ENGINES times; all of them arrive and stay LARGE_FIRE_ON_SCENE_S on
    scene. Each outage (engine_id, start_s, end_s) is a stand-in run with no location (protocol OUT_OF_SERVICE_PROTOCOL)
    that holds its engine from start to end: like any call without a location, the protocol sends that engine when it is
    free and otherwise waits for it, ahead of every waiting call (OUT_OF_SERVICE_WAIT_RANK), and the stand-in still ends
    at end_s.
    Returns a dict: dispatches (the record's 32 columns), runs (seconds since DAY_ZERO), source_row (the record row each
    run copies; -1 for a stand-in), large_fire (per run), travel (modeled travel from every station:
    estimate_network_travel_seconds at the scenario's dispatch times, then apply_agency_boundaries), call_nodes,
    spread_quantile (one draw per run from seed and its incident id, so a call keeps its draws when a condition adds
    calls), fates (call_fates), and per run inside_city, wait_rank (SOC priority; stand-ins first),
    quint_penalty (M0.4, by call type), brush (F82) and low_acuity (research.md 3.2).

    Example: scenario = build_scenario(calls, no_outages, pool, locations, location_seconds, dispatches, m0_runs, engines,
                                       network_seconds, network_km, call_nodes, network_travel_model, aid_protocols,
                                       quint_penalty_by_type, seed=101)
             scenario["dispatches"].shape -> (18500, 32), roughly, for a nominal October-March
    """
    time_columns = ["call_received_at"] + TIME_COLUMNS
    copied = []                                                  # (source row, new incident id, shift in seconds, large fire, block)
    for call in calls.itertuples(index=False):
        rows = [pool.at[call.source_incident, "arrival_row"]] * LARGE_FIRE_ENGINES if call.large_fire else pool.at[call.source_incident, "run_rows"]
        shift_s = call.received_s - pool.at[call.source_incident, "received_s"]
        copied += [(row, call.incident_id, shift_s, call.large_fire, call.location) for row in rows]
    copied = pd.DataFrame(copied, columns=["source_row", "incident_id", "shift_s", "large_fire", "location"])
    table = dispatches.loc[copied["source_row"]].reset_index(drop=True)
    runs = m0_runs.loc[copied["source_row"]].reset_index(drop=True)
    for column in time_columns:
        table[column] = table[column] + pd.to_timedelta(copied["shift_s"], unit="s")
        runs[column] = runs[column] + copied["shift_s"]
    table["incident_id"] = copied["incident_id"].astype(dispatches["incident_id"].dtype)
    runs["incident_id"] = copied["incident_id"].to_numpy()
    copied_placed = copied["location"].to_numpy() >= 0
    if copied_placed.any():                                      # the block's position, agency side and city flag
        block = locations.loc[copied.loc[copied_placed, "location"]]
        for column, source in [("incident_lon", "lon"), ("incident_lat", "lat"), ("jurisdiction", "jurisdiction"), ("focal", "focal")]:
            table.loc[copied_placed, column] = pd.Series(block[source].to_numpy(), index=table.index[copied_placed], dtype=dispatches[column].dtype)
    large = copied["large_fire"].to_numpy(dtype=bool)
    if large.any():                                              # every engine of a large fire arrives and stays LARGE_FIRE_ON_SCENE_S
        table.loc[large, "cleared_at"] = table.loc[large, "arrived_at"] + pd.Timedelta(seconds=LARGE_FIRE_ON_SCENE_S)
        runs.loc[large, "cleared_at"] = runs.loc[large, "arrived_at"] + LARGE_FIRE_ON_SCENE_S
        table.loc[large, "on_scene_s"] = LARGE_FIRE_ON_SCENE_S
        table.loc[large, "n_engines_sent"] = LARGE_FIRE_ENGINES
        table.loc[large, "arrival_order"] = table[large].groupby("incident_id").cumcount().to_numpy() + 1
        table.loc[large, "cancelled"] = False
    if len(outages):                                             # stand-in runs, typed like the record (a real row as template)
        stand_in = dispatches.iloc[[0] * len(outages)].reset_index(drop=True)
        home = engines.set_index("engine_id").loc[outages["engine_id"]]
        start = DAY_ZERO + pd.to_timedelta(outages["start_s"].to_numpy(), unit="s")
        values = {"incident_id": [f"out{seed}-{n:05d}" for n in range(len(outages))], "problem": "engine out of service",
                  "call_type": "Out of service", "protocol": OUT_OF_SERVICE_PROTOCOL, "determinant_level": pd.NA, "determinant_code": pd.NA,
                  "priority": pd.NA, "n_engines_sent": 1, "incident_lon": np.nan, "incident_lat": np.nan,
                  "jurisdiction": np.where(home["engine_agency"].eq("GFR"), "GFR", "ACFR"),          # its own agency's side of the line
                  "focal": False, "engine_id": outages["engine_id"].to_numpy(), "engine_type": home["engine_type"].to_numpy(),
                  "engine_agency": home["engine_agency"].to_numpy(), "engine_lon_at_dispatch": home["home_lon"].to_numpy(),
                  "engine_lat_at_dispatch": home["home_lat"].to_numpy(), "from_station": home["home_station"].astype(object).to_numpy(),
                  "at_home": True, "call_received_at": start, "dispatched_at": start, "enroute_at": pd.NaT, "arrived_at": pd.NaT,
                  "cleared_at": DAY_ZERO + pd.to_timedelta(outages["end_s"].to_numpy(), unit="s"), "turnout_s": pd.NA, "travel_s": pd.NA,
                  "on_scene_s": pd.NA, "arrival_order": pd.NA, "staged": False, "cancelled": True, "auto_aid": pd.NA, "qc_note": ""}
        for column, value in values.items():
            stand_in[column] = pd.Series(value if np.ndim(value) else [value] * len(stand_in), dtype=dispatches[column].dtype)
        stand_in_runs = pd.DataFrame({"incident_id": stand_in["incident_id"].astype(object).to_numpy(), "engine_id": outages["engine_id"].to_numpy(),
                                      "call_received_at": outages["start_s"].to_numpy(), "dispatched_at": outages["start_s"].to_numpy(),
                                      "enroute_at": np.nan, "arrived_at": np.nan, "cleared_at": outages["end_s"].to_numpy()})
        table = pd.concat([table, stand_in], ignore_index=True)
        runs = pd.concat([runs, stand_in_runs[runs.columns]], ignore_index=True)
        copied = pd.concat([copied, pd.DataFrame({"source_row": -1, "incident_id": stand_in["incident_id"], "shift_s": np.nan,
                                                  "large_fire": False, "location": -1})], ignore_index=True)
    order = np.argsort(runs["call_received_at"].to_numpy(), kind="stable")
    table, runs = table.iloc[order].reset_index(drop=True), runs.iloc[order].reset_index(drop=True)
    source_row = pd.Series(copied["source_row"].to_numpy()[order], index=table.index)
    location = copied["location"].to_numpy()[order]
    placed = location >= 0
    seconds_rows = network_seconds.reindex(source_row.to_numpy()).set_axis(table.index)
    km_rows = network_km.reindex(source_row.to_numpy()).set_axis(table.index)
    seconds_rows.loc[placed] = location_seconds.loc[location[placed], seconds_rows.columns].to_numpy()
    km_rows.loc[placed] = np.nan                                 # the linear travel model uses network seconds only
    nodes = call_nodes.reindex(source_row.to_numpy()).set_axis(table.index)
    nodes.loc[placed, "node"] = locations.loc[location[placed], "node"].to_numpy()
    nodes.loc[placed, "snap_m"] = locations.loc[location[placed], "snap_m"].to_numpy()
    stand_in_row = table["protocol"].eq(OUT_OF_SERVICE_PROTOCOL).fillna(False)
    run_number = runs.groupby("incident_id", sort=False).cumcount().to_numpy()
    spread_quantile = [np.random.default_rng([seed, zlib.crc32(str(incident).encode())]).random(number + 1)[number]
                       for incident, number in zip(runs["incident_id"], run_number)]
    travel = apply_agency_boundaries(estimate_network_travel_seconds(table, engines, seconds_rows, km_rows, travel_model, verbose=False),
                                     table, engines, aid_protocols, verbose=False)
    call_type_for_quint = table["protocol"].map(QUINT_RULE_CALL_TYPES).fillna("all other calls")
    scenario = {
        "dispatches": table, "runs": runs, "source_row": source_row,
        "large_fire": pd.Series(copied["large_fire"].to_numpy(dtype=bool)[order], index=table.index), "travel": travel,
        "call_nodes": nodes, "spread_quantile": pd.Series(spread_quantile, index=table.index),
        "fates": call_fates(runs), "inside_city": table["focal"].fillna(False).astype(bool),
        "wait_rank": table["priority"].astype("string").map(PRIORITY_RANK).fillna(len(PRIORITY_RANK)).mask(stand_in_row, OUT_OF_SERVICE_WAIT_RANK),
        "quint_penalty": call_type_for_quint.map(quint_penalty_by_type), "brush": table["protocol"].eq(BRUSH_FIRE_PROTOCOL).fillna(False),
        "low_acuity": ((table["call_type"].eq("Medical") & table["priority"].eq("Low")).fillna(False)
                       | table["protocol"].isin(LOW_ACUITY_SERVICE_PROTOCOLS).fillna(False))}
    if verbose:
        show_table(f"scenario dispatch table (seed {seed})", table, verbose)
        copies = (source_row.to_numpy() >= 0) & ~placed
        record = dispatches.loc[source_row[copies & ~scenario["large_fire"].to_numpy()]]
        kept = table[copies & ~scenario["large_fire"].to_numpy()]
        duration_change = ((kept["cleared_at"] - kept["dispatched_at"]).to_numpy() - (record["cleared_at"] - record["dispatched_at"]).to_numpy())
        record_travel = apply_agency_boundaries(estimate_network_travel_seconds(dispatches.loc[source_row[copies]], engines, network_seconds.loc[source_row[copies]],
                                                                                network_km.loc[source_row[copies]], travel_model, verbose=False),
                                                dispatches.loc[source_row[copies]], engines, aid_protocols, verbose=False)
        service_changed = (travel[copies].isna().to_numpy() != record_travel.isna().to_numpy()).any(axis=1)
        print(f"calls {table['incident_id'].nunique():,} ({calls['kind'].value_counts().to_dict()}), runs {len(table):,}, stand-in runs "
              f"{int(stand_in_row.sum())}, large-fire runs {int(scenario['large_fire'].sum())}, runs placed at a residential block {int(placed.sum())}\n"
              f"checks: copied rows all from the training months: {bool((dispatches.loc[source_row[copies], 'call_received_at'] < TRAINING_MONTHS_END).all())}; "
              f"largest change of a copied run's dispatch-to-clear time: {np.nanmax(np.abs(duration_change.astype('timedelta64[s]').astype(float))) if len(kept) else 0:.0f} s; "
              f"copied runs whose engines in service differ from their source's (moved across an engine's first or last day): {int(service_changed.sum())}\n"
              f"multi-run calls with call fates: {len(scenario['fates']):,}; low-acuity runs {int(scenario['low_acuity'].sum()):,}\n")
    return scenario


def check_scenario_conditions(scenario, simulated, verbose=VERBOSE):
    """Section 13.3: what the outages and large fires of a scenario did in a simulated run. Outages: runs of real calls
    sent to an engine during its scheduled outage (from the stand-in's scheduled start to its end; must be 0), and
    stand-ins that waited for their engine to come back from a call (count, median minutes). Large fires: per fire, the engines that reached it, minutes from the call to the
    first and the last arrival, and hours on scene.

    Example: checks, large_fires = check_scenario_conditions(scenario, protocol_run)
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


def weekly_block_ratio_interval(numerator, denominator, weeks):
    """Section 13.2: two-sided interval (BOOTSTRAP_INTERVAL) of sum(numerator) / sum(denominator) from
    BLOCK_BOOTSTRAP_DRAWS weekly block-bootstrap draws (weeks resampled with replacement, seed BLOCK_BOOTSTRAP_SEED), as
    weekly_block_interval does for a mean. The three Series share one index (one row per week here).

    Example: weekly_block_ratio_interval(pd.Series([4.0, 2.0]), pd.Series([2.0, 2.0]), pd.Series([0, 1])) -> (1.0, 2.0)
             # each draw takes two weeks: 8/4 = 2, 6/4 = 1.5 or 4/4 = 1
    """
    sums, counts = numerator.groupby(weeks).sum().to_numpy(), denominator.groupby(weeks).sum().to_numpy()
    picks = np.random.default_rng(BLOCK_BOOTSTRAP_SEED).integers(0, len(sums), size=(BLOCK_BOOTSTRAP_DRAWS, len(sums)))
    low, high = np.quantile(sums[picks].sum(axis=1) / counts[picks].sum(axis=1), BOOTSTRAP_INTERVAL)
    return float(low), float(high)


def weekly_calibration_parts(dispatches_table, runs_table, first_due, census, segments, top_cells, days):
    """Section 13.2: per week (days since DAY_ZERO // 7, over `days`), the numerator and denominator of each statistic the
    nominal scenarios must reproduce, for one version (the record's M0.4 run or a scenario's protocol run): calls (other
    calls per day, study fire calls per week, runs per call, share of other calls in the record's 20 busiest cells),
    engine busy hours per day, the room (low-acuity decisions per day, the share with a choice at ACTION_WINDOW_S, the
    share where the rule would send another engine, the readiness saved and the extra travel when it would), fire calls
    whose first-due station had no free engine, minutes to the first engine at fire calls, and the time-weighted fire
    readiness loss. Stand-in runs (outages) are left out. Returns a DataFrame (rows: weeks; columns: (statistic, part)).

    Example: parts = weekly_calibration_parts(dispatches, m04_protocol, first_due_m04, census, record_segments, top_cells, range(182))
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
    """Section 13.2 (the done-criterion): each statistic of weekly_calibration_parts for the record (its value and weekly
    block-bootstrap interval) beside the nominal scenarios (the value pooled over all their weeks, and per seed), whether
    the pooled value falls inside the record's interval, and the spread of the weekly values (standard deviation) in both.
    scenario_parts: seed -> weekly_calibration_parts.

    Example: calibration = compare_scenarios_with_record(record_parts, {101: parts_101, 102: parts_102})
             calibration.loc["study fire calls per week", "inside record interval"] -> True
    """
    rows = {}
    for name in record_parts.columns.get_level_values(0).unique():
        numerator, denominator = record_parts[(name, "numerator")], record_parts[(name, "denominator")]
        low, high = weekly_block_ratio_interval(numerator, denominator, pd.Series(record_parts.index, index=record_parts.index))
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
    """Section 13.4: the room for another engine under one condition, from its protocol run over `days`: calls, low-acuity
    decisions per day, the share with a choice at ACTION_WINDOW_S, how often and how much the rule would change, fire
    calls whose first-due station had no free engine, and the time-weighted fire readiness loss.

    Example: describe_room(scenario["dispatches"], census, first_due, segments, range(182))["rule would send another engine"]
             -> 0.117  # nominal October-March, roughly
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


if __name__ == "__main__":
    # Section 1: reproduce the 2025 dispatch log
    dispatches = load_inputs()
    runs = build_runs(dispatches)
    simulated = simulate(runs)
    compare_times(runs, simulated)
    table = simulated_dispatch_table(dispatches, simulated)
    curves = response_curves(table)
    curves_2025 = response_curves(dispatches, verbose=False)        # same definition on the 2025 table
    check = compare_curves(curves, curves_2025)
    write_outputs(table, curves, check)

    # Section 2: the same calls when CAD's quickest-unit rule picks the engines
    engines = load_engines()
    travel_curve = fit_travel_curve(dispatches)
    estimated_travel = estimate_travel_seconds(dispatches, engines, travel_curve)
    protocol = simulate(runs, available=define_available_engines, choose=dispatch_by_protocol, engines=engines,
                        estimated_travel=estimated_travel)
    engine_check = compare_engines(runs, protocol, dispatches)
    protocol_table = simulated_dispatch_table(dispatches, protocol, engines)
    protocol_curves = response_curves(protocol_table)
    protocol_check = compare_curves(protocol_curves, curves_2025)
    first_on_scene = compare_first_on_scene({"2025": dispatches, "protocol": protocol_table})
    write_outputs(protocol_table, protocol_curves, protocol_check, prefix="protocol_",
                  extra_tables={"engine_check": engine_check, "first_on_scene": first_on_scene})

    # Section 3: the same rule with travel along the road network
    graph = load_road_network()
    call_nodes, engine_nodes = snap_to_network(graph, dispatches, engines)
    network_seconds, network_km = network_from_stations(graph, dispatches, call_nodes, engine_nodes)
    chosen_form, network_travel_model, travel_comparison = compare_travel_models(dispatches, engines, network_seconds,
                                                                                 network_km, travel_curve)
    network_travel = estimate_network_travel_seconds(dispatches, engines, network_seconds, network_km, network_travel_model)
    network_protocol = simulate(runs, available=define_available_engines, choose=dispatch_by_protocol, engines=engines,
                                estimated_travel=network_travel)
    network_engine_check = compare_engines(runs, network_protocol, dispatches)
    network_table = simulated_dispatch_table(dispatches, network_protocol, engines)
    network_curves = response_curves(network_table)
    network_check = compare_curves(network_curves, curves_2025)
    columns = {
        "2025 record": evaluation_column(runs, simulated, dispatches, dispatches, dispatches, check, np.nan),
        "step 2: straight line": evaluation_column(
            runs, protocol, dispatches, protocol_table, control_table(runs, protocol, dispatches, engines, estimated_travel),
            protocol_check, travel_comparison.loc["step 2: straight-line curve", "mean |error| min"]),
        "step 3: road network": evaluation_column(
            runs, network_protocol, dispatches, network_table,
            control_table(runs, network_protocol, dispatches, engines, network_travel),
            network_check, travel_comparison.loc[chosen_form, "mean |error| min"]),
    }
    evaluation = evaluation_matrix(columns)
    write_outputs(network_table, network_curves, network_check, prefix="network_",
                  extra_tables={"engine_check": network_engine_check, "travel_models": travel_comparison,
                                "evaluation_matrix": evaluation})

    # Section 4: the same rule with the GFR / ACFR line: other-agency engines only for automatic-aid call types
    aid_protocols = load_aid_protocols(dispatches)
    boundary_travel = apply_agency_boundaries(network_travel, dispatches, engines, aid_protocols)
    boundary_protocol = simulate(runs, available=define_available_engines, choose=dispatch_by_protocol, engines=engines,
                                 estimated_travel=boundary_travel)
    boundary_engine_check = compare_engines(runs, boundary_protocol, dispatches)
    boundary_table = simulated_dispatch_table(dispatches, boundary_protocol, engines)
    boundary_curves = response_curves(boundary_table)
    boundary_check = compare_curves(boundary_curves, curves_2025)
    columns["step 4: + agency boundaries"] = evaluation_column(
        runs, boundary_protocol, dispatches, boundary_table,
        control_table(runs, boundary_protocol, dispatches, engines, network_travel),   # travel itself is unchanged
        boundary_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    write_outputs(boundary_table, boundary_curves, boundary_check, prefix="boundary_",
                  extra_tables={"engine_check": boundary_engine_check, "evaluation_matrix": evaluation})

    # Section 5: where idle engines are. 5.1 (cases 2 and 3): an engine that just cleared is on its way home
    assert chosen_form == "linear on network time", "travel_while_returning needs the linear travel model"
    seconds_from_node, node_position = make_seconds_from_node(graph)
    home_nodes = engine_nodes.set_index("engine_id")["node"].astype("int64")
    returning_protocol = simulate(
        runs, available=define_available_engines, choose=dispatch_by_protocol, engines=engines, estimated_travel=boundary_travel,
        where_engines_are=lambda row, now, state, travel: travel_while_returning(
            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model))
    returning_engine_check = compare_engines(runs, returning_protocol, dispatches)
    returning_table = simulated_dispatch_table(dispatches, returning_protocol, engines)
    returning_curves = response_curves(returning_table)
    returning_check = compare_curves(returning_curves, curves_2025)
    columns["step 5.1: + returning engines"] = evaluation_column(
        runs, returning_protocol, dispatches, returning_table,
        control_table(runs, returning_protocol, dispatches, engines, network_travel),
        returning_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    write_outputs(returning_table, returning_curves, returning_check, prefix="returning_",
                  extra_tables={"engine_check": returning_engine_check, "evaluation_matrix": evaluation})

    # 5.1b: the same, with dwell and slowdown of the drive home fitted to AVL positions
    fitted_dwell_s, fitted_slowdown, progress_comparison = fit_return_progress(
        dispatches, engines, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model)
    avl_protocol = simulate(
        runs, available=define_available_engines, choose=dispatch_by_protocol, engines=engines, estimated_travel=boundary_travel,
        where_engines_are=lambda row, now, state, travel: travel_while_returning(
            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
            fitted_dwell_s, fitted_slowdown))
    avl_engine_check = compare_engines(runs, avl_protocol, dispatches)
    avl_table = simulated_dispatch_table(dispatches, avl_protocol, engines)
    avl_curves = response_curves(avl_table)
    avl_check = compare_curves(avl_curves, curves_2025)
    columns["step 5.1b: + AVL-fitted return"] = evaluation_column(
        runs, avl_protocol, dispatches, avl_table, control_table(runs, avl_protocol, dispatches, engines, network_travel),
        avl_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    write_outputs(avl_table, avl_curves, avl_check, prefix="returning_avl_",
                  extra_tables={"engine_check": avl_engine_check, "return_progress": progress_comparison,
                                "evaluation_matrix": evaluation})

    # Section 6 (M0): the 5.1b run with modeled travel for every run, waiting instead of taking busy engines; E0 checks it
    print(f"step 5.1b: calls whose first arrival is the recorded second: {share_first_arrivals_as_recorded(runs, avl_protocol):.1%}\n")
    m0_runs = clean_runs_for_m0(runs)
    wait_rank = dispatches["priority"].astype("string").map(PRIORITY_RANK).fillna(len(PRIORITY_RANK))
    m0_report = {}
    m0_protocol = simulate(
        m0_runs, available=define_available_engines, choose=dispatch_closest, engines=engines, estimated_travel=boundary_travel,
        where_engines_are=lambda row, now, state, travel: travel_while_returning(
            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
            fitted_dwell_s, fitted_slowdown),
        wait_rank=wait_rank, report=m0_report, counterfactual=True)
    m0_engine_check = compare_engines(runs, m0_protocol, dispatches)
    m0_table = simulated_dispatch_table(dispatches, m0_protocol, engines)
    m0_curves = response_curves(m0_table)
    m0_check = compare_curves(m0_curves, curves_2025)
    e0_states = check_counterfactual_run(runs, m0_protocol, dispatches, m0_report)
    e0_strata = compare_curves_by_stratum(m0_table, dispatches)
    columns["step 6 (M0): modeled travel for every run"] = evaluation_column(   # control = 2025 record: P90 M0 minus 2025
        runs, m0_protocol, dispatches, m0_table, dispatches, m0_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    decide_e0(e0_states, e0_strata)
    write_outputs(m0_table, m0_curves, m0_check, prefix="m0_",
                  extra_tables={"engine_check": m0_engine_check, "e0_states": e0_states.to_frame("value"),
                                "e0_strata": e0_strata, "evaluation_matrix": evaluation})

    # Section 7 (M0.1): M0, but the chosen engine takes modeled travel x a ratio drawn per run from real spread; E0 again
    spread_quantile = pd.Series(np.random.default_rng(TRAVEL_SPREAD_SEED).random(len(runs)), index=runs.index)
    travel_spread, spread_table, spread_check = fit_travel_spread(dispatches, network_seconds, network_km,
                                                                  network_travel_model, spread_quantile)
    call_inside_city = dispatches["focal"].fillna(False).astype(bool)
    m01_report = {}
    m01_protocol = simulate(
        m0_runs, available=define_available_engines, engines=engines, estimated_travel=boundary_travel,
        choose=lambda run, available, travel: dispatch_closest_with_spread(run, available, travel, travel_spread,
                                                                           call_inside_city, spread_quantile),
        where_engines_are=lambda row, now, state, travel: travel_while_returning(
            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
            fitted_dwell_s, fitted_slowdown),
        wait_rank=wait_rank, report=m01_report, counterfactual=True)
    m01_origins = describe_engine_origins(m01_report["dispatches"], dispatches, boundary_travel, call_nodes, home_nodes,
                                          seconds_from_node, node_position, network_travel_model, fitted_dwell_s, fitted_slowdown)
    describe_travel_draws(m01_origins, travel_spread, call_inside_city)
    m01_engine_check = compare_engines(runs, m01_protocol, dispatches)
    m01_table = simulated_dispatch_table(dispatches, m01_protocol, engines)
    m01_curves = response_curves(m01_table)
    m01_check = compare_curves(m01_curves, curves_2025)
    m01_states = check_counterfactual_run(runs, m01_protocol, dispatches, m01_report)
    m01_strata = compare_curves_by_stratum(m01_table, dispatches)
    columns["step 7 (M0.1): + travel spread"] = evaluation_column(   # control = 2025 record: P90 M0.1 minus 2025
        runs, m01_protocol, dispatches, m01_table, dispatches, m01_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    decide_e0(m01_states, m01_strata)
    write_outputs(m01_table, m01_curves, m01_check, prefix="m01_",
                  extra_tables={"travel_spread": spread_table, "travel_spread_check": spread_check,
                                "engine_check": m01_engine_check, "e0_states": m01_states.to_frame("value"),
                                "e0_strata": m01_strata, "evaluation_matrix": evaluation})

    # Section 8 (M0.2): M0.1, but in calls with two or more runs the call decides which engines arrive; E0 again
    fates_by_arrival = call_fates(m0_runs)
    m02_report = {}
    m02_protocol = simulate(
        m0_runs, available=define_available_engines, engines=engines, estimated_travel=boundary_travel,
        choose=lambda run, available, travel: dispatch_closest_with_spread(run, available, travel, travel_spread,
                                                                           call_inside_city, spread_quantile),
        where_engines_are=lambda row, now, state, travel: travel_while_returning(
            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
            fitted_dwell_s, fitted_slowdown),
        wait_rank=wait_rank, report=m02_report, counterfactual=True, fates_by_arrival=fates_by_arrival)
    m02_fates = describe_call_fates(runs, m02_protocol, m02_report["dispatches"], fates_by_arrival)
    m02_origins = describe_engine_origins(m02_report["dispatches"], dispatches, boundary_travel, call_nodes, home_nodes,
                                          seconds_from_node, node_position, network_travel_model, fitted_dwell_s, fitted_slowdown)
    describe_travel_draws(m02_origins, travel_spread, call_inside_city)
    m02_engine_check = compare_engines(runs, m02_protocol, dispatches)
    m02_table = simulated_dispatch_table(dispatches, m02_protocol, engines)
    m02_curves = response_curves(m02_table)
    m02_check = compare_curves(m02_curves, curves_2025)
    m02_states = check_counterfactual_run(runs, m02_protocol, dispatches, m02_report, arrivals_by_call=True)
    m02_strata = compare_curves_by_stratum(m02_table, dispatches)
    columns["step 8 (M0.2): + arrivals decided by the call"] = evaluation_column(   # control = 2025 record
        runs, m02_protocol, dispatches, m02_table, dispatches, m02_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    decide_e0(m02_states, m02_strata)
    write_outputs(m02_table, m02_curves, m02_check, prefix="m02_",
                  extra_tables={"call_fates": m02_fates, "engine_check": m02_engine_check,
                                "e0_states": m02_states.to_frame("value"), "e0_strata": m02_strata,
                                "evaluation_matrix": evaluation})

    # Section 9 (M0.3): M0.2, but a quint (it carries an aerial ladder) is sent only if it saves more than a fitted penalty
    quint_penalty_s, penalty_fit = fit_quint_penalty(dispatches, m0_runs, engines, boundary_travel)
    quint_rank_penalty = (engines.set_index("engine_id")["engine_type"] == "Q").astype(float) * quint_penalty_s
    m03_report = {}
    m03_protocol = simulate(
        m0_runs, available=define_available_engines, engines=engines, estimated_travel=boundary_travel,
        choose=lambda run, available, travel: dispatch_closest_with_spread(run, available, travel, travel_spread,
                                                                           call_inside_city, spread_quantile, quint_rank_penalty),
        where_engines_are=lambda row, now, state, travel: travel_while_returning(
            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
            fitted_dwell_s, fitted_slowdown),
        wait_rank=wait_rank, report=m03_report, counterfactual=True, fates_by_arrival=fates_by_arrival)
    describe_type_choices(dispatches, {"M0.2": m02_protocol, "M0.3": m03_protocol})
    m03_engine_check = compare_engines(runs, m03_protocol, dispatches)
    m03_table = simulated_dispatch_table(dispatches, m03_protocol, engines)
    m03_curves = response_curves(m03_table)
    m03_check = compare_curves(m03_curves, curves_2025)
    m03_states = check_counterfactual_run(runs, m03_protocol, dispatches, m03_report, arrivals_by_call=True)
    m03_strata = compare_curves_by_stratum(m03_table, dispatches)
    columns["step 9 (M0.3): + quint penalty"] = evaluation_column(   # control = 2025 record
        runs, m03_protocol, dispatches, m03_table, dispatches, m03_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    decide_e0(m03_states, m03_strata)
    write_outputs(m03_table, m03_curves, m03_check, prefix="m03_",
                  extra_tables={"quint_penalty_fit": penalty_fit, "engine_check": m03_engine_check,
                                "e0_states": m03_states.to_frame("value"), "e0_strata": m03_strata,
                                "evaluation_matrix": evaluation})

    # Section 10 (M0.4): M0.3, but the quint penalty depends on the call type and brush fires draw their own travel spread
    call_type_for_quint = dispatches["protocol"].map(QUINT_RULE_CALL_TYPES).fillna("all other calls")   # run row -> label
    quint_penalty_by_type, type_penalty_fit = fit_quint_penalty_by_call_type(dispatches, m0_runs, engines, boundary_travel,
                                                                             call_type_for_quint, quint_penalty_s)
    quint_penalty_of_run = call_type_for_quint.map(quint_penalty_by_type)                 # run row -> seconds added to a quint
    quint_indicator = (engines.set_index("engine_id")["engine_type"] == "Q").astype(float)  # engine id -> 1 for a quint
    brush_run = dispatches["protocol"].eq(BRUSH_FIRE_PROTOCOL).fillna(False)
    brush_spread, brush_spread_table, brush_spread_check = fit_travel_spread(
        dispatches.loc[brush_run], network_seconds.loc[brush_run], network_km.loc[brush_run], network_travel_model,
        spread_quantile, bin_count=BRUSH_SPREAD_BINS)
    m04_report = {}
    m04_protocol = simulate(
        m0_runs, available=define_available_engines, engines=engines, estimated_travel=boundary_travel,
        choose=lambda run, available, travel: dispatch_closest_with_spread(
            run, available, travel, brush_spread if brush_run[run.name] else travel_spread, call_inside_city,
            spread_quantile, quint_indicator * quint_penalty_of_run[run.name]),
        where_engines_are=lambda row, now, state, travel: travel_while_returning(
            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
            fitted_dwell_s, fitted_slowdown),
        wait_rank=wait_rank, report=m04_report, counterfactual=True, fates_by_arrival=fates_by_arrival)
    m04_origins = describe_engine_origins(m04_report["dispatches"], dispatches, boundary_travel, call_nodes, home_nodes,
                                          seconds_from_node, node_position, network_travel_model, fitted_dwell_s, fitted_slowdown)
    print("brush-fire runs only:")
    describe_travel_draws(m04_origins[m04_origins["run"].map(brush_run).astype(bool).to_numpy()], brush_spread, call_inside_city)
    describe_type_choices(dispatches, {"M0.3": m03_protocol, "M0.4": m04_protocol},
                          groups=[("all", call_type_for_quint.notna()),
                                  *[(label, call_type_for_quint == label) for label in quint_penalty_by_type]])
    m04_engine_check = compare_engines(runs, m04_protocol, dispatches)
    m04_table = simulated_dispatch_table(dispatches, m04_protocol, engines)
    m04_curves = response_curves(m04_table)
    m04_check = compare_curves(m04_curves, curves_2025)
    m04_states = check_counterfactual_run(runs, m04_protocol, dispatches, m04_report, arrivals_by_call=True)
    m04_strata = compare_curves_by_stratum(m04_table, dispatches)
    fire_check = compare_fire_arrivals(dispatches, {"M0.3": m03_table, "M0.4": m04_table})
    brush_quint_calls = compare_quint_calls(dispatches, {"M0.3": m03_table, "M0.4": m04_table})
    columns["step 10 (M0.4): + quint penalty by call type, brush travel"] = evaluation_column(   # control = 2025 record
        runs, m04_protocol, dispatches, m04_table, dispatches, m04_check, travel_comparison.loc[chosen_form, "mean |error| min"])
    evaluation = evaluation_matrix(columns)
    decide_e0(m04_states, m04_strata)
    write_outputs(m04_table, m04_curves, m04_check, prefix="m04_",
                  extra_tables={"quint_penalty_by_call_type": type_penalty_fit, "brush_travel_spread": brush_spread_table,
                                "brush_travel_spread_check": brush_spread_check, "fire_check": fire_check,
                                "brush_quint_calls": brush_quint_calls,
                                "engine_check": m04_engine_check, "e0_states": m04_states.to_frame("value"),
                                "e0_strata": m04_strata, "evaluation_matrix": evaluation})

    # Section 11 (E1, research.md RQ1): how well the protocol protects fire calls, on the frozen M0.4 run beside the record
    first_due_record = check_first_due(dispatches, m0_runs, boundary_travel, engines)
    first_due_m04 = check_first_due(dispatches, m04_protocol, boundary_travel, engines)
    first_due_summary, first_due_by_station = summarize_first_due({"2025 record": first_due_record, "M0.4 protocol run": first_due_m04})
    readiness_cells, travel_to_cells, fire_hour_share = build_readiness_cells(dispatches, graph, engines, home_nodes, aid_protocols,
                                                                              seconds_from_node, node_position, network_travel_model)
    readiness = {name: fire_readiness(runs_table, engines, readiness_cells, travel_to_cells, fire_hour_share)
                 for name, runs_table in [("2025 record", m0_runs), ("M0.4 protocol run", m04_protocol)]}
    e1_tables = {
        "first_due_calls": pd.concat({"2025 record": first_due_record, "M0.4 protocol run": first_due_m04}, names=["version"]),
        "first_due_summary": first_due_summary, "first_due_by_station": first_due_by_station,
        "readiness_summary": pd.concat({name: result[2] for name, result in readiness.items()}, axis=1),
        "readiness_cells": pd.concat({name: result[1] for name, result in readiness.items()}, names=["version"])}
    for name, table in e1_tables.items():
        table.to_csv(OUTPUT_DIR / f"e1_{name}.csv")
    pd.concat({name: result[0] for name, result in readiness.items()}, names=["version"]).to_parquet(OUTPUT_DIR / "e1_readiness_segments.parquet")
    print(f"wrote {', '.join(f'e1_{name}.csv' for name in e1_tables)}, e1_readiness_segments.parquet to {OUTPUT_DIR}\n")

    # Section 12 (research.md 3.4): the opportunity-cost rule on low-acuity calls: its room, its year against the protocol, branches
    low_acuity = ((dispatches["call_type"].eq("Medical") & dispatches["priority"].eq("Low")).fillna(False)
                  | dispatches["protocol"].isin(LOW_ACUITY_SERVICE_PROTOCOLS).fillna(False))
    print(f"low-acuity runs (SOC low-priority medical and service calls): {int(low_acuity.sum()):,} of {len(dispatches):,}\n")
    readiness_of, opportunity_cost = make_readiness_tools(readiness_cells, travel_to_cells, engines)
    census = census_decisions(dispatches, m04_protocol, boundary_travel, engines, quint_penalty_of_run, quint_indicator, low_acuity,
                              opportunity_cost, ACTION_WINDOWS_S)

    def run_year(quantile, action_window_s=None, report=None):
        """The full-year M0.4 setting with travel quantiles `quantile`; with action_window_s, the opportunity-cost rule on
        low-acuity calls instead of the protocol."""
        def choose(run, available, travel):
            spread, penalty = (brush_spread if brush_run[run.name] else travel_spread), quint_indicator * quint_penalty_of_run[run.name]
            if action_window_s is None:
                return dispatch_closest_with_spread(run, available, travel, spread, call_inside_city, quantile, penalty)
            return dispatch_by_opportunity_cost(run, available, travel, spread, call_inside_city, quantile, penalty, low_acuity,
                                                action_window_s, opportunity_cost)
        return simulate(m0_runs, available=define_available_engines, engines=engines, estimated_travel=boundary_travel, choose=choose,
                        where_engines_are=lambda row, now, state, travel: travel_while_returning(
                            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
                            fitted_dwell_s, fitted_slowdown),
                        wait_rank=wait_rank, report=report, counterfactual=True, fates_by_arrival=fates_by_arrival, verbose=False)

    comparisons = {}
    for seed in POLICY_SEEDS:
        quantile = spread_quantile if seed == TRAVEL_SPREAD_SEED else pd.Series(np.random.default_rng(seed).random(len(runs)), index=runs.index)
        protocol_run = m04_protocol if seed == TRAVEL_SPREAD_SEED else run_year(quantile)
        protocol_table = simulated_dispatch_table(dispatches, protocol_run, engines, verbose=False)
        first_due_protocol = check_first_due(dispatches, protocol_run, boundary_travel, engines, verbose=False)
        readiness_protocol = fire_readiness(protocol_run, engines, readiness_cells, travel_to_cells, fire_hour_share, verbose=False)[0]
        for window_s in (ACTION_WINDOWS_S if seed == TRAVEL_SPREAD_SEED else [ACTION_WINDOW_S]):
            rule_report = {}
            rule_run = run_year(quantile, window_s, rule_report)
            if seed == TRAVEL_SPREAD_SEED and window_s == ACTION_WINDOW_S:
                print("state checks of the opportunity-cost rule's year (seed of M0.4, 60 s window):")
                rule_states = check_counterfactual_run(runs, rule_run, dispatches, rule_report, arrivals_by_call=True)
            readiness_rule = fire_readiness(rule_run, engines, readiness_cells, travel_to_cells, fire_hour_share, verbose=False)[0]
            print(f"seed {seed}, action window {window_s} s:")
            comparisons[(seed, window_s)] = compare_policies(
                dispatches, protocol_table, simulated_dispatch_table(dispatches, rule_run, engines, verbose=False), first_due_protocol,
                readiness_protocol, readiness_rule, fire_hour_share, low_acuity)
    policy_comparison = pd.concat(comparisons, names=["seed", "action_window_s", "outcome"])
    main_window = policy_comparison.xs(ACTION_WINDOW_S, level="action_window_s")
    over_seeds = pd.DataFrame({
        "mean difference s": main_window["difference s"].astype(float).groupby(level="outcome", sort=False).mean(),
        "lowest s": main_window["difference s"].astype(float).groupby(level="outcome", sort=False).min(),
        "highest s": main_window["difference s"].astype(float).groupby(level="outcome", sort=False).max(),
        "classes over seeds": main_window["class"].fillna("").groupby(level="outcome", sort=False).agg(lambda c: dict(c.value_counts()))})
    print(f"--- opportunity-cost rule ({ACTION_WINDOW_S} s window) minus protocol over {len(POLICY_SEEDS)} seeds\n{over_seeds.round(2).to_string()}\n")

    def simulate_window(window_runs, choose):
        """A stretch of calls under the M0.4 setting (Section 12.4 branches), with its own call fates."""
        return simulate(window_runs, available=define_available_engines, engines=engines, estimated_travel=boundary_travel, choose=choose,
                        where_engines_are=lambda row, now, state, travel: travel_while_returning(
                            row, now, state, travel, call_nodes, home_nodes, seconds_from_node, node_position, network_travel_model,
                            fitted_dwell_s, fitted_slowdown),
                        wait_rank=wait_rank, counterfactual=True, fates_by_arrival=call_fates(window_runs), verbose=False)

    branches, branch_cells = branch_decisions(
        dispatches, m0_runs, m04_protocol, census, simulate_window,
        lambda run, available, travel: dispatch_closest_with_spread(run, available, travel, brush_spread if brush_run[run.name] else travel_spread,
                                                                    call_inside_city, spread_quantile, quint_indicator * quint_penalty_of_run[run.name]),
        lambda run, available, travel: choose_by_opportunity_cost(run, available, travel, quint_indicator * quint_penalty_of_run[run.name],
                                                                  low_acuity, ACTION_WINDOW_S, opportunity_cost),
        readiness_of, readiness_cells, travel_to_cells, engines)
    oc_tables = {"census": census, "policy_comparison": policy_comparison, "over_seeds": over_seeds,
                 "rule_states": rule_states.to_frame("value"), "branches": branches, "branch_cells": branch_cells}
    for name, table in oc_tables.items():
        table.to_csv(OUTPUT_DIR / f"oc_{name}.csv")
    print(f"wrote {', '.join(f'oc_{name}.csv' for name in oc_tables)} to {OUTPUT_DIR}\n")

    # Section 13 (research.md 3.3): the scenario generator: nominal scenarios checked against October-March, then the conditions
    call_pool = build_call_pool(dispatches)
    resident_locations, resident_location_seconds = build_resident_locations(call_pool, graph, home_nodes, seconds_from_node, node_position, engines)
    call_counts, call_dispersion = fit_call_rates(call_pool)
    training_days = range((TRAINING_MONTHS_END - DAY_ZERO).days)
    busiest_cells = call_pool[~call_pool["fire"]].groupby(["cell_x", "cell_y"]).size().nlargest(20).index
    no_outages = pd.DataFrame(columns=["engine_id", "start_s", "end_s"])

    def make_scenario(scenario_calls, scenario_outages, scenario_seed, verbose=False):
        """A scenario's simulate inputs from its calls and outages (build_scenario with this run's fitted inputs)."""
        return build_scenario(scenario_calls, scenario_outages, call_pool, resident_locations, resident_location_seconds, dispatches, m0_runs,
                              engines, network_seconds, network_km, call_nodes, network_travel_model, aid_protocols, quint_penalty_by_type,
                              scenario_seed, verbose=verbose)

    def simulate_scenario(scenario, action_window_s=None, report=None):
        """A scenario under the M0.4 protocol; with action_window_s, the opportunity-cost rule on its low-acuity calls instead."""
        def choose(run, available, travel):
            spread, penalty = (brush_spread if scenario["brush"][run.name] else travel_spread), quint_indicator * scenario["quint_penalty"][run.name]
            if action_window_s is None:
                return dispatch_closest_with_spread(run, available, travel, spread, scenario["inside_city"], scenario["spread_quantile"], penalty)
            return dispatch_by_opportunity_cost(run, available, travel, spread, scenario["inside_city"], scenario["spread_quantile"], penalty,
                                                scenario["low_acuity"], action_window_s, opportunity_cost)
        return simulate(scenario["runs"], available=define_available_engines, engines=engines, estimated_travel=scenario["travel"], choose=choose,
                        where_engines_are=lambda row, now, state, travel: travel_while_returning(
                            row, now, state, travel, scenario["call_nodes"], home_nodes, seconds_from_node, node_position, network_travel_model,
                            fitted_dwell_s, fitted_slowdown),
                        wait_rank=scenario["wait_rank"], report=report, counterfactual=True, fates_by_arrival=scenario["fates"], verbose=False)

    def measure_scenario_run(scenario, scenario_run):
        """The first-due check, the census at ACTION_WINDOW_S and fire readiness (with waits) of a scenario's run."""
        return (check_first_due(scenario["dispatches"], scenario_run, scenario["travel"], engines, verbose=False),
                census_decisions(scenario["dispatches"], scenario_run, scenario["travel"], engines, scenario["quint_penalty"], quint_indicator,
                                 scenario["low_acuity"], opportunity_cost, [ACTION_WINDOW_S], verbose=False),
                fire_readiness(scenario_run, engines, readiness_cells, travel_to_cells, fire_hour_share, wait_for_busy=True, verbose=False)[0])

    # 13.1-13.2: nominal scenarios (whole October-March each) under the protocol, against the record
    record_segments = fire_readiness(m04_protocol, engines, readiness_cells, travel_to_cells, fire_hour_share, wait_for_busy=True, verbose=False)[0]
    record_parts = weekly_calibration_parts(dispatches, m04_protocol, first_due_m04, census, record_segments, busiest_cells, training_days)
    nominal_scenarios, scenario_parts, scenario_checks = {}, {}, {}
    for scenario_seed in SCENARIO_SEEDS:
        first_seed = scenario_seed == SCENARIO_SEEDS[0]
        scenario_calls, scenario_day_factors = generate_calls(call_pool, call_counts, call_dispersion, training_days, scenario_seed, verbose=first_seed)
        nominal_scenario = make_scenario(scenario_calls, no_outages, scenario_seed, verbose=first_seed)
        nominal_report = {}
        nominal_run = simulate_scenario(nominal_scenario, report=nominal_report)
        print(f"nominal scenario, seed {scenario_seed}: protocol run simulated")
        scenario_checks[(f"nominal, seed {scenario_seed}", "protocol")] = check_counterfactual_run(
            nominal_scenario["runs"], nominal_run, nominal_scenario["dispatches"], nominal_report, arrivals_by_call=True, verbose=first_seed)
        nominal_first_due, nominal_census, nominal_segments = measure_scenario_run(nominal_scenario, nominal_run)
        scenario_parts[scenario_seed] = weekly_calibration_parts(nominal_scenario["dispatches"], nominal_run, nominal_first_due, nominal_census,
                                                                 nominal_segments, busiest_cells, training_days)
        nominal_scenarios[scenario_seed] = (scenario_calls, scenario_day_factors, nominal_scenario, nominal_run, nominal_first_due, nominal_census,
                                            nominal_segments)
    calibration = compare_scenarios_with_record(record_parts, scenario_parts)

    # 13.3-13.4: each condition on the first seed's calls: what it does, its room, and the opportunity-cost rule against the protocol
    base_seed = SCENARIO_SEEDS[0]
    base_calls, base_day_factors = nominal_scenarios[base_seed][:2]
    surge_sets = {count: add_short_surges(base_calls, call_pool, resident_locations, training_days, count, base_seed, verbose=count == SURGE_EXTRA_CALLS[0])
                  for count in SURGE_EXTRA_CALLS}
    surge_condition_counts = {f"short surges, {count} extra calls": count for count in SURGE_EXTRA_CALLS}
    conditions = {"nominal": (base_calls, no_outages),
                  **{f"other calls x{factor}": (add_demand_growth(base_calls, base_day_factors, call_pool, call_counts, resident_locations, factor, base_seed),
                                                no_outages) for factor in DEMAND_FACTORS},
                  **{name: (surge_sets[count][0], no_outages) for name, count in surge_condition_counts.items()},
                  **{f"{count} engines out each day": (base_calls, add_outages(engines, training_days, count, base_seed)) for count in OUTAGE_ENGINE_COUNTS},
                  **{f"station {station} closed": (base_calls, close_station(engines, station, training_days)) for station in HOTSPOT_STATIONS},
                  "a large fire each week": (add_large_fires(base_calls, call_pool, training_days, base_seed), no_outages)}
    training_end_s = len(training_days) * 86400
    condition_rooms, condition_comparisons, surge_segments = {}, {}, {}
    for condition_name, (condition_calls, condition_outages) in conditions.items():
        print(f"=== condition: {condition_name}")
        if condition_name == "nominal":
            condition_scenario, condition_protocol_run, condition_first_due, condition_census, condition_segments = nominal_scenarios[base_seed][2:]
        else:
            condition_scenario = make_scenario(condition_calls, condition_outages, base_seed,
                                               verbose=condition_name in (f"{OUTAGE_ENGINE_COUNTS[0]} engines out each day", "a large fire each week"))
            condition_report = {}
            condition_protocol_run = simulate_scenario(condition_scenario, report=condition_report)
            scenario_checks[(condition_name, "protocol")] = check_counterfactual_run(
                condition_scenario["runs"], condition_protocol_run, condition_scenario["dispatches"], condition_report, arrivals_by_call=True, verbose=False)
            condition_first_due, condition_census, condition_segments = measure_scenario_run(condition_scenario, condition_protocol_run)
        condition_effects, _ = check_scenario_conditions(condition_scenario, condition_protocol_run,
                                                         verbose=len(condition_outages) > 0 or bool(condition_scenario["large_fire"].any()))
        scenario_checks[(condition_name, "protocol")] = pd.concat([scenario_checks[(condition_name, "protocol")] if condition_name != "nominal"
                                                                   else scenario_checks[(f"nominal, seed {base_seed}", "protocol")], condition_effects])
        condition_rule_report = {}
        condition_rule_run = simulate_scenario(condition_scenario, ACTION_WINDOW_S, condition_rule_report)
        scenario_checks[(condition_name, "opportunity-cost rule")] = check_counterfactual_run(
            condition_scenario["runs"], condition_rule_run, condition_scenario["dispatches"], condition_rule_report, arrivals_by_call=True, verbose=False)
        condition_rule_segments = fire_readiness(condition_rule_run, engines, readiness_cells, travel_to_cells, fire_hour_share,
                                                 wait_for_busy=True, verbose=False)[0]
        if condition_name in surge_condition_counts:
            surge_segments[surge_condition_counts[condition_name]] = (condition_segments, condition_rule_segments)
        condition_rooms[condition_name] = describe_room(condition_scenario["dispatches"], condition_census, condition_first_due, condition_segments,
                                                        training_days)
        condition_comparisons[condition_name] = compare_policies(
            condition_scenario["dispatches"], simulated_dispatch_table(condition_scenario["dispatches"], condition_protocol_run, engines, verbose=False),
            simulated_dispatch_table(condition_scenario["dispatches"], condition_rule_run, engines, verbose=False), condition_first_due,
            condition_segments[condition_segments["start_s"] < training_end_s], condition_rule_segments[condition_rule_segments["start_s"] < training_end_s],
            fire_hour_share, condition_scenario["low_acuity"], verbose=False).drop(index="fire calls, Jul-Sep")

    # 13.5: short surges in every first-due area (a stress sweep): what a surge costs in fire readiness while it lasts,
    # against the same hours without it (four seeds), and what the opportunity-cost rule changes then (first seed)
    surge_tables = []
    for count in SURGE_EXTRA_CALLS:
        for scenario_seed in SCENARIO_SEEDS:
            if scenario_seed == base_seed:
                seed_surges = surge_sets[count][1]
                seed_surge_segments, seed_rule_segments = surge_segments[count]
            else:
                seed_surged_calls, seed_surges = add_short_surges(nominal_scenarios[scenario_seed][0], call_pool, resident_locations, training_days,
                                                                  count, scenario_seed, verbose=False)
                seed_scenario = make_scenario(seed_surged_calls, no_outages, scenario_seed)
                seed_report = {}
                seed_run = simulate_scenario(seed_scenario, report=seed_report)
                scenario_checks[(f"short surges, {count} extra calls, seed {scenario_seed}", "protocol")] = check_counterfactual_run(
                    seed_scenario["runs"], seed_run, seed_scenario["dispatches"], seed_report, arrivals_by_call=True, verbose=False)
                seed_surge_segments = fire_readiness(seed_run, engines, readiness_cells, travel_to_cells, fire_hour_share, wait_for_busy=True,
                                                     verbose=False)[0]
                seed_rule_segments = None
            loss_with_surge = readiness_loss_in_windows(seed_surge_segments, seed_surges["start_s"], seed_surges["end_s"])
            surge_tables.append(seed_surges.assign(
                seed=scenario_seed,
                readiness_loss_without_surge_s=readiness_loss_in_windows(nominal_scenarios[scenario_seed][6], seed_surges["start_s"], seed_surges["end_s"]),
                readiness_loss_with_surge_s=loss_with_surge,
                rule_minus_protocol_s=np.nan if seed_rule_segments is None else
                readiness_loss_in_windows(seed_rule_segments, seed_surges["start_s"], seed_surges["end_s"]) - loss_with_surge))
            print(f"short surges, {count} extra calls, seed {scenario_seed}: readiness during the surges measured")
    scenario_surges = pd.concat(surge_tables, ignore_index=True)
    scenario_surges["readiness_cost_s"] = scenario_surges["readiness_loss_with_surge_s"] - scenario_surges["readiness_loss_without_surge_s"]
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
    middle_count = SURGE_EXTRA_CALLS[len(SURGE_EXTRA_CALLS) // 2]
    print(f"--- short surges ({SURGE_HOURS} h within {SURGE_RADIUS_M} m, one a day, visiting every first-due area in turn): the fire readiness "
          f"loss while a surge lasts, against the same hours without it (four seeds), and the rule minus the protocol then (seed {base_seed})\n"
          f"{scenario_surge_areas.xs('all areas', level='first-due area').round(2).to_string()}\n"
          f"areas where a surge of {middle_count} extra calls costs most:\n"
          f"{scenario_surge_areas.xs(middle_count, level='extra calls').drop(index='all areas').sort_values('readiness cost (s)', ascending=False).head(5).round(2).to_string()}\n")
    room = pd.DataFrame(condition_rooms)
    scenario_policy_comparison = pd.concat(condition_comparisons, names=["condition", "outcome"])
    rule_summary = pd.DataFrame({condition_name: {
        label: f"{comparison.at[outcome, 'difference s']:+.2f} ({comparison.at[outcome, 'class']})" if comparison.at[outcome, "class"] else
               f"{comparison.at[outcome, 'difference s']:+.2f}"
        for label, outcome in [("fire calls", "fire calls, whole year"), ("first-due committed", "fire calls, first-due station committed (protocol)"),
                               ("readiness R", "fire readiness R, weekly hour-weighted mean"), ("high-acuity P90", "high-acuity medical, P90 first arrival"),
                               ("low-acuity cost", "low-acuity calls, mean first arrival (cost)")]}
        for condition_name, comparison in condition_comparisons.items()}).T
    violations = pd.DataFrame(scenario_checks).loc[["runs never served", "runs whose engine was taken by a newer run",
                                                    "runs sent while their engine was still on an earlier run",
                                                    "calls whose number of arrivals differs from 2025", "runs with times out of order",
                                                    "runs served without a clear"]].sum()
    print(f"--- the room for another engine under each condition (seed {base_seed}, protocol runs, October-March)\n{room.round(3).to_string()}\n"
          f"--- the opportunity-cost rule ({ACTION_WINDOW_S} s) minus the protocol under each condition (seconds; class against m = "
          f"{FIRE_MARGIN_S} s, delta = {MEDICAL_MARGIN_S} s)\n{rule_summary.to_string()}\n"
          f"state violations over all scenario runs: {int(violations.sum())}; real runs sent to an engine out of service: "
          f"{int(pd.DataFrame(scenario_checks).loc['real runs sent to an engine out of service'].sum())}\n")
    scenario_tables = {"calibration": calibration, "room": room, "policy_comparison": scenario_policy_comparison,
                       "checks": pd.DataFrame(scenario_checks), "surges": scenario_surges, "surge_areas": scenario_surge_areas}
    for name, table in scenario_tables.items():
        table.to_csv(OUTPUT_DIR / f"scenario_{name}.csv")
    print(f"wrote {', '.join(f'scenario_{name}.csv' for name in scenario_tables)} to {OUTPUT_DIR}\n")
