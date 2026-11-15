"""The dispatch environment: Gainesville Fire Rescue and Alachua County Fire Rescue engines answering calls.

This module simulates how engines answer calls. It uses SimPy to replay the calls of fiscal year 2025 (October 2024
to September 2025) or the calls of a synthetic scenario. When a run of a call needs an engine, a dispatch policy
chooses one of the engines that are in service and free. The environment then moves that engine with a realistic
travel time and keeps track of where every engine is, including engines still driving home from their last call.

Three objects connect the environment to the rest of the study:
- The world is a dict that build_world fills once. It holds the record, the engines, the road travel times and every
  fitted input: the drive home, the travel spread (how real travel spreads around modeled travel), the quint penalty
  and the fire-call map used for fire readiness.
- A scenario is a dict that holds the calls to simulate. world["record"] holds the 2025 record as a scenario;
  synthetic scenarios have the same keys.
- A policy is a function policy(world, scenario, run, available_engines, travel_s, context) that returns the engine
  to send, or None to let the run wait; context holds what is known at the decision. simulate(world, scenario, policy)
  runs a scenario under a policy.
"""
import functools
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import scipy.optimize
import scipy.sparse.csgraph
import simpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]                                     # [DESIGN] this file lives in code/dispatch_simulator/
DISPATCHES_PARQUET = PROJECT_ROOT / "data" / "derived" / "engine_dispatches_fy2025" / "engine_dispatches.parquet"  # [SOURCE] one row per engine run, FY2025
ENGINES_CSV = PROJECT_ROOT / "data" / "derived" / "engine_dispatches_fy2025" / "engines.csv"                         # [SOURCE] engines, home stations, positions
ROAD_NETWORK_GRAPHML = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "alachua_drive.graphml"  # [SOURCE] OSM drive network with free-flow travel times (osmnx)
AUTOMATIC_AID_FY2024 = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "acfr_gfr_automatic_aid_fy2024.parquet"  # [SOURCE] GFR-ACFR automatic aid dispatches, FY2024
TIME_COLUMNS = ["dispatched_at", "enroute_at", "arrived_at", "cleared_at"]  # [SOURCE] the four CAD times of one engine run, in order
DAY_ZERO = pd.Timestamp("2024-10-01")                                       # [SOURCE] fiscal year 2025 starts; the simulation clock counts seconds from here
YEAR_END = pd.Timestamp("2025-10-01")                                       # [SOURCE] fiscal year 2025 ends (exclusive)
SERVICE_EDGE_DAYS = 14                                                      # [DESIGN] an engine first sent this close to the year's start is in service from the start; one last sent this close to its end is in service to the end
AREA_LABELS = {True: "inside city", False: "outside city"}                  # [SOURCE] whether a call is inside Gainesville (the record's focal flag)
GEOD = pyproj.Geod(ellps="WGS84")                                           # [STANDARD] straight-line distances on the WGS84 ellipsoid
SNAP_WARN_M = 500                                                           # [DESIGN] a call farther than this from the road network is reported
TRAVEL_TEST_FROM = pd.Timestamp("2025-07-01")                               # [DESIGN] fitted inputs use the runs before this date and are tested on the later runs
RETURN_CHECK_SECONDS = 3600                                                 # [DESIGN] an engine that cleared longer ago than this is back at its station (no drive home is longer)
NODES_KEPT = 512                                                            # [DESIGN] shortest-path arrays kept in memory, one per road node
RETURN_CORRIDOR_FACTOR = 1.5                                                # [DESIGN] an engine is on its way home when the distance from the scene through the engine to its station is at most this many times the direct distance
RETURN_MIN_KM = 0.3                                                         # [DESIGN] a scene this close to the engine's station gives no usable share of the way home
RETURN_MINUTE_BANDS = [0, 1, 2, 4, 8, 60]                                   # [DESIGN] bands of minutes since clearing, for checking the drive home
PRIORITY_RANK = {"High": 0, "Moderate": 1, "Low": 2}                        # [SOURCE] SOC 2024 Appendix E risk levels; runs waiting for an engine are served in this order, runs with no level last
TRAVEL_SPREAD_BINS = 8                                                      # [DESIGN] equal-count bins of modeled travel per area for the travel spread; about 1,000 fitting runs each
TRAVEL_SPREAD_SEED = 20241001                                               # [DESIGN] seed of the runs' travel quantiles; they are drawn once, so every policy sees the same draw
BRUSH_SPREAD_BINS = 2                                                       # [DESIGN] bins of modeled travel per area for brush fires' own travel spread; 58-97 fitting runs each
QUINT_PENALTY_GRID_S = np.arange(-300, 301, 10)                             # [DESIGN] candidate quint penalties per call type, seconds; negative means CAD prefers the quint
STUDY_FIRE_PROTOCOLS = ["F67", "F69", "F71", "F82"]                         # [SOURCE] the study's fire calls: outside, structure, vehicle and brush fires
BRUSH_FIRE_PROTOCOL = "F82"                                                 # [SOURCE] MPDS fire protocol 82: vegetation, wildland, brush or grass fire
QUINT_RULE_CALL_TYPES = {"F56": "F56 elevator", BRUSH_FIRE_PROTOCOL: "F82 brush fire"}  # [SOURCE] in 2025 CAD sent the quint to 98% of elevator calls at stations with both types, and a quint to 5-10% of multi-engine brush fires; every other call type shares one penalty
TRAINING_MONTHS_END = pd.Timestamp("2025-04-01")                           # [SOURCE] the training months are October to March
READINESS_CELL_M = 1000                                                     # [DESIGN] side of the square cells of the fire-call map, metres
PROJECTED_CRS = "EPSG:32617"                                                # [STANDARD] WGS 84 / UTM zone 17N: metres over Alachua County
LOW_ACUITY_SERVICE_PROTOCOLS = ["F53"]                                      # [SOURCE] low-acuity calls are SOC low-priority medical calls and service calls (F53 citizen assist)
HIGH_ACUITY_LEVELS = ["C", "D", "E"]                                        # [SOURCE] Charlie, Delta and Echo medical calls, the high-acuity class
CALL_CLASSES = ["study fire", "high-acuity medical", "low-acuity", "other"]  # [DESIGN] call classes of the learned policy's reward, in the order run_attributes numbers them
OUT_OF_SERVICE_PROTOCOL = "OUT"                                             # [DESIGN] protocol code of the stand-in run that holds an engine while it is out of service
OUT_OF_SERVICE_WAIT_RANK = -1                                               # [DESIGN] a pending outage takes its engine at its next clear, before any waiting call (call ranks start at 0)
VERBOSE = True                                                              # [DESIGN] print every input and output table


def show_table(name, table, verbose=VERBOSE):
    """Print a table's name, shape and first five rows, with every column.

    Every function that builds a table calls this, so that a reader can check each intermediate result.

    Input
        name    : the label printed above the table (str), e.g. "engines (input)"
        table   : the table to print (DataFrame)
        verbose : whether to print anything (bool)
    Output
        None. The table is printed when verbose is True.
    Example
        show_table("demo", pd.DataFrame({"a": [1, 2]})) prints "--- demo (2, 1)" followed by the two rows.
    """
    if verbose:
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(f"--- {name} {table.shape}\n{table.head(5).to_string()}\n")


# ----------------------------------------------------------------------------------------------------------------------
# The 2025 record
# ----------------------------------------------------------------------------------------------------------------------

def load_inputs(verbose=VERBOSE):
    """Load the 2025 dispatch record, in which each row is one engine sent to a call.

    Input
        verbose : whether to print the first rows and the counts (bool)
    Output
        The record (DataFrame, 37,194 rows and 32 columns). Each row describes one engine run: the call
        (incident_id, protocol, priority, location, jurisdiction), the engine (engine_id, type, agency and its
        position when sent) and the four CAD times (dispatched_at, enroute_at, arrived_at, cleared_at).
    Example
        load_inputs().shape -> (37194, 32)
    """
    dispatches = pd.read_parquet(DISPATCHES_PARQUET)
    show_table("2025 engine dispatches (input)", dispatches, verbose)
    if verbose:
        print(f"incidents {dispatches.incident_id.nunique():,}, engines {dispatches.engine_id.nunique()}, "
              f"time columns {dispatches[TIME_COLUMNS].dtypes.astype(str).to_dict()}")
    return dispatches


def build_runs(dispatches, verbose=VERBOSE):
    """Turn the record into runs that the simulator can replay: convert times to seconds and repair broken timelines.

    The simulator counts time in seconds from DAY_ZERO, midnight on 1 October 2024. A few runs in the record have times
    that no engine could follow, so they are repaired. A negative turnout or on-scene time becomes 0. A run that arrived
    but never cleared stays on scene for the median on-scene time of complete runs (730 s). A run that neither arrived
    nor cleared is released after the median dispatch-to-clear time of cancelled runs (333 s).

    Input
        dispatches : the record from load_inputs (DataFrame, one row per run)
        verbose    : whether to print checks of the raw times and the repaired runs (bool)
    Output
        The runs (DataFrame, one row per run, same row labels as dispatches) with the columns incident_id, engine_id,
        call_received_at, dispatched_at, enroute_at, arrived_at and cleared_at. Times are seconds since DAY_ZERO.
    Example
        build_runs(dispatches).loc[0, ["engine_id", "call_received_at", "dispatched_at"]].tolist()
        -> ['E62', 6775.0, 6796.0]          # the call came in at 01:52:55 and E62 was sent at 01:53:16
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
    repaired = runs.copy()
    complete = repaired["arrived_at"].notna() & repaired["cleared_at"].notna()
    cancelled = repaired["arrived_at"].isna() & repaired["cleared_at"].notna()
    median_on_scene = (repaired["cleared_at"] - repaired["arrived_at"])[complete].median()
    median_cancelled = (repaired["cleared_at"] - repaired["dispatched_at"])[cancelled].median()
    repaired["enroute_at"] = repaired["enroute_at"].clip(lower=repaired["dispatched_at"])
    repaired["cleared_at"] = repaired["cleared_at"].clip(lower=repaired["arrived_at"])
    repaired["cleared_at"] = repaired["cleared_at"].fillna(repaired["arrived_at"] + median_on_scene)
    repaired["cleared_at"] = repaired["cleared_at"].fillna(repaired["dispatched_at"] + median_cancelled)
    if verbose:
        changed = (repaired[TIME_COLUMNS] != runs[TIME_COLUMNS]) & ~(repaired[TIME_COLUMNS].isna() & runs[TIME_COLUMNS].isna())
        rows = changed.any(axis=1)
        print(f"--- timelines repaired: {int(rows.sum())} runs (median on-scene {median_on_scene:.0f} s, "
              f"median cancelled {median_cancelled:.0f} s)\n"
              f"{pd.concat({'recorded': runs.loc[rows, TIME_COLUMNS], 'repaired': repaired.loc[rows, TIME_COLUMNS]}, axis=1).to_string()}\n")
    return repaired


def load_engines(verbose=VERBOSE):
    """Load the engines with their type, agency, home station and the dates each one is in service.

    An engine whose first dispatch in the record falls within SERVICE_EDGE_DAYS of the year's start counts as in service
    from the start; otherwise its service starts at that first dispatch. The end of service works the same way.

    Input
        verbose : whether to print the engines and those in service for part of the year only (bool)
    Output
        The engines (DataFrame, one row per engine, 29 rows) with engine_id, engine_type ("E" engine or "Q" quint),
        engine_agency, home_station, home_lon, home_lat, in_service_from and in_service_to.
    Example
        load_engines().loc[0, ["engine_id", "home_station", "in_service_from"]].tolist()
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


def load_aid_protocols(dispatches, verbose=VERBOSE):
    """Find the call types that may get an engine from the other agency: those that got automatic aid in FY2024.

    GFR and ACFR send engines across their line only for "a limited set of emergent calls" (SOC 2024, printed p. 31),
    and that list is not published. The function therefore learns the list from the FY2024 aid dispatches, so that the
    2025 record can test it. Every call type with at least one engine sent by automatic aid counts.

    Input
        dispatches : the 2025 record (DataFrame), used only to print how many of its runs the list covers
        verbose    : whether to print the aid dispatches and the coverage (bool)
    Output
        The call types (protocol codes) that may get the other agency's engine (set of str).
    Example
        sorted(load_aid_protocols(dispatches))[:3] -> ['E02', 'E06', 'E09']
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


# ----------------------------------------------------------------------------------------------------------------------
# Roads and travel time
# ----------------------------------------------------------------------------------------------------------------------

def straight_line_km(lon_from, lat_from, lon_to, lat_to):
    """Compute the straight-line (geodesic) distance in km between points on the WGS84 ellipsoid.

    Input
        lon_from, lat_from : longitudes and latitudes of the start points, degrees (floats or arrays)
        lon_to, lat_to     : longitudes and latitudes of the end points, degrees (same shape)
    Output
        The distances in km (float or array of the same shape).
    Example
        straight_line_km(-82.3249, 29.6468, -82.3550, 29.6355) -> 3.17
    """
    return GEOD.inv(lon_from, lat_from, lon_to, lat_to)[2] / 1000


def load_road_network(verbose=VERBOSE):
    """Load the county's drive network and keep its largest strongly connected part, so every node can reach every other.

    Input
        verbose : whether to print the network's size and edge speeds (bool)
    Output
        The road network (networkx MultiDiGraph from osmnx). Each edge has a free-flow travel_time in seconds.
    Example
        load_road_network().number_of_nodes() -> 15048
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
    """Find the road node nearest to each run's call and to each engine's home station.

    Input
        graph      : the road network from load_road_network
        dispatches : the record (DataFrame, one row per run); a call without coordinates gets no node
        engines    : the engines from load_engines (DataFrame)
        verbose    : whether to print the snap distances (bool)
    Output
        Two tables. call_nodes (DataFrame, one row per run): node (road node id, nullable int) and snap_m (metres from
        the call to that node). engine_nodes (DataFrame, one row per engine): engine_id, node and snap_m.
    Example
        snap_to_network(graph, dispatches, engines)[1].loc[0, ["engine_id", "snap_m"]].tolist() -> ['E1', 12.3...]
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


def make_seconds_from_node(graph):
    """Build a function that gives the free-flow network seconds from one road node to every node.

    The returned function computes each node's shortest paths once and keeps them. When it already keeps NODES_KEPT
    nodes and needs a new one, it drops the least recently used node.

    Input
        graph : the road network from load_road_network
    Output
        Two objects. seconds_from_node (function): it takes a road node id and returns an array of seconds, one entry per
        node, in the order of node_position. node_position (Series): the position of each road node in those arrays,
        indexed by road node id.
    Example
        seconds_from_node, node_position = make_seconds_from_node(graph)
        seconds_from_node(84918009)[node_position[84855238]] -> about 41.0
    """
    node_list = list(graph.nodes)
    node_position = pd.Series(np.arange(len(node_list)), index=node_list)
    matrix = nx.to_scipy_sparse_array(ox.convert.to_digraph(graph, weight="travel_time"), nodelist=node_list, weight="travel_time")

    @functools.lru_cache(maxsize=NODES_KEPT)
    def seconds_from_node(node):
        return scipy.sparse.csgraph.dijkstra(matrix, directed=True, indices=int(node_position[node]))

    return seconds_from_node, node_position


def network_from_stations(dispatches, call_nodes, engine_nodes, seconds_from_node, node_position, verbose=VERBOSE):
    """Compute the free-flow network seconds from every engine's home station to every run's call (fastest path).

    Input
        dispatches        : the record (DataFrame, one row per run), which gives the row labels
        call_nodes        : each run's call node, from snap_to_network
        engine_nodes      : each engine's home node, from snap_to_network
        seconds_from_node : the function from make_seconds_from_node
        node_position     : each road node's position in the arrays of seconds_from_node (Series, indexed by node id)
        verbose           : whether to print the table's first rows (bool)
    Output
        The network seconds (DataFrame, rows: runs, columns: engines). A call without coordinates gets NaN.
    Example
        network_from_stations(dispatches, call_nodes, engine_nodes, seconds_from_node, node_position).loc[0, "E62"]
        -> about 287.0
    """
    located = call_nodes["node"].notna().to_numpy()
    call_positions = node_position[call_nodes.loc[located, "node"].astype("int64")].to_numpy()
    values = np.full((len(dispatches), len(engine_nodes)), np.nan)
    for column, node in enumerate(engine_nodes["node"].astype("int64")):
        values[located, column] = seconds_from_node(node)[call_positions]
    network_seconds = pd.DataFrame(values, index=dispatches.index, columns=engine_nodes["engine_id"])
    show_table("network free-flow seconds, home station -> call (rows: runs)", network_seconds.round(0), verbose)
    if verbose:
        print(f"unreachable pairs: {int(np.isinf(network_seconds.to_numpy()).sum())}\n")
    return network_seconds


def fit_travel_model(dispatches, network_seconds, verbose=VERBOSE):
    """Fit the travel model: a straight line that predicts an engine's travel seconds from free-flow network seconds.

    The model is travel = a + b x network seconds. It is fitted by least mean absolute error on runs whose engine left
    from its station before TRAVEL_TEST_FROM, and it is tested on the later runs. The fit leaves out runs whose recorded
    travel is 0 or less and runs whose quality note contains "implausible" or "equals_incident".

    Input
        dispatches      : the record (DataFrame, one row per run), with each run's recorded travel_s
        network_seconds : free-flow network seconds from every station to every call (from network_from_stations)
        verbose         : whether to print the fitted parameters and the test errors (bool)
    Output
        Two objects. travel_model (function): it takes free-flow network seconds (float or array) and returns modeled
        travel seconds. check (DataFrame, one row): the number of test runs, and the mean absolute error, the median
        absolute error and the mean signed error (bias) of the test runs, in minutes.
    Example
        travel_model, check = fit_travel_model(dispatches, network_seconds)
        travel_model(300.0) -> about 449.0        # a = 148.4 s, b = 1.002
    """
    rows = np.arange(len(dispatches))
    own_engine = network_seconds.columns.get_indexer(dispatches["engine_id"])
    seconds = network_seconds.to_numpy()[rows, own_engine]
    observed = dispatches["travel_s"].to_numpy(dtype=float, na_value=np.nan)
    usable = (dispatches["at_home"].astype(bool).to_numpy() & (observed > 0) & np.isfinite(seconds)
              & ~dispatches["qc_note"].str.contains("implausible|equals_incident").to_numpy())
    train = usable & (dispatches["dispatched_at"] < TRAVEL_TEST_FROM).to_numpy()
    test = usable & ~train

    def linear_seconds(parameters, free_flow_seconds):
        intercept, slope = parameters
        return intercept + slope * free_flow_seconds

    result = scipy.optimize.minimize(lambda parameters: np.mean(np.abs(linear_seconds(parameters, seconds[train]) - observed[train])),
                                     [60.0, 1.0], method="Nelder-Mead", options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-8})
    error_min = (linear_seconds(result.x, seconds[test]) - observed[test]) / 60
    check = pd.DataFrame({"linear on network time": {"test runs": len(error_min), "mean |error| min": np.mean(np.abs(error_min)),
                                                     "median |error| min": np.median(np.abs(error_min)), "bias min": np.mean(error_min)}}).T
    if verbose:
        print(f"travel model fitted on {train.sum():,} runs before {TRAVEL_TEST_FROM.date()} that left from home, tested on "
              f"{test.sum():,} later runs: travel = {result.x[0]:.1f} s + {result.x[1]:.3f} x network seconds\n{check.round(3).to_string()}\n")
    return (lambda free_flow_seconds: linear_seconds(result.x, free_flow_seconds)), check


def estimate_network_travel_seconds(dispatches, engines, network_seconds, travel_model, verbose=VERBOSE):
    """Compute the modeled travel seconds from every engine's home station to every run's call.

    The modeled travel is NaN for an engine that is out of service at the run's dispatch time, and for every engine
    when the call has no location.

    Input
        dispatches      : the runs to estimate for (DataFrame with dispatched_at; the record or a scenario's table)
        engines         : the engines (DataFrame with in_service_from and in_service_to)
        network_seconds : free-flow network seconds, rows: those runs, columns: engines
        travel_model    : the function from fit_travel_model
        verbose         : whether to print the table and the rank of the engine that the record sent (bool)
    Output
        The modeled travel (DataFrame, rows: runs, columns: engines, seconds).
    Example
        estimate_network_travel_seconds(dispatches, engines, network_seconds, travel_model).shape -> (37194, 29)
    """
    estimates = pd.DataFrame(travel_model(network_seconds.to_numpy()), index=dispatches.index, columns=network_seconds.columns)
    for engine in engines.itertuples():
        out_of_service = ~dispatches["dispatched_at"].between(engine.in_service_from, engine.in_service_to)
        estimates.loc[out_of_service, engine.engine_id] = np.nan
    show_table("estimated travel seconds along the road network, home station -> call (rows: runs)", estimates.round(0), verbose)
    if verbose:
        rank = estimates.rank(axis=1, method="min")                          # rank 1 is the shortest; engines at the same node tie
        rank_of_recorded_engine = rank.to_numpy()[np.arange(len(rank)), rank.columns.get_indexer(dispatches["engine_id"])]
        print("rank of the engine the record sent among in-service engines (by estimated network travel from home):",
              pd.Series(rank_of_recorded_engine).clip(upper=4).map({1: "1st", 2: "2nd", 3: "3rd", 4: "4th or later"})
              .fillna("no estimate").value_counts(normalize=True).round(3).to_dict(), "\n")
    return estimates


def apply_agency_boundaries(estimated_travel, dispatches, engines, aid_protocols, verbose=VERBOSE):
    """Remove engines across the agency line from the modeled travel, except for call types with automatic aid.

    GFR engines serve GFR's jurisdiction and every other engine serves the rest of the county. An engine of the other
    agency keeps its travel only when the call type gets automatic aid (aid_protocols); otherwise its travel becomes NaN.

    Input
        estimated_travel : modeled travel (DataFrame, rows: runs, columns: engines, seconds)
        dispatches       : the same runs (DataFrame with jurisdiction and protocol)
        engines          : the engines (DataFrame with engine_agency)
        aid_protocols    : call types that may get the other agency's engine (set of str, from load_aid_protocols)
        verbose          : whether to print how many run-engine pairs the line removes (bool)
    Output
        The modeled travel with the agency line applied (DataFrame, same shape; removed run-engine pairs are NaN).
    Example
        apply_agency_boundaries(estimated_travel, dispatches, engines, aid_protocols).loc[0].isna().sum() -> 13
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
              f"runs whose recorded engine is now outside the rule: {int(engine_removed.sum())} ({engine_removed.mean():.2%})\n")
    return bounded


# ----------------------------------------------------------------------------------------------------------------------
# Where engines are: the drive home after a call
# ----------------------------------------------------------------------------------------------------------------------

def return_progress(elapsed_s, drive_home_s, dwell_s=0.0, slowdown=1.0):
    """Compute how much of the drive home an engine has finished some time after it cleared a call.

    The engine first stays dwell_s seconds near the scene. It then drives home, and this drive takes slowdown times as
    long as the modeled drive of drive_home_s seconds.

    Input
        elapsed_s    : seconds since the engine cleared the call (float), e.g. 300.0
        drive_home_s : modeled drive from the scene to the engine's station, in seconds (float), e.g. 600.0
        dwell_s      : seconds the engine stays near the scene before it drives off (float); fitted value 0.0
        slowdown     : the real drive time divided by the modeled drive time (float); fitted value 1.18
    Output
        The fraction of the drive home that the engine has finished, a number from 0 to 1 (float). For example, 0.42
        means the engine has covered 42% of the way home. The function returns None when the engine is already home.
    Example
        return_progress(300.0, 600.0, 0.0, 1.18) -> 0.42      # 300 / (1.18 x 600)
        return_progress(800.0, 600.0, 0.0, 1.18) -> None      # the drive takes 1.18 x 600 = 708 s, so it is home
    """
    if elapsed_s >= dwell_s + slowdown * drive_home_s:
        return None
    return max(0.0, elapsed_s - dwell_s) / (slowdown * drive_home_s)


def fit_return_progress(dispatches, engines, call_nodes, home_nodes, seconds_from_node, node_position, travel_model,
                        verbose=VERBOSE):
    """Fit the dwell time and the slowdown of return_progress to where engines really were when CAD sent them.

    The record gives each engine's AVL position when CAD sent it. The fit uses runs whose engine had arrived at its
    previous call, cleared it less than RETURN_CHECK_SECONDS before, and stood between that scene and its station or at
    the station itself. An engine stands between the two when the straight-line distance from the scene through the
    engine to the station is at most RETURN_CORRIDOR_FACTOR times the direct distance from the scene to the station.
    The fit keeps engines that are already home, because keeping only engines still on the road would favor the slow
    drives and bias the fit. The actual share of the way home is the distance from the scene to the engine divided by
    the sum of that distance and the distance from the engine to the station. The fit minimizes the mean absolute share
    error on runs before TRAVEL_TEST_FROM. The later runs test the fit and compare it with the model that has no dwell
    and no slowdown.

    Input
        dispatches        : the record (DataFrame, one row per run)
        engines           : the engines (DataFrame with home_lon and home_lat)
        call_nodes        : each run's call node, from snap_to_network
        home_nodes        : the road node of each engine's home station (Series, indexed by engine id)
        seconds_from_node : the function from make_seconds_from_node
        node_position     : each road node's position in the arrays of seconds_from_node (Series, indexed by node id)
        travel_model      : the function from fit_travel_model
        verbose           : whether to print the fit and the test errors by minutes since clearing (bool)
    Output
        Three objects: the fitted dwell in seconds (float), the fitted slowdown (float), and check (DataFrame, two rows:
        "no dwell, no slowdown" and "AVL-fitted"; columns: test runs, mean absolute share error, median predicted share).
    Example
        fit_return_progress(dispatches, engines, call_nodes, home_nodes, seconds_from_node, node_position, travel_model)[:2]
        -> about (0.0, 1.18)
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
    sample["drive_home_s"] = [float(travel_model(seconds_from_node(int(scene_node))[node_position[home_nodes[engine_id]]]))
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
    test["no dwell, no slowdown"] = predicted_share(test, 0.0, 1.0)
    test["AVL-fitted"] = predicted_share(test, dwell_s, slowdown)
    check = pd.DataFrame({name: {"test runs": len(test), "mean |share error|": np.mean(np.abs(test[name] - test.actual_share)),
                                 "median predicted share": np.median(test[name])}
                          for name in ["no dwell, no slowdown", "AVL-fitted"]}).T
    if verbose:
        minute_band = pd.cut(test.elapsed_s / 60, RETURN_MINUTE_BANDS)
        by_minute = test.groupby(minute_band, observed=True)[["actual_share", "no dwell, no slowdown", "AVL-fitted"]].median()
        by_minute["share at home"] = test.groupby(minute_band, observed=True).at_home.mean()
        by_minute["runs"] = test.groupby(minute_band, observed=True).size()
        print(f"runs sent within {RETURN_CHECK_SECONDS // 60} min of clearing a call they reached, in the scene -> home corridor: "
              f"{len(sample):,}, {sample.at_home.mean():.0%} at home (fit {train.sum():,} before {TRAVEL_TEST_FROM.date()}, "
              f"test {len(test):,}); median actual share of the way home {sample.actual_share.median():.2f}\n"
              f"fitted ({result.message}): dwell {dwell_s:.0f} s near the scene, then {slowdown:.2f} x the modeled drive home\n"
              f"{check.round(3).to_string()}\nmedian share by minutes since clear (test runs):\n{by_minute.round(2).to_string()}\n")
    return dwell_s, slowdown, check


def share_of_way_home(engine_id, last_clear, now, call_nodes, home_nodes, seconds_from_node, node_position, travel_model,
                      dwell_s=0.0, slowdown=1.0):
    """Find where an engine is at a given moment: on its way home from its last scene, or at its station.

    The simulator stores each engine's last clear in last_clear. After a call the engine reached, last_clear is
    (clear time, row of that run), and the share of the way home is return_progress(elapsed, drive home, dwell_s,
    slowdown). After a cancelled run, last_clear is (release time, row of that run, f), where f is the share of the way
    to that call that the engine had driven. Such an engine drives straight back without dwelling, so its share of the
    way home is 1 - f x (1 - b), where b is the share of the drive back that it has finished.

    Input
        engine_id         : the engine (str), e.g. "E3"
        last_clear        : the engine's last clear in the simulation (tuple, see above), or None if it has not run yet
        now               : the current simulation time, seconds since DAY_ZERO (float)
        call_nodes        : each run's call node (DataFrame with column node)
        home_nodes        : the road node of each engine's home station (Series, indexed by engine id)
        seconds_from_node : the function from make_seconds_from_node
        node_position     : each road node's position in the arrays of seconds_from_node (Series, indexed by node id)
        travel_model      : the function from fit_travel_model
        dwell_s, slowdown : the drive-home parameters from fit_return_progress (floats)
    Output
        A pair. While the engine drives home: (share of the way home, array of network seconds from its last scene to
        every node). Otherwise (None, None), which means the engine is at its station: it has not run yet, its last run
        left no scene, it cleared RETURN_CHECK_SECONDS or more ago, the scene has no location, or it is already home.
    Example
        An engine that cleared 300 s ago from a scene whose modeled drive home is 600 s (dwell 0, slowdown 1.18):
        share_of_way_home(engine_id, (cleared_s, scene_row), cleared_s + 300, ...)[0] -> 0.42      # 300 / (1.18 x 600)
        share_of_way_home(engine_id, None, now, ...) -> (None, None)                             # has not run yet
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
    drive_home = float(travel_model(from_scene[node_position[home_nodes[engine_id]]]))
    if outbound:                                                 # released from a cancelled run after a share f of the way out
        progress_back = return_progress(elapsed, outbound[0] * drive_home, 0.0, slowdown)
        share_home = None if progress_back is None else 1 - outbound[0] * (1 - progress_back)
    else:
        share_home = return_progress(elapsed, drive_home, dwell_s, slowdown)
    return (None, None) if share_home is None else (share_home, from_scene)


def travel_while_returning(row, now, engine_state, travel_s, call_nodes, home_nodes, seconds_from_node, node_position,
                           travel_model, dwell_s=0.0, slowdown=1.0):
    """Correct a run's modeled travel for engines that are still driving home from their last call.

    For an engine that has covered a share p of its way home from scene S, the modeled travel to the call is
    (1 - p) x travel(S -> call) + p x travel(home -> call). Every other engine keeps its travel from its station. The
    travel model must use network seconds only, as the linear model of fit_travel_model does.

    Input
        row               : the run's row label (int)
        now               : the current simulation time, seconds since DAY_ZERO (float)
        engine_state      : each engine's last clear, keyed by engine id (dict, kept by simulate)
        travel_s          : each engine's modeled travel from its station to this run's call, in seconds (Series indexed
                            by engine id; NaN means the engine is not allowed)
        call_nodes        : each run's call node (DataFrame with column node)
        home_nodes, seconds_from_node, node_position, travel_model, dwell_s, slowdown : as in share_of_way_home
    Output
        Each engine's modeled travel from where it is now to this call, in seconds (Series indexed by engine id). For a
        call without a location the function returns the input travel unchanged; otherwise it returns a new copy.
    Example
        travel_while_returning(row, now, engine_state, record_travel.loc[row], call_nodes, home_nodes, seconds_from_node,
                               node_position, travel_model)["E3"] -> about 512.0
    """
    call_node = call_nodes.at[row, "node"]
    if pd.isna(call_node):
        return travel_s
    adjusted = travel_s.copy()
    for engine_id, last_clear in engine_state.items():
        if pd.isna(adjusted.get(engine_id)):
            continue
        share_home, from_scene = share_of_way_home(engine_id, last_clear, now, call_nodes, home_nodes, seconds_from_node,
                                                   node_position, travel_model, dwell_s, slowdown)
        if share_home is None:
            continue
        from_scene_to_call = float(travel_model(from_scene[node_position[int(call_node)]]))
        adjusted[engine_id] = (1 - share_home) * from_scene_to_call + share_home * travel_s[engine_id]
    return adjusted


# ----------------------------------------------------------------------------------------------------------------------
# Real travel time and engine types
# ----------------------------------------------------------------------------------------------------------------------

def draw_travel_ratio(spread, inside_city, modeled_seconds, quantile):
    """Draw the ratio of real to modeled travel for one trip, at a given quantile of the ratios of similar trips.

    Similar trips are fitting trips in the same area (inside or outside the city) whose modeled travel falls in the
    same bin. Bins are right-closed, as pd.qcut makes them. A trip shorter or longer than every fitting trip uses the
    first or the last bin.

    Input
        spread          : the fitted travel spread from fit_travel_spread (dict keyed by inside_city). Each value is a
                          pair: the inner bin edges in seconds, and one sorted array of observed ratios per bin.
        inside_city     : whether the call is inside the city (bool)
        modeled_seconds : the trip's modeled travel, in seconds (float)
        quantile        : the quantile to draw at, between 0 and 1 (float); each run has its own
    Output
        The ratio of real travel to modeled travel (float).
    Example
        toy = {True: (np.array([300.0]), [np.array([0.5, 1.0, 2.0]), np.array([0.9, 1.0, 1.1])])}
        draw_travel_ratio(toy, True, 200.0, 0.5) -> 1.0     # 200 s falls in the first bin; its median ratio
        draw_travel_ratio(toy, True, 200.0, 0.99) -> 2.0
        draw_travel_ratio(toy, True, 400.0, 0.99) -> 1.1    # 400 s falls in the second bin
    """
    inner_edges, ratios_by_bin = spread[inside_city]
    ratios = ratios_by_bin[int(np.searchsorted(inner_edges, modeled_seconds, side="left"))]
    return float(np.quantile(ratios, quantile, method="inverted_cdf"))


def fit_travel_spread(dispatches, network_seconds, travel_model, spread_quantile, bin_count=TRAVEL_SPREAD_BINS,
                      verbose=VERBOSE):
    """Fit how real travel spreads around modeled travel, and test the fit on held-out months.

    The fit uses the travel model's own fitting runs: runs whose engine left from its station before TRAVEL_TEST_FROM.
    For each run, the ratio is recorded travel divided by modeled travel. The ratios are kept per area (call inside or
    outside the city) and per bin of modeled travel; each area has bin_count bins with equal numbers of fitting runs.
    The later runs (July to September) test the fit. As in the simulation, each test run gets its modeled travel times
    the ratio at the run's own quantile. If the fit is right, recorded travel falls below the predicted 10th percentile
    and above the predicted 90th percentile about 10% of the time each. Brush fires get their own travel spread: pass
    only their rows and bin_count=BRUSH_SPREAD_BINS.

    Input
        dispatches      : the runs to fit on (DataFrame with travel_s, at_home, focal, qc_note, dispatched_at)
        network_seconds : free-flow network seconds for the same runs (rows: runs, columns: engines)
        travel_model    : the function from fit_travel_model
        spread_quantile : each run's quantile, between 0 and 1 (Series indexed by run row)
        bin_count       : the number of bins of modeled travel per area (int)
        verbose         : whether to print the trips, the bins and the check (bool)
    Output
        Three objects. spread (dict, the input of draw_travel_ratio). spread_table (DataFrame, one row per area and bin):
        the bin's range of modeled travel in minutes, the number of its fitting runs, the ratio's percentiles P1, P10,
        P50, P90 and P99, the number of held-out runs, and the shares of held-out runs below P10 and above P90.
        spread_check (DataFrame, rows: all, inside city, outside city): the number of test runs, the P10, P50 and P90 of
        recorded, modeled and drawn travel in minutes, and the shares of test runs whose recorded travel falls below the
        predicted P10 and above the predicted P90.
    Example
        travel_spread, spread_table, spread_check = fit_travel_spread(dispatches, network_seconds, travel_model, spread_quantile)
        spread_table.loc[(False, 0), "ratio P50"] -> 1.097      # the shortest trips outside the city take ~10% longer
    """
    rows = np.arange(len(dispatches))
    own_engine = network_seconds.columns.get_indexer(dispatches["engine_id"])
    seconds = network_seconds.to_numpy()[rows, own_engine]
    recorded = dispatches["travel_s"].to_numpy(dtype=float, na_value=np.nan)
    usable = (dispatches["at_home"].astype(bool).to_numpy() & (recorded > 0) & np.isfinite(seconds)
              & ~dispatches["qc_note"].str.contains("implausible|equals_incident").to_numpy())
    trips = pd.DataFrame({"inside_city": dispatches["focal"].fillna(False).astype(bool).to_numpy(),
                          "modeled_s": travel_model(seconds), "recorded_s": recorded,
                          "fitting_run": (dispatches["dispatched_at"] < TRAVEL_TEST_FROM).to_numpy()},
                         index=dispatches.index)[usable]
    trips["ratio"] = trips["recorded_s"] / trips["modeled_s"]
    # ponytail: the ratios are kept as recorded, including CAD button slips (P1 as low as 0.02 on the shortest trips); trim them if a use needs physical travel only
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
              f"from home, tested on {len(test):,} later runs (held out from the ratios)\n"
              f"--- ratio recorded / modeled travel by area and bin of modeled travel; held-out shares outside the "
              f"bin's P10-P90 (about 0.10 each if calibrated)\n{spread_table.round(3).to_string()}\n"
              f"--- held-out check (Jul-Sep): travel minutes, and how often recorded travel falls outside the predicted "
              f"P10-P90 (about 0.10 each if calibrated)\n{spread_check.round(3).T.to_string()}\n")
    return spread, spread_table, spread_check


def fit_quint_penalty(dispatches, runs, engines, travel, call_type_of_run, verbose=VERBOSE):
    """Fit how many seconds a quint must save over an engine before CAD sends it, separately for each call type.

    A quint is a pumper that also carries an aerial ladder. CAD does not always send the closest unit when a quint could
    go: for an elevator call CAD prefers the quint, and for a brush fire it rarely sends one (a quint went to 5-10% of multi-engine brush fires). The fit uses
    single-engine calls whose engine left from its station, because there the choice and its starting point are known.
    The engines CAD could send are those with a modeled travel (in service and allowed by the agency line) that were not
    on a run at that moment. The rule adds the penalty to the modeled travel of each quint and then sends the engine
    with the least travel. For each call type, the fit keeps the penalty in QUINT_PENALTY_GRID_S under which the rule
    most often sends CAD's engine in the fitting months (before TRAVEL_TEST_FROM). The test months (July to September)
    check that penalty. A negative penalty means CAD prefers the quint unless an engine is that much closer.

    Input
        dispatches       : the record (DataFrame, one row per run)
        runs             : the repaired runs from build_runs, which say when each engine was busy
        engines          : the engines (DataFrame with engine_type)
        travel           : modeled travel with the agency line (DataFrame, rows: runs, columns: engines, seconds)
        call_type_of_run : each run's call-type label (Series indexed by run row): "F56 elevator", "F82 brush fire" or
                           "all other calls"
        verbose          : whether to print the chosen penalties and the agreement around them (bool)
    Output
        Two objects. penalty_by_type (dict): the penalty in seconds for each call-type label. quint_penalty_fit
        (DataFrame, rows: call type and penalty; columns "fit: all" and "test: all"): the share of runs where the rule
        sends CAD's engine.
    Example
        penalty_by_type, quint_penalty_fit = fit_quint_penalty(dispatches, runs, engines, record_travel, call_type_of_run)
        penalty_by_type -> {'F56 elevator': -260.0, 'F82 brush fire': 260.0, 'all other calls': 10.0}
    """
    engine_ids = travel.columns
    is_quint = (engines.set_index("engine_id").loc[engine_ids, "engine_type"] == "Q").to_numpy()
    penalty_by_type, fit_by_type = {}, {}
    for label, label_rows in call_type_of_run.groupby(call_type_of_run).groups.items():
        part = dispatches.loc[label_rows]
        rows = part.index[(part["n_engines_sent"] == 1) & part["at_home"].astype(bool) & part["incident_lon"].notna()]
        dispatched_s = runs.loc[rows, "dispatched_at"].to_numpy()
        busy = np.zeros((len(rows), len(engine_ids)), dtype=bool)
        for column, engine_id in enumerate(engine_ids):          # busy: on a run that started before this dispatch and clears after it
            own = runs[runs["engine_id"] == engine_id].sort_values("dispatched_at")
            starts, ends = own["dispatched_at"].to_numpy(), own["cleared_at"].to_numpy()
            # ponytail: only the engine's last run that started before this dispatch is checked; overlapping runs of one engine are rare (3 in 2025)
            last = np.searchsorted(starts, dispatched_s, side="left") - 1
            busy[:, column] = (last >= 0) & (ends[np.maximum(last, 0)] > dispatched_s)
        candidate_travel = travel.loc[rows, engine_ids].to_numpy()
        candidate_travel = np.where(busy | np.isnan(candidate_travel), np.inf, candidate_travel)
        cad = engine_ids.get_indexer(part.loc[rows, "engine_id"])
        fitting = (part.loc[rows, "dispatched_at"] < TRAVEL_TEST_FROM).to_numpy()
        agreement = {}
        for penalty in QUINT_PENALTY_GRID_S:
            same = np.argmin(candidate_travel + penalty * is_quint, axis=1) == cad
            agreement[int(penalty)] = {f"{months}: all": same[in_months].mean() if in_months.any() else np.nan
                                       for months, in_months in [("fit", fitting), ("test", ~fitting)]}
        fit = pd.DataFrame(agreement).T.rename_axis("quint penalty s")
        penalty_by_type[label] = float(fit["fit: all"].idxmax())
        fit_by_type[label] = fit
    quint_penalty_fit = pd.concat(fit_by_type, names=["call type"])
    if verbose:
        chosen = pd.DataFrame({label: quint_penalty_fit.loc[(label, int(penalty))] for label, penalty in penalty_by_type.items()}).T
        chosen.insert(0, "chosen penalty s", pd.Series(penalty_by_type))
        shown = quint_penalty_fit[quint_penalty_fit.index.get_level_values(1) % 60 == 0].unstack(0)
        print(f"--- quint penalty by call type (single-engine calls, engine from its station; negative = the quint "
              f"preferred)\n{chosen.round(3).to_string()}\n"
              f"share of runs where the rule sends CAD's engine, by penalty:\n{shown.round(3).to_string()}\n")
    return penalty_by_type, quint_penalty_fit


# ----------------------------------------------------------------------------------------------------------------------
# Calls
# ----------------------------------------------------------------------------------------------------------------------

def call_fates(runs):
    """Collect what the record fixes about the outcome of each call with two or more runs.

    In such a call, the simulator decides which engines arrive for the call as a whole, not row by row. It lets the
    first engines to reach the scene arrive, up to the number of engines that arrived in the record. It releases the
    other runs at the moments when the record released them.

    Input
        runs : the runs (DataFrame with incident_id, arrived_at and cleared_at in seconds)
    Output
        A dict keyed by incident id. Each value is a tuple of three parts: the number of runs that arrived (int), the
        moments when the other runs were released (sorted list of seconds), and the on-scene seconds of the runs that
        arrived, in arrival order (list).
    Example
        call_fates(runs)["100124-0344"] -> (1, [38904.0], [121.0])     # one arrival, one release, 121 s on scene
        len(call_fates(runs)) -> 2941
    """
    fates = {}
    for incident, call in runs.groupby("incident_id"):
        if len(call) < 2:
            continue
        arrived = call[call["arrived_at"].notna()].sort_values("arrived_at")
        fates[incident] = (len(arrived), sorted(call.loc[call["arrived_at"].isna(), "cleared_at"].tolist()),
                           (arrived["cleared_at"] - arrived["arrived_at"]).tolist())
    return fates


def run_attributes(table, quint_penalty_by_type):
    """Derive the per-run attributes that the simulator and the policies read, from a dispatch table.

    Input
        table                 : a dispatch table (DataFrame with focal, priority, protocol and call_type), the record or
                                a scenario's table; stand-in runs of engine outages have protocol OUT_OF_SERVICE_PROTOCOL
        quint_penalty_by_type : the quint penalty in seconds for each call-type label (dict, from fit_quint_penalty)
    Output
        A dict of Series indexed by the table's rows:
        inside_city   : whether the call is inside Gainesville (bool)
        wait_rank     : the order in which waiting runs get an engine, lowest first: 0 for high priority, 1 for
                        moderate, 2 for low, 3 for a missing priority, and OUT_OF_SERVICE_WAIT_RANK for outage stand-ins
        quint_penalty : seconds added to a quint's travel when ranking engines for this run
        brush         : whether the call is a brush fire; brush fires draw their travel from their own travel spread
        low_acuity    : whether the call is a low-priority medical call or a service call (the opportunity-cost rule
                        may change the engine only on these)
        call_class    : the call's class for the learned policy's reward, an index into CALL_CLASSES (int): 0 a study
                        fire call, 1 a high-acuity medical call (Charlie, Delta or Echo), 2 a low-acuity call, 3 any
                        other call; the first class that applies wins
    Example
        run_attributes(dispatches, penalty_by_type)["low_acuity"].sum() -> number of low-acuity runs in the record
    """
    stand_in = table["protocol"].eq(OUT_OF_SERVICE_PROTOCOL).fillna(False)
    call_type_for_quint = table["protocol"].map(QUINT_RULE_CALL_TYPES).fillna("all other calls")
    low_acuity = ((table["call_type"].eq("Medical") & table["priority"].eq("Low")).fillna(False)
                  | table["protocol"].isin(LOW_ACUITY_SERVICE_PROTOCOLS).fillna(False))
    study_fire = table["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False).to_numpy()
    high_acuity = (table["call_type"].eq("Medical") & table["determinant_level"].isin(HIGH_ACUITY_LEVELS)).fillna(False).to_numpy()
    return {"inside_city": table["focal"].fillna(False).astype(bool),
            "wait_rank": table["priority"].astype("string").map(PRIORITY_RANK).fillna(len(PRIORITY_RANK)).mask(stand_in, OUT_OF_SERVICE_WAIT_RANK),
            "quint_penalty": call_type_for_quint.map(quint_penalty_by_type),
            "brush": table["protocol"].eq(BRUSH_FIRE_PROTOCOL).fillna(False),
            "low_acuity": low_acuity,
            "call_class": pd.Series(np.select([study_fire, high_acuity, low_acuity.to_numpy()], [0, 1, 2], 3), index=table.index)}


# ----------------------------------------------------------------------------------------------------------------------
# Fire readiness and the opportunity cost of sending an engine
# ----------------------------------------------------------------------------------------------------------------------

def build_readiness_cells(dispatches, graph, engines, home_nodes, aid_protocols, seconds_from_node, node_position,
                          travel_model, verbose=VERBOSE, fire=True):
    """Build the fire-call map for fire readiness: where fire calls happen and how long each engine takes to get there.

    With fire=False the same map is built from every other call (the non-fire demand surface of the learned policy).
    The study's fire calls in the training months (before TRAINING_MONTHS_END) are counted on square cells with sides
    of READINESS_CELL_M metres. Each cell keeps its share of those calls in two parts. One part holds the call types
    with automatic aid, to which an engine of either agency may come. The other part holds the remaining call types,
    to which only an engine of the cell's own agency may come; the cell's agency is the agency of the majority of its
    calls. The cell's position is the mean position of its calls, snapped to the road network. The fire-call intensity
    lambda_F(x, t) at cell x and hour t is then the share of cell x times the share of hour t.

    Input
        dispatches        : the record (DataFrame, one row per run)
        graph             : the road network from load_road_network
        engines           : the engines (DataFrame)
        home_nodes        : the road node of each engine's home station (Series, indexed by engine id)
        aid_protocols     : call types with automatic aid (set of str)
        seconds_from_node, node_position, travel_model : as in share_of_way_home
        verbose           : whether to print the cells and the busiest ones (bool)
        fire              : True for the study's fire calls, False for every other call (bool)
    Output
        Three objects. cells (DataFrame, one row per cell with a training fire call): cell_x, cell_y, calls_aid,
        calls_other, gfr_share, lon, lat, cell_gfr, weight_aid, weight_own, node and snap_m; the two weights sum to 1
        over all cells. travel_to_cells (DataFrame, rows: engines, columns: cells): modeled travel seconds from each
        home station. hour_share (Series, hours 0-23): the share of training fire calls received in each hour.
    Example
        cells, travel_to_cells, hour_share = build_readiness_cells(dispatches, graph, engines, home_nodes, aid_protocols,
                                                                   seconds_from_node, node_position, travel_model)
        cells[["weight_aid", "weight_own"]].to_numpy().sum() -> 1.0
    """
    study_fire = dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False)
    calls = dispatches[(study_fire if fire else ~study_fire) & dispatches["incident_lon"].notna()
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
    travel_to_cells = pd.DataFrame([travel_model(seconds_from_node(home_nodes[engine_id])[positions])
                                    for engine_id in engines["engine_id"]], index=engines["engine_id"], columns=cells.index)
    hour_share = calls["call_received_at"].dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0.0)
    if verbose:
        label = "fire calls" if fire else "non-fire calls"
        show_table(f"readiness cells (training {label} per cell)", cells, verbose)
        nearest = travel_to_cells.idxmin().to_numpy()
        print(f"training {label}: {len(calls):,} in {len(cells):,} cells of {READINESS_CELL_M} m; with automatic aid "
              f"{int(cells['calls_aid'].sum()):,}, own agency only {int(cells['calls_other'].sum()):,}; cells mostly GFR "
              f"{int(cells['cell_gfr'].sum())}; snap distance median {cells['snap_m'].median():.0f} m, P90 {cells['snap_m'].quantile(0.9):.0f} m\n"
              f"busiest cells and their nearest engine (all idle):\n"
              f"{cells.assign(nearest_engine=nearest, nearest_min=travel_to_cells.min().to_numpy() / 60).nlargest(5, 'calls_aid')[['cell_x', 'cell_y', 'calls_aid', 'calls_other', 'cell_gfr', 'nearest_engine', 'nearest_min']].round(2).to_string()}\n"
              f"share of training {label} by hour (0-23): {hour_share.round(3).tolist()}\n")
    return cells, travel_to_cells, hour_share


def make_readiness_tools(cells, travel_to_cells, engines, tau_max=None):
    """Build two functions: fire readiness for one set of idle engines, and the opportunity cost of sending an engine.

    Fire readiness R is the modeled travel from the nearest allowed idle engine to each cell, averaged over the cells
    with each cell's share of fire calls as its weight. Idle engines are at their stations. For call types with
    automatic aid, an engine of either agency is allowed; for the other call types, only an engine of the cell's own
    agency is allowed. A lower R means better protection. The opportunity cost of sending engine e is how much R rises
    if e leaves now: OC(e) = R(idle engines without e) - R(idle engines). measures.fire_readiness computes the same R
    over whole runs in a vectorized form. The two sum in a different order, so merging them would change results; the
    code keeps both so that every result stays unchanged. With tau_max, a cell that has no allowed idle engine gets the
    travel tau_max instead of an infinite one, so that R and OC stay finite for the learned policy's observation; the
    share of demand without an allowed idle engine is returned in both cases.

    Input
        cells           : the cells from build_readiness_cells
        travel_to_cells : modeled travel seconds from each engine's station to each cell (rows: engines, columns: cells)
        engines         : the engines (DataFrame with engine_agency)
        tau_max         : None to keep infinite travel for uncovered cells, or a travel in minutes that replaces it (float)
    Output
        Two functions. readiness_of(idle) takes a boolean array over the engines of travel_to_cells and returns three
        values: R in minutes, an array of each cell's travel in minutes, and the share of demand whose cell has no
        allowed idle engine. opportunity_cost(idle_engine_ids, candidate_ids) takes two lists of engine ids and returns a
        Series of OC in minutes, indexed by candidate id; with uncovered=True it returns a DataFrame with the columns oc
        and uncovered_rise (the demand share that loses its last allowed idle engine if the candidate leaves). Without
        tau_max, OC is infinite when the candidate is the only idle engine allowed in some weighted cell.
    Example
        readiness_of, opportunity_cost = make_readiness_tools(cells, travel_to_cells, engines)
        opportunity_cost(["E1", "E3", "E5"], ["E1", "E3"]) -> E1 about 0.21, E3 about 0.08 (minutes)
    """
    engine_ids = travel_to_cells.index
    position = pd.Series(np.arange(len(engine_ids)), index=engine_ids)
    travel_min = travel_to_cells.to_numpy() / 60
    own_agency = (engines.set_index("engine_id").loc[engine_ids, "engine_agency"] == "GFR").to_numpy()[:, None] == cells["cell_gfr"].to_numpy()[None, :]
    weight_aid, weight_own = cells["weight_aid"].to_numpy(), cells["weight_own"].to_numpy()

    def readiness_of(idle):
        tau_any = np.where(idle[:, None], travel_min, np.inf).min(axis=0)
        tau_own = np.where(idle[:, None] & own_agency, travel_min, np.inf).min(axis=0)
        uncovered = float(weight_aid[np.isinf(tau_any)].sum() + weight_own[np.isinf(tau_own)].sum())
        if tau_max is not None:                                  # finite travel for cells with no allowed idle engine
            tau_any, tau_own = np.minimum(tau_any, tau_max), np.minimum(tau_own, tau_max)
        weighted = np.where(weight_aid > 0, tau_any, 0.0) * weight_aid + np.where(weight_own > 0, tau_own, 0.0) * weight_own
        return float(weighted.sum()), weighted / (weight_aid + weight_own), uncovered

    def opportunity_cost(idle_engine_ids, candidate_ids, uncovered=False):
        idle = np.zeros(len(engine_ids), dtype=bool)
        idle[position[list(idle_engine_ids)].to_numpy()] = True
        now, _, uncovered_now = readiness_of(idle)
        costs, rises = {}, {}
        for engine_id in candidate_ids:
            without = idle.copy()
            without[position[engine_id]] = False
            readiness_without, _, uncovered_without = readiness_of(without)
            costs[engine_id], rises[engine_id] = readiness_without - now, uncovered_without - uncovered_now
        if uncovered:
            return pd.DataFrame({"oc": costs, "uncovered_rise": rises}, dtype=float)
        return pd.Series(costs, dtype=float)

    return readiness_of, opportunity_cost


# ----------------------------------------------------------------------------------------------------------------------
# The simulator
# ----------------------------------------------------------------------------------------------------------------------

def define_available_engines(engine_pool, now_seconds, engines):
    """List the engines that may be sent now: in service at this moment and not on a run.

    Input
        engine_pool : each engine's SimPy resource, keyed by engine id (dict); an engine on a run has count 1
        now_seconds : the current simulation time, seconds since DAY_ZERO (float)
        engines     : the engines (DataFrame with in_service_from and in_service_to)
    Output
        The available engine ids, in the engines table's order (list of str).
    Example
        define_available_engines(engine_pool, 6796.0, engines) -> ['E1', 'E2', 'E4', ...]   # 25 of 29 engines
    """
    now = DAY_ZERO + pd.Timedelta(seconds=now_seconds)
    in_service = engines.loc[engines["in_service_from"].le(now) & engines["in_service_to"].ge(now), "engine_id"]
    return [engine_id for engine_id in in_service if engine_pool[engine_id].count == 0]


def shift_arrival(times, travel_seconds):
    """Move a run's arrival and clear so that its travel (en route to arrival) takes travel_seconds.

    A run with no en-route time or no arrival time is a cancelled run, and it keeps its times.

    Input
        times          : the run's four CAD times in seconds (Series: dispatched_at, enroute_at, arrived_at, cleared_at)
        travel_seconds : the travel the run should take, in seconds (float)
    Output
        The four times with arrival and clear moved by the same amount (Series).
    Example
        shift_arrival(pd.Series({"dispatched_at": 0.0, "enroute_at": 60.0, "arrived_at": 400.0, "cleared_at": 900.0}),
                      300.0).tolist() -> [0.0, 60.0, 360.0, 860.0]
    """
    if pd.isna(times["enroute_at"]) or pd.isna(times["arrived_at"]):
        return times
    shift = times["enroute_at"] + travel_seconds - times["arrived_at"]
    return times + pd.Series({"dispatched_at": 0.0, "enroute_at": 0.0, "arrived_at": shift, "cleared_at": shift})


def simulate(world, scenario, policy, report=None, verbose=VERBOSE):
    """Simulate every call of a scenario in SimPy, with a dispatch policy choosing the engines.

    Each engine is a SimPy resource of capacity one. When a run starts, the environment lists the engines in service
    and free, and computes each one's modeled travel from where it is now (travel_while_returning). The policy then
    names the engine to send. The environment draws that engine's real travel: its modeled travel times a ratio from
    the fitted travel spread, taken at the run's own quantile. Because the quantile belongs to the run, two policies
    that send different engines to the run share its luck. The run keeps its recorded turnout and on-scene time, so its
    arrival and its clear move together by the same amount; a run that waited for an engine also has all its times
    moved by its wait.
    - A call without a location has no modeled travel. Its recorded engine goes with the recorded times when it is free,
      and otherwise the run waits for it.
    - When no engine can go, the run waits. Each clear offers the free engines to the waiting runs, in order of
      wait_rank (lower first) and then of waiting time. A run that gets an engine moves all its times by its wait. A
      run that the record cancelled stops waiting at its recorded release and gets no engine. A run with a negative
      wait_rank (an outage stand-in) looks for its engine before the runs that clear in the same second free their
      engines, so it gets an engine that clears in that second ahead of every waiting call.
    - In a call with two or more runs (scenario["fates"]), the call decides which runs arrive: the first engines to
      reach the scene arrive and take the recorded on-scene times in order, up to the recorded number of arrivals. At
      each recorded cancellation moment, the call releases one run. It releases a run that is still waiting if there
      is one, and otherwise the engine under way that would arrive last; when no engine is under way, the call does
      not send its next run.
    - An engine released on its way to a call turns home from where it was; an engine that leaves a scene drives home.

    Input
        world    : the dict from build_world (engines, road travel, drive-home and travel-spread parameters)
        scenario : the calls to simulate (dict): runs (DataFrame, one row per run, times in seconds since DAY_ZERO),
                   travel (modeled travel from every station, rows: runs, columns: engines), call_nodes, fates (dict
                   keyed by incident id, from call_fates), and spread_quantile, inside_city, wait_rank, quint_penalty,
                   brush and low_acuity (Series indexed by run row)
        policy   : the dispatch policy (function), called as policy(world, scenario, run, available_engines, travel_s,
                   context); it must return one of available_engines whose travel_s is not NaN, or None to wait. context
                   (dict) holds what is known at the decision: now (seconds since DAY_ZERO), engine_state (each engine's
                   last clear), busy_runs (the row of the run holding each busy engine, from the moment the engine is
                   reserved), simulated (the run table so far)
                   and waiting_runs (how many runs wait for an engine)
        report   : None, or an empty dict that receives: kept_recorded (rows that kept their recorded times),
                   waited_seconds (the seconds each row waited, keyed by row), never_served (rows still waiting at
                   the end), cancelled_while_waiting (rows released before getting an engine) and dispatches
                   (DataFrame, one row per dispatch: run, time_s, engine_id, the engine's last clear, its modeled
                   travel, the travel taken)
        verbose  : whether to print the first rows of the result (bool)
    Output
        The simulated runs (DataFrame, one row per run): dispatched_at, enroute_at, arrived_at and cleared_at in seconds
        since DAY_ZERO, and engine_id. A run that never got an engine has no engine_id. If the record released it while
        it waited, it keeps only its release moment as cleared_at (in a call with two or more runs, also its dispatch
        moment). If it was still waiting when the simulation ended, all its times are missing.
    Example
        simulate(world, world["record"], policies.closest_engine).loc[0].tolist()
        -> [6796.0, 6858.0, 7250.51, 7399.51, 'E62']     # the record's first run: E62 sent at 01:53:16, arrives 392.5 s after leaving
    """
    runs, estimated_travel, wait_rank, fates = scenario["runs"], scenario["travel"], scenario["wait_rank"], scenario["fates"]
    engines = world["engines"]
    env = simpy.Environment()
    engine_pool = {engine_id: simpy.Resource(env, capacity=1) for engine_id in engines["engine_id"]}
    engine_state = {}                                            # keyed by engine id: (last clear time in seconds, row of the run it left or None, share of the way out if released)
    simulated = pd.DataFrame(np.nan, index=runs.index, columns=TIME_COLUMNS)
    simulated["engine_id"] = pd.Series(index=runs.index, dtype="object")
    waiting = []                                                 # runs with no engine yet: (wait_rank, start of the wait, row, wake-up event)
    kept_recorded, waited_seconds, cancelled_while_waiting, dispatch_log = [], {}, [], []
    busy_runs = {}                                               # engine id -> row of the run holding it now

    def pick(row):
        run = runs.loc[row]
        travel_s = travel_while_returning(row, env.now, engine_state, estimated_travel.loc[row], scenario["call_nodes"],
                                          world["home_nodes"], world["seconds_from_node"], world["node_position"],
                                          world["travel_model"], world["dwell_s"], world["slowdown"])
        available = define_available_engines(engine_pool, env.now, engines)
        if travel_s.isna().all():                                # a call without a location gets its recorded engine and recorded times
            engine_id, travel_seconds = (run["engine_id"], None) if run["engine_id"] in available else (None, None)
        else:
            context = {"now": env.now, "engine_state": engine_state, "busy_runs": busy_runs, "simulated": simulated,
                       "waiting_runs": len(waiting)}
            engine_id = policy(world, scenario, run, available, travel_s, context)
            assert engine_id is None or (engine_id in available and pd.notna(travel_s[engine_id])), \
                f"the policy chose {engine_id} for run {row}, which is not available or not allowed"
            travel_seconds = None
            if engine_id is not None:
                modeled_s = float(travel_s[engine_id])
                spread = world["brush_spread"] if scenario["brush"][row] else world["travel_spread"]
                travel_seconds = modeled_s * draw_travel_ratio(spread, bool(scenario["inside_city"][row]), modeled_s,
                                                               scenario["spread_quantile"][row])
        if report is not None and engine_id is not None:         # every caller sends the chosen engine at once, so the dispatch is logged here
            dispatch_log.append((row, env.now, engine_id, engine_state.get(engine_id), travel_s.get(engine_id, np.nan), travel_seconds))
        return engine_id, travel_seconds

    def serve_waiting():
        # ponytail: waiting runs are offered engines only at clears; an engine entering service mid-wait waits for the next clear
        for item in sorted(waiting, key=lambda item: item[:2]):
            engine_id, travel_seconds = pick(item[2])
            if engine_id is not None:
                waiting.remove(item)                             # the request is granted now, so the next run sees the engine busy
                busy_runs[engine_id] = item[2]
                item[3].succeed((engine_id, travel_seconds, engine_pool[engine_id].request()))

    def engine_run(row):
        times = runs.loc[row, TIME_COLUMNS]
        start = times.dropna().min()
        if wait_rank[row] >= 0:
            yield env.timeout(start - env.now)
            yield env.timeout(0)                                 # runs clearing in this same second free their engines first
        elif start > env.now:                                    # an outage stand-in starts when its call does, so it looks for
            yield env.timeout(start - env.now)                   # its engine at once: a new process runs before this second's clears
        engine_id, travel_seconds = pick(row)
        request = None
        if engine_id is None:                                    # no engine to send: wait for a clear
            wake = env.event()
            item = (wait_rank[row], env.now, row, wake)
            waiting.append(item)
            cancelled = pd.isna(times["arrived_at"])             # its release time comes from the record
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
        busy_runs[engine_id] = row                               # recorded at the reservation, before other runs of this second look
        with (request or engine_pool[engine_id].request()) as engine:
            yield engine
            for column, at in recorded.items():
                yield env.timeout(at - env.now)
                simulated.at[row, column] = env.now
        busy_runs.pop(engine_id, None)
        enroute_s = simulated.at[row, "enroute_at"]
        if "arrived_at" not in recorded.index and travel_seconds and not pd.isna(enroute_s):
            # ponytail: the share of the way out assumes the engine left from home; an engine that left another scene is placed on the line from its station to the call
            outbound_share = min(1.0, max(0.0, (env.now - enroute_s) / travel_seconds))
            engine_state[engine_id] = (env.now, row, outbound_share) if outbound_share > 0 else (env.now, None)
        else:
            engine_state[engine_id] = (env.now, row if "arrived_at" in recorded.index else None)
        serve_waiting()

    def call_run(row, call_state, under_way):
        # one run of a call whose arrivals the call decides (scenario["fates"]), not this row of the record
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
            waiting.append((wait_rank[row], env.now, row, wake))
            granted = yield wake
            if granted is None:
                cancelled_while_waiting.append(row)
                simulated.loc[row, ["dispatched_at", "cleared_at"]] = [times["dispatched_at"], env.now]
                return
            engine_id, travel_seconds, request = granted
            waited_seconds[row] = env.now - times["dispatched_at"]
        turnout_s = times["enroute_at"] - times["dispatched_at"]
        simulated.at[row, "engine_id"] = engine_id
        busy_runs[engine_id] = row                               # recorded at the reservation, before other runs of this second look
        with (request or engine_pool[engine_id].request()) as engine:
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
        busy_runs.pop(engine_id, None)
        serve_waiting()

    def release_extra_runs(incident, call_state, under_way):
        # at each recorded cancellation moment the call releases one run: a run still waiting for an engine if there is
        # one, otherwise the engine under way that would arrive last; when no engine is under way, its next run is not sent
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
        if incident in fates:
            arrivals, release_moments, on_scene_s = fates[incident]
            call_state = {"arrivals": arrivals, "release_moments": release_moments, "on_scene_s": on_scene_s,
                          "arrived": 0, "releases_pending": 0}
            under_way = {}                                       # keyed by run: (the time its engine would arrive, its process)
            env.process(release_extra_runs(incident, call_state, under_way))
            for row in call_runs:
                env.process(call_run(row, call_state, under_way))
            return
        for row in call_runs:
            env.process(engine_run(row))

    for call_runs in runs.groupby("incident_id", sort=False).groups.values():
        env.process(call(list(call_runs)))
    env.run()
    show_table("simulated CAD times (seconds since day zero) and engine used", simulated, verbose)
    if report is not None:
        report.update(kept_recorded=kept_recorded, waited_seconds=waited_seconds, never_served=[item[2] for item in waiting],
                      cancelled_while_waiting=cancelled_while_waiting,
                      dispatches=pd.DataFrame(dispatch_log, columns=["run", "time_s", "engine_id", "last_clear", "modeled_s",
                                                                     "travel_s"]))
    return simulated


# ----------------------------------------------------------------------------------------------------------------------
# The world: every fixed input, fitted once
# ----------------------------------------------------------------------------------------------------------------------

def build_world(verbose=VERBOSE):
    """Load the record and fit every fixed input of the environment, in an order where each step uses earlier results.

    The steps are: the record and its repaired runs, the engines and the agency line, the road network and the travel
    model, the modeled travel from every station to every call, the drive home, the travel spread (all calls and brush
    fires), the quint penalty, and the fire-call map with its opportunity cost.

    Input
        verbose : whether every step prints its tables (bool)
    Output
        The world (dict) with these keys:
        dispatches, runs, engines, aid_protocols : the record, its repaired runs, the engines, the aid call types
        graph, call_nodes, home_nodes, seconds_from_node, node_position, network_seconds : the road network, each run's
            call node, each engine's station node, the shortest-path function, and the record's network seconds
        travel_model, dwell_s, slowdown : the travel model (it turns network seconds into travel seconds) and the
            drive-home parameters
        travel_spread, brush_spread : the fitted travel spreads for draw_travel_ratio
        quint_penalty_by_type, quint_indicator : the penalty per call type, and a Series indexed by engine id that is
            1.0 for a quint and 0.0 for any other engine
        readiness_cells, travel_to_cells, fire_hour_share, readiness_of, opportunity_cost : the fire-call map (cells,
            travel to them, share of fire calls by hour) and the functions for fire readiness and opportunity cost
        fits : the tables that check each fitted input (dict of DataFrames: travel_model, return_progress,
            travel_spread, travel_spread_check, brush_travel_spread, brush_travel_spread_check, quint_penalty)
        record : the 2025 record as a scenario (dict with the keys simulate reads, plus dispatches and large_fire)
    Example
        world = build_world(verbose=False)
        world["quint_penalty_by_type"] -> {'F56 elevator': -260.0, 'F82 brush fire': 260.0, 'all other calls': 10.0}
    """
    dispatches = load_inputs(verbose)
    runs = build_runs(dispatches, verbose)
    engines = load_engines(verbose)
    aid_protocols = load_aid_protocols(dispatches, verbose)
    graph = load_road_network(verbose)
    call_nodes, engine_nodes = snap_to_network(graph, dispatches, engines, verbose)
    seconds_from_node, node_position = make_seconds_from_node(graph)
    home_nodes = engine_nodes.set_index("engine_id")["node"].astype("int64")
    network_seconds = network_from_stations(dispatches, call_nodes, engine_nodes, seconds_from_node, node_position, verbose)
    travel_model, travel_model_check = fit_travel_model(dispatches, network_seconds, verbose)
    record_travel = apply_agency_boundaries(estimate_network_travel_seconds(dispatches, engines, network_seconds, travel_model, verbose),
                                            dispatches, engines, aid_protocols, verbose)
    dwell_s, slowdown, return_progress_check = fit_return_progress(dispatches, engines, call_nodes, home_nodes, seconds_from_node,
                                                                   node_position, travel_model, verbose)
    spread_quantile = pd.Series(np.random.default_rng(TRAVEL_SPREAD_SEED).random(len(runs)), index=runs.index)
    travel_spread, spread_table, spread_check = fit_travel_spread(dispatches, network_seconds, travel_model, spread_quantile,
                                                                  verbose=verbose)
    brush = dispatches["protocol"].eq(BRUSH_FIRE_PROTOCOL).fillna(False)
    brush_spread, brush_spread_table, brush_spread_check = fit_travel_spread(
        dispatches.loc[brush], network_seconds.loc[brush], travel_model, spread_quantile, BRUSH_SPREAD_BINS, verbose)
    call_type_for_quint = dispatches["protocol"].map(QUINT_RULE_CALL_TYPES).fillna("all other calls")
    quint_penalty_by_type, quint_penalty_fit = fit_quint_penalty(dispatches, runs, engines, record_travel, call_type_for_quint, verbose)
    readiness_cells, travel_to_cells, fire_hour_share = build_readiness_cells(dispatches, graph, engines, home_nodes, aid_protocols,
                                                                              seconds_from_node, node_position, travel_model, verbose)
    readiness_of, opportunity_cost = make_readiness_tools(readiness_cells, travel_to_cells, engines)
    world = {"dispatches": dispatches, "runs": runs, "engines": engines, "aid_protocols": aid_protocols, "graph": graph,
             "call_nodes": call_nodes, "home_nodes": home_nodes, "seconds_from_node": seconds_from_node,
             "node_position": node_position, "network_seconds": network_seconds, "travel_model": travel_model,
             "dwell_s": dwell_s, "slowdown": slowdown, "travel_spread": travel_spread, "brush_spread": brush_spread,
             "quint_penalty_by_type": quint_penalty_by_type,
             "quint_indicator": (engines.set_index("engine_id")["engine_type"] == "Q").astype(float),
             "readiness_cells": readiness_cells, "travel_to_cells": travel_to_cells, "fire_hour_share": fire_hour_share,
             "readiness_of": readiness_of, "opportunity_cost": opportunity_cost,
             "fits": {"travel_model": travel_model_check, "return_progress": return_progress_check,
                      "travel_spread": spread_table, "travel_spread_check": spread_check,
                      "brush_travel_spread": brush_spread_table, "brush_travel_spread_check": brush_spread_check,
                      "quint_penalty": quint_penalty_fit}}
    world["record"] = {"dispatches": dispatches, "runs": runs, "travel": record_travel, "call_nodes": call_nodes,
                       "spread_quantile": spread_quantile, "fates": call_fates(runs),
                       "large_fire": pd.Series(False, index=dispatches.index),
                       **run_attributes(dispatches, quint_penalty_by_type)}
    return world
