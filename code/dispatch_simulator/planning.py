"""Multi-stage planning of fire stations and engines under population growth (research_planning.md, section 4).

The planning study asks where and when to build or move a station and where to add an engine or a quint, stage by
stage under a fixed budget, while the population grows. This module holds everything the study needs besides training:
- planning inputs: 2020 census tracts with their 2010, 2020 and 2025 populations, the demand points of the training
  months with their call rates and engine workload, the current stations, and the BEBR county paths;
- growth draws: tract population multipliers for 2030 to 2045 from a county path and a spatial mixture;
- a fast evaluator: an approximate hypercube model of the protocol that turns a configuration and a demand into the
  expected travel of the first engine to arrive, its tract means, the regional mean M and the inequality D;
- candidate sites, screened by the stage cost;
- the planning environment (a gymnasium Env for MaskablePPO), the baselines and the validation draws;
- the bridge to the dispatch simulator, which verifies a configuration on generated incident streams.
Run it to print every step on concrete cases: python code/dispatch_simulator/planning.py
"""
import copy

import geopandas as gpd
import gymnasium as gym
import numpy as np
import pandas as pd
import pyproj

import environment
import scenarios
from environment import DAY_ZERO, PROJECT_ROOT, PROJECTED_CRS, READINESS_CELL_M, STUDY_FIRE_PROTOCOLS, TRAINING_MONTHS_END, VERBOSE, show_table

STAGE_YEARS = (2030, 2035, 2040, 2045)                      # [DESIGN] evaluation years; the decisions come in 2025, 2030, 2035 and 2040
PATH_WEIGHTS = {"Low": 0.25, "Medium": 0.5, "High": 0.25}   # [DESIGN] researcher-selected sampling weights of the BEBR county paths
COUNTY_2025 = 298_485                                       # [SOURCE] BEBR 2026 estimate of Alachua County's population in 2025
FLEET_DATE = pd.Timestamp("2025-03-31")                     # [DESIGN] the starting fleet: vehicles in service on the last day of the training months
MAX_NEW_STATIONS = 4                                        # [DESIGN] one new station per stage at most
RELOCATION_KM = 5.0                                         # [DESIGN] straight-line allowance for moving a station
CANDIDATE_SITES = 20                                        # [DESIGN] frozen candidate sites, the documented Station 9 site included
CANDIDATE_SPACING_KM = 1.0                                  # [DESIGN] candidate sites are at least this far apart
KAPPA = 10.0                                                # [DESIGN] reward penalty per minute of regional mean above its 2025 level (frozen after validation)
TARGET_TOLERANCE_MIN = 0.05                                 # [DESIGN] a year violates the mean target when M exceeds M_2025 by more than this (half the 0.1 min the study interprets)
RANK_DEPTH = 6                                              # [DESIGN] vehicles considered per demand point; against depth 8 and 200 iterations the outcome changes by under 1e-5 min
BUSY_CAP = 0.95                                             # [DESIGN] upper bound of a vehicle's busy probability in the fixed point
FIXED_POINT_ITERATIONS = 15                                 # [DESIGN] iterations of the busy-probability fixed point (damping 0.5); see RANK_DEPTH for the accuracy check
STATION9_NODE = 84821948                                    # [SOURCE] road node on SW 47th Way nearest Archer Road: the Southwest Public Safety Center (groundbreaking venue 4320 SW 47th Way), approximate
STATION9 = "GFR_09"                                         # [SOURCE] the station the city moves there (late 2027)
CENSUS_2010 = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "census2010_blocks_alachua.csv"   # [SOURCE] 2010 blocks: residents and internal points
CENSUS_2020 = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "census2020_blocks_alachua.gpkg"  # [SOURCE] 2020 blocks with geometry
BEBR_FILE = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "bebr_alachua_projections_2026.csv"  # [SOURCE] BEBR 2026 county projections
WINDOW_HOURS = (TRAINING_MONTHS_END - DAY_ZERO).total_seconds() / 3600   # hours of the training months, the demand base
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "planning"                # [DESIGN] the frozen setup, models and training signals of the planning study
SETUP_FILE = MODEL_DIR / "setup.json"                                    # [DESIGN] calibration scales and candidate sites, written once by the evaluator validation


# ----------------------------------------------------------------------------------------------------------------------
# Planning inputs
# ----------------------------------------------------------------------------------------------------------------------

def load_tracts(verbose=VERBOSE):
    """Build the 2020 census tracts with their populations and the two spatial growth patterns.

    Input
        verbose : whether to print the table (bool)
    Output
        A GeoDataFrame, one row per tract (PROJECTED_CRS): tract (11-digit GEOID), pop2010 (2010 block residents whose
        internal point lies in the tract), pop2020, pop2025 (pop2020 scaled to COUNTY_2025, an estimate), trend_share
        (the tract's share of positive 2010-2020 growth) and proportional_share (its share of 2020 residents).
    Example
        load_tracts(False)["pop2025"].sum() -> 298485.0
    """
    blocks = gpd.read_file(CENSUS_2020)
    blocks["tract"] = blocks["GEOID"].str[:11]
    tracts = blocks[["tract", "POP100", "geometry"]].dissolve(by="tract", aggfunc="sum").rename(columns={"POP100": "pop2020"})
    tracts = tracts.to_crs(PROJECTED_CRS).reset_index()
    old = pd.read_csv(CENSUS_2010, dtype={"GEOID": str})
    points = gpd.GeoDataFrame(old[["POP100"]], geometry=gpd.points_from_xy(old["INTPTLON"], old["INTPTLAT"]), crs="EPSG:4326").to_crs(PROJECTED_CRS)
    joined = gpd.sjoin(points, tracts[["tract", "geometry"]], predicate="within", how="left")
    tracts["pop2010"] = tracts["tract"].map(joined.groupby("tract")["POP100"].sum()).fillna(0.0)
    tracts["pop2025"] = tracts["pop2020"] * COUNTY_2025 / tracts["pop2020"].sum()
    growth = (tracts["pop2020"] - tracts["pop2010"]).clip(lower=0)
    tracts["trend_share"] = growth / growth.sum()
    tracts["proportional_share"] = tracts["pop2020"] / tracts["pop2020"].sum()
    show_table("2020 census tracts with 2010, 2020 and 2025 residents", tracts.drop(columns="geometry").round(4), verbose)
    if verbose:
        print(f"tracts {len(tracts)}; residents 2010 {tracts['pop2010'].sum():,.0f} (2010 blocks {old['POP100'].sum():,}), "
              f"2020 {tracts['pop2020'].sum():,.0f}, 2025 {tracts['pop2025'].sum():,.0f}; tracts that lost residents {int((tracts['pop2020'] < tracts['pop2010']).sum())}\n")
    return tracts


def build_demand_points(world, generator, tracts, verbose=VERBOSE):
    """Aggregate the training months' incidents into demand points for the fast evaluator.

    A demand point gathers the incidents at one road node that share an agency-eligibility group (GFR jurisdiction or
    not, automatic aid or not) and a quint penalty. Rates count incidents per hour over the training months.

    Input
        world     : the dict from environment.build_world
        generator : the dict from scenarios.fit_generator (its pool holds the incidents of October to March with a location)
        tracts    : the tracts from load_tracts
        verbose   : whether to print the table and checks (bool)
    Output
        A DataFrame, one row per demand point: node, x and y (metres), gfr (the call is in GFR's jurisdiction), aid (its
        call type gets automatic aid), penalty_min (the quint penalty in minutes), tract (row of tracts), fire_rate and
        other_rate (arriving incidents per hour: study fires and other calls), fire_load and other_load (engine-hours
        per hour of all incidents, arriving or not).
    Example
        build_demand_points(world, generator, tracts, False)[["fire_rate", "other_rate"]].sum().sum() * WINDOW_HOURS
        -> the number of arriving incidents of October to March
    """
    pool, record, runs = generator["pool"], world["record"], world["record"]["runs"]
    first_row = pool["run_rows"].map(lambda rows: int(rows[0]))
    busy_h = ((runs["cleared_at"] - runs["dispatched_at"]) / 3600).clip(lower=0).fillna(0.0)
    hours = pool["run_rows"].map(lambda rows: float(busy_h.loc[rows].sum()))
    located = gpd.GeoDataFrame(index=pool.index, geometry=gpd.points_from_xy(pool["x"], pool["y"]), crs=PROJECTED_CRS)
    tract_of = gpd.sjoin(located, tracts[["geometry"]], predicate="within", how="left")["index_right"]
    tract_of = tract_of[~tract_of.index.duplicated()].reindex(pool.index)
    incidents = pd.DataFrame({"node": record["call_nodes"].loc[first_row, "node"].to_numpy(), "x": pool["x"], "y": pool["y"],
                              "gfr": (pool["jurisdiction"] == "GFR").to_numpy(), "aid": pool["protocol"].isin(world["aid_protocols"]).to_numpy(),
                              "penalty_min": record["quint_penalty"].loc[first_row].to_numpy() / 60, "tract": tract_of.to_numpy(),
                              "fire": pool["fire"].to_numpy(), "arrived": (pool["arrival_row"] >= 0).to_numpy(), "hours": hours.to_numpy()},
                             index=pool.index)
    dropped = int(incidents["node"].isna().sum() + incidents["tract"].isna().sum())
    incidents = incidents.dropna(subset=["node", "tract"]).astype({"node": "int64", "tract": "int64"})
    incidents["fire_n"] = (incidents["fire"] & incidents["arrived"]).astype(float)
    incidents["other_n"] = (~incidents["fire"] & incidents["arrived"]).astype(float)
    incidents["fire_h"] = incidents["hours"].where(incidents["fire"], 0.0)
    incidents["other_h"] = incidents["hours"].where(~incidents["fire"], 0.0)
    points = incidents.groupby(["node", "gfr", "aid", "penalty_min"], sort=False).agg(
        x=("x", "mean"), y=("y", "mean"), tract=("tract", lambda t: int(t.mode().iloc[0])), fire_n=("fire_n", "sum"),
        other_n=("other_n", "sum"), fire_h=("fire_h", "sum"), other_h=("other_h", "sum")).reset_index()
    for kind in ["fire", "other"]:
        points[f"{kind}_rate"] = points.pop(f"{kind}_n") / WINDOW_HOURS
        points[f"{kind}_load"] = points.pop(f"{kind}_h") / WINDOW_HOURS
    show_table("demand points of October to March", points.round(5), verbose)
    if verbose:
        print(f"incidents {len(pool):,} -> demand points {len(points):,} ({dropped} incidents without a road node or tract left out); "
              f"arriving incidents {points[['fire_rate', 'other_rate']].sum().sum() * WINDOW_HOURS:,.0f}; "
              f"engine-hours {points[['fire_load', 'other_load']].sum().sum() * WINDOW_HOURS:,.0f} "
              f"(mean busy fraction of 28 vehicles {points[['fire_load', 'other_load']].sum().sum() / 28:.3f})\n")
    return points


def current_stations(world, verbose=VERBOSE):
    """List the stations and vehicles in service on FLEET_DATE: the starting configuration.

    Input
        world   : the dict from environment.build_world
        verbose : whether to print the table (bool)
    Output
        A DataFrame, one row per station (a vehicle without a station id is its own station): station, node (road
        node), gfr (GFR station), x and y (metres), engines (E), quints (Q) and vehicle_ids (list of str).
    Example
        current_stations(world, False)[["engines", "quints"]].sum().tolist() -> [23, 5]
    """
    engines = world["engines"]
    fleet = engines[(engines["in_service_from"] <= FLEET_DATE) & (engines["in_service_to"] > FLEET_DATE)].copy()
    fleet["station"] = fleet["home_station"].fillna(fleet["engine_id"])
    fleet["node"] = fleet["engine_id"].map(world["home_nodes"])
    stations = fleet.groupby("station", sort=False).agg(node=("node", "first"), gfr=("engine_agency", lambda a: bool((a == "GFR").all())),
                                                        engines=("engine_type", lambda t: int((t == "E").sum())),
                                                        quints=("engine_type", lambda t: int((t == "Q").sum())),
                                                        vehicle_ids=("engine_id", list)).reset_index()
    x, y = node_xy(world, stations["node"])
    stations["x"], stations["y"] = x, y
    show_table("stations and vehicles in service on the last day of the training months", stations, verbose)
    return stations


def node_xy(world, nodes):
    """Return the projected coordinates (metres, PROJECTED_CRS) of road nodes, as two arrays."""
    graph = world["graph"]
    lon = np.array([graph.nodes[int(node)]["x"] for node in nodes])
    lat = np.array([graph.nodes[int(node)]["y"] for node in nodes])
    x, y = pyproj.Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def candidate_pool(world, generator):
    """Return the candidate-site pool: for each 1 km cell with residents, the road node of its most populous block.

    Output
        A DataFrame with node, x, y (metres) and gfr (the block lies in GFR's jurisdiction), one row per cell.
    """
    locations = generator["locations"]
    top = locations.sort_values("residents", ascending=False).drop_duplicates(["cell_x", "cell_y"])
    pool = pd.DataFrame({"node": top["node"].astype("int64").to_numpy(), "gfr": (top["jurisdiction"] == "GFR").to_numpy()})
    pool = pool.drop_duplicates("node").reset_index(drop=True)
    pool["x"], pool["y"] = node_xy(world, pool["node"])
    return pool


def load_county_paths():
    """Return the BEBR county population of each path for STAGE_YEARS (dict: path -> array of 4 floats)."""
    bebr = pd.read_csv(BEBR_FILE)
    return {path: np.array([float(bebr.loc[(bebr["series"] == path) & (bebr["year"] == year), "population"].iloc[0]) for year in STAGE_YEARS])
            for path in PATH_WEIGHTS}


def build_planning_inputs(world, generator, verbose=VERBOSE):
    """Assemble the planning inputs and the site list of the fast evaluator.

    Sites are the current stations' nodes, then the documented Station 9 site, then the candidate pool.

    Output
        A dict: tracts, points, stations, county_paths, sites (DataFrame: node, x, y, gfr, kind = "station",
        "station 9 plan" or "pool"), and the travel model and network access needed to price any node.
    Example
        inputs = build_planning_inputs(world, generator)
    """
    tracts = load_tracts(verbose)
    points = build_demand_points(world, generator, tracts, verbose)
    stations = current_stations(world, verbose)
    pool = candidate_pool(world, generator)
    x9, y9 = node_xy(world, [STATION9_NODE])
    plan_site = pd.DataFrame({"node": [STATION9_NODE], "gfr": [True], "x": x9, "y": y9})
    sites = pd.concat([stations[["node", "gfr", "x", "y"]].assign(kind="station"), plan_site.assign(kind="station 9 plan"),
                       pool[~pool["node"].isin(stations["node"]) & (pool["node"] != STATION9_NODE)].assign(kind="pool")], ignore_index=True)
    if verbose:
        print(f"sites: {int((sites['kind'] == 'station').sum())} current stations, 1 documented Station 9 site, "
              f"{int((sites['kind'] == 'pool').sum())} pool nodes (one per 1 km cell with residents)\n")
    inputs = {"tracts": tracts, "points": points, "stations": stations, "county_paths": load_county_paths(), "sites": sites}
    inputs["blocks"] = residential_blocks(generator, inputs)
    inputs["source_mix"] = source_mix(world, generator, verbose)
    return inputs


def residential_blocks(generator, inputs):
    """The generator's residential blocks, where added calls go: node, gfr, tract and residents (one row per block)."""
    locations = generator["locations"]
    return pd.DataFrame({"node": locations["node"].astype("int64").to_numpy(), "gfr": (locations["jurisdiction"] == "GFR").to_numpy(),
                         "tract": block_tracts(generator, inputs), "residents": locations["residents"].to_numpy(dtype=float)})


def source_mix(world, generator, verbose=VERBOSE):
    """The mixture of the other calls that added calls copy (the generator draws their sources countywide).

    Output
        A DataFrame, one row per (aid, penalty_min) group: share (of all other incidents), arrive (the share of them
        with an arrival) and hours (mean engine-hours per incident, cancelled ones included).
    """
    pool, record, runs = generator["pool"], world["record"], world["record"]["runs"]
    other = pool[~pool["fire"]]
    first_row = other["run_rows"].map(lambda rows: int(rows[0]))
    busy_h = ((runs["cleared_at"] - runs["dispatched_at"]) / 3600).clip(lower=0).fillna(0.0)
    table = pd.DataFrame({"aid": other["protocol"].isin(world["aid_protocols"]).to_numpy(),
                          "penalty_min": record["quint_penalty"].loc[first_row].to_numpy() / 60,
                          "arrive": (other["arrival_row"] >= 0).to_numpy(dtype=float),
                          "hours": other["run_rows"].map(lambda rows: float(busy_h.loc[rows].sum())).to_numpy()})
    mix = table.groupby(["aid", "penalty_min"]).agg(share=("arrive", "size"), arrive=("arrive", "mean"), hours=("hours", "mean")).reset_index()
    mix["share"] = mix["share"] / mix["share"].sum()
    show_table("other calls that added calls copy: mixture by automatic aid and quint penalty", mix.round(4), verbose)
    return mix


def growth_multipliers(inputs, path, w):
    """Tract population multipliers (relative to 2025) for 2025 and STAGE_YEARS under one growth draw.

    A growing path allocates the county change by w x trend_share + (1 - w) x proportional_share; a declining path
    scales every tract in proportion. A tract without residents keeps a multiplier of 1.

    Input
        inputs : the dict from build_planning_inputs; path : "Low", "Medium" or "High"; w : the trend weight (0 to 1)
    Output
        An array of shape (5, tracts): row 0 is 2025 (all ones), rows 1-4 are 2030, 2035, 2040, 2045.
    Example
        growth_multipliers(inputs, "High", 1.0)[-1].max() -> the largest tract multiplier in 2045 on the high path
    """
    tracts = inputs["tracts"]
    base = tracts["pop2025"].to_numpy()
    shares = w * tracts["trend_share"].to_numpy() + (1 - w) * tracts["proportional_share"].to_numpy()
    rows = [np.ones(len(base))]
    for total in inputs["county_paths"][path]:
        change = total - COUNTY_2025
        population = base + change * shares if change >= 0 else base * total / COUNTY_2025
        rows.append(np.where(base > 0, population / np.where(base > 0, base, 1.0), 1.0))
    return np.vstack(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Configurations and the fast evaluator
# ----------------------------------------------------------------------------------------------------------------------

def start_configuration(inputs):
    """Return the starting configuration: the current stations in their slots, MAX_NEW_STATIONS empty slots after them.

    Output
        A dict of arrays over station slots: site (row of inputs["sites"], -1 when empty), gfr, engines, quints; and name
        (list of str).
    """
    stations = inputs["stations"]
    slots = len(stations) + MAX_NEW_STATIONS
    config = {"site": np.full(slots, -1), "gfr": np.zeros(slots, dtype=bool), "engines": np.zeros(slots, dtype=int),
              "quints": np.zeros(slots, dtype=int), "name": list(stations["station"]) + [f"NEW_{i + 1}" for i in range(MAX_NEW_STATIONS)]}
    n = len(stations)
    config["site"][:n] = np.arange(n)
    config["gfr"][:n] = stations["gfr"].to_numpy()
    config["engines"][:n] = stations["engines"].to_numpy()
    config["quints"][:n] = stations["quints"].to_numpy()
    return config


def vehicles(config):
    """Return the vehicles of a configuration as arrays: site, gfr, quint, slot (one entry per vehicle)."""
    slot = np.concatenate([np.repeat(np.arange(len(config["site"])), config["engines"]),
                           np.repeat(np.arange(len(config["site"])), config["quints"])])
    quint = np.concatenate([np.zeros(config["engines"].sum(), dtype=bool), np.ones(config["quints"].sum(), dtype=bool)])
    return config["site"][slot], config["gfr"][slot], quint, slot


class FastEvaluator:
    """Approximate hypercube model of the protocol on the demand points.

    Each demand point ranks the eligible vehicles by modeled travel plus the quint penalty, as the protocol does. Vehicle
    v is busy with probability p_v; the first free vehicle in rank order responds, and when all are busy the nearest
    eligible vehicle is assumed to respond after it frees. The busy probabilities solve a fixed point: a vehicle's load
    is the engine-hours per hour of the incidents it responds to first, times a calibration scale. Growth adds other
    calls at the generator's residential blocks (by residents within the tract, with the countywide mix of other calls),
    as scenarios.add_demand_growth does; a declining path thins the historical other calls.

    Input (constructor)
        world        : the dict from environment.build_world (network and travel model)
        inputs       : the dict from build_planning_inputs
        scale        : the calibration factor of the loads (float, 1.0 before calibration)
        travel_scale : the calibration factor from modeled to simulated travel of other calls (float, 1.0 before
                       calibration): the simulator draws each run's travel around the modeled time, so its mean differs
        fire_travel_scale : the same factor for study fire calls (float; travel_scale when None), since brush fires draw
                       their travel from their own spread
    Output
        evaluate(config, multipliers) returns a dict, travel in minutes of the first-arriving engine or quint:
        M_F (fire-weighted regional mean over study fire calls), D_F (standard deviation of the tract means of study fire
        calls, weighted by each tract's fixed fire rate), tract_fire_mean, M_O (regional mean over the other calls), and
        for reference M, D, D_fixed and tract_mean over all calls; busy (array over vehicles); waiting_fire and waiting
        (shares of fire calls and of all calls that find every eligible vehicle busy); outcome (array over demand points:
        expected travel for other calls; the base points first, then the residential-block growth points).
    Example
        evaluator = FastEvaluator(world, inputs); evaluator.evaluate(start_configuration(inputs), np.ones(n_tracts))["M"]
    """

    def __init__(self, world, inputs, scale=1.0, travel_scale=1.0, fire_travel_scale=None):
        points = inputs["points"]
        positions = world["node_position"][points["node"]].to_numpy()
        seconds = np.vstack([world["seconds_from_node"](int(node))[positions] for node in inputs["sites"]["node"]])
        travel = world["travel_model"](seconds) / 60
        gfr_point, aid = points["gfr"].to_numpy(), points["aid"].to_numpy()
        self.travel_gfr = np.where(gfr_point | aid, travel, np.inf)                 # sites x points, minutes, for GFR vehicles
        self.travel_county = np.where(~gfr_point | aid, travel, np.inf)             # for the other agency's vehicles
        self.penalty = points["penalty_min"].to_numpy()
        self.tract = points["tract"].to_numpy()
        self.tracts = len(inputs["tracts"])
        self.rates = (points["fire_rate"].to_numpy(), points["other_rate"].to_numpy())
        self.loads = (points["fire_load"].to_numpy(), points["other_load"].to_numpy())
        self.base_tract_rate = np.bincount(self.tract, weights=self.rates[0] + self.rates[1], minlength=self.tracts)
        self.scale, self.travel_scale = scale, travel_scale
        self.fire_travel_scale = travel_scale if fire_travel_scale is None else fire_travel_scale
        self.fire_tract_rate = np.bincount(self.tract, weights=self.rates[0], minlength=self.tracts)   # fixed: fire demand does not grow
        # added other calls go to residential blocks (by residents within the tract) and copy the countywide source mix
        blocks, mix = inputs["blocks"], inputs["source_mix"]
        nodes = blocks["node"].unique()
        block_seconds = np.vstack([world["seconds_from_node"](int(node))[world["node_position"][nodes].to_numpy()] for node in inputs["sites"]["node"]])
        block_travel = pd.DataFrame(world["travel_model"](block_seconds) / 60, columns=nodes)
        tract_other = np.bincount(self.tract, weights=self.rates[1], minlength=self.tracts)
        tract_residents = blocks.groupby("tract")["residents"].sum().reindex(range(self.tracts), fill_value=0.0).to_numpy()
        arrive_overall = float((mix["share"] * mix["arrive"]).sum())
        columns, gfr_rows, aid_rows, penalty_rows, tract_rows, arrive_rows, load_rows = [], [], [], [], [], [], []
        for group in mix.itertuples():
            for block in blocks.itertuples():
                if block.residents <= 0 or tract_residents[block.tract] <= 0:
                    continue
                base = tract_other[block.tract] * block.residents / tract_residents[block.tract] / arrive_overall
                columns.append(block.node), gfr_rows.append(block.gfr), aid_rows.append(group.aid), penalty_rows.append(group.penalty_min)
                tract_rows.append(block.tract), arrive_rows.append(base * group.share * group.arrive), load_rows.append(base * group.share * group.hours)
        travel_added = block_travel[columns].to_numpy()
        gfr_added, aid_added = np.array(gfr_rows), np.array(aid_rows)
        self.travel_gfr = np.hstack([self.travel_gfr, np.where(gfr_added | aid_added, travel_added, np.inf)])
        self.travel_county = np.hstack([self.travel_county, np.where(~gfr_added | aid_added, travel_added, np.inf)])
        self.penalty = np.concatenate([self.penalty, penalty_rows])
        self.tract = np.concatenate([self.tract, tract_rows]).astype(int)
        self.n_base = len(points)
        self.added = (np.array(arrive_rows), np.array(load_rows))         # arriving rate and load per unit of tract growth

    def evaluate(self, config, multipliers):
        site, gfr, quint, _ = vehicles(config)
        m = multipliers[self.tract]
        shrink, grow = np.minimum(m[:self.n_base], 1.0), np.maximum(m[self.n_base:] - 1.0, 0.0)
        rate = np.concatenate([self.rates[0] + self.rates[1] * shrink, self.added[0] * grow])
        load = np.concatenate([self.loads[0] + self.loads[1] * shrink, self.added[1] * grow])
        travel = np.where(gfr[:, None], self.travel_gfr[site], self.travel_county[site])
        rank = travel + np.where(quint[:, None], self.penalty[None, :], 0.0)
        depth = min(RANK_DEPTH, len(site))
        order = np.argpartition(rank, depth - 1, axis=0)[:depth]
        order = np.take_along_axis(order, np.argsort(np.take_along_axis(rank, order, axis=0), axis=0), axis=0)
        ranked = np.take_along_axis(travel, order, axis=0)
        valid = np.isfinite(ranked)
        p = np.full(len(site), 0.1)
        for _ in range(FIXED_POINT_ITERATIONS):
            q = np.where(valid, p[order], 1.0)
            free_before = np.cumprod(np.vstack([np.ones((1, q.shape[1])), q[:-1]]), axis=0)
            first = free_before * (1 - q)
            all_busy = free_before[-1] * q[-1]
            weight = first * load[None, :]
            weight[0] += all_busy * load
            p = 0.5 * p + 0.5 * np.minimum(BUSY_CAP, self.scale * np.bincount(order.ravel(), weights=weight.ravel(), minlength=len(site)))
        nearest = travel.min(axis=0)                                              # all busy: the nearest eligible vehicle (no quint penalty)
        modeled = (first * np.where(valid, ranked, 0.0)).sum(axis=0) + all_busy * nearest
        fire_rate = np.concatenate([self.rates[0], np.zeros(len(rate) - self.n_base)])
        other_rate = rate - fire_rate
        fire_out, outcome = self.fire_travel_scale * modeled, self.travel_scale * modeled
        with np.errstate(invalid="ignore", divide="ignore"):
            tract_fire_mean = np.bincount(self.tract, weights=fire_rate * fire_out, minlength=self.tracts) / self.fire_tract_rate
            tract_rate = np.bincount(self.tract, weights=rate, minlength=self.tracts)
            mixed = fire_rate * fire_out + other_rate * outcome
            tract_mean = np.bincount(self.tract, weights=mixed, minlength=self.tracts) / tract_rate
        return {"M_F": float((fire_rate * fire_out).sum() / fire_rate.sum()), "D_F": weighted_sd(tract_fire_mean, self.fire_tract_rate),
                "tract_fire_mean": tract_fire_mean, "M_O": float((other_rate * outcome).sum() / other_rate.sum()),
                "M": float(mixed.sum() / rate.sum()), "D": weighted_sd(tract_mean, tract_rate), "D_fixed": weighted_sd(tract_mean, self.base_tract_rate),
                "tract_mean": tract_mean, "busy": p, "waiting_fire": float((fire_rate * all_busy).sum() / fire_rate.sum()),
                "waiting": float((rate * all_busy).sum() / rate.sum()), "outcome": outcome}


def slot_busy(config, busy):
    """Mean busy probability of each station slot's vehicles (array over slots; 0 for an empty slot), from an evaluation
    of that configuration."""
    _, _, _, vehicle_slot = vehicles(config)
    return np.bincount(vehicle_slot, weights=busy, minlength=len(config["site"])) / np.maximum(config["engines"] + config["quints"], 1)


def weighted_sd(values, weights):
    """Weighted standard deviation over the entries with positive weight (float)."""
    use = weights > 0
    mean = np.average(values[use], weights=weights[use])
    return float(np.sqrt(np.average((values[use] - mean) ** 2, weights=weights[use])))


def stage_cost(result, targets):
    """The planning cost of one evaluated year, in minutes: fire inequality D_F plus KAPPA times the excess of the fire
    mean M_F and of the other-call mean M_O over their 2025 levels (targets: dict with M_F and M_O)."""
    return result["D_F"] + KAPPA * (max(0.0, result["M_F"] - targets["M_F"]) + max(0.0, result["M_O"] - targets["M_O"]))


def violations(result, targets):
    """(fire, other) booleans: the year's mean exceeds its 2025 level by more than TARGET_TOLERANCE_MIN."""
    return (result["M_F"] > targets["M_F"] + TARGET_TOLERANCE_MIN, result["M_O"] > targets["M_O"] + TARGET_TOLERANCE_MIN)


def targets_of(evaluator, inputs):
    """The 2025 levels of the two means: the current configuration at 2025 demand (dict with M_F and M_O)."""
    now = evaluator.evaluate(start_configuration(inputs), np.ones(evaluator.tracts))
    return {"M_F": now["M_F"], "M_O": now["M_O"]}


def move_station(config, slot, site, inputs):
    """Return a copy of the configuration with a station moved to a site (agency and vehicles unchanged), or built there
    with one engine when the slot is empty."""
    config = copy.deepcopy(config)
    if config["site"][slot] < 0:
        config["gfr"][slot] = bool(inputs["sites"]["gfr"].iloc[site])
        config["engines"][slot] = 1
    config["site"][slot] = site
    return config


def add_vehicle(config, slot, quint):
    """Return a copy of the configuration with one more engine (quint=False) or quint at a station slot."""
    config = copy.deepcopy(config)
    config["quints" if quint else "engines"][slot] += 1
    return config


# ----------------------------------------------------------------------------------------------------------------------
# Candidate sites
# ----------------------------------------------------------------------------------------------------------------------

def screen_candidates(inputs, evaluator, verbose=VERBOSE):
    """Pick the frozen candidate sites: the documented Station 9 site and the best-scoring pool nodes.

    A pool node's score is the largest reduction of the stage cost at 2045 on the medium path, under w = 0 and w = 1,
    over building a station there and moving any current station within RELOCATION_KM to it. Nodes are taken in score
    order at least CANDIDATE_SPACING_KM from the documented site and from each other.

    Output
        An array of CANDIDATE_SITES site rows (the documented site first) and the scores table (DataFrame).
    """
    sites, start = inputs["sites"], start_configuration(inputs)
    target = targets_of(evaluator, inputs)
    current = np.flatnonzero(start["site"] >= 0)
    rows = []
    for w in (0.0, 1.0):
        demand = growth_multipliers(inputs, "Medium", w)[-1]
        base = stage_cost(evaluator.evaluate(start, demand), target)
        for site in np.flatnonzero(sites["kind"] != "station"):
            options = [("build", -1, stage_cost(evaluator.evaluate(move_station(start, len(inputs["stations"]), site, inputs), demand), target))]
            near = current[np.hypot(sites["x"].iloc[start["site"][current]] - sites["x"].iloc[site],
                                    sites["y"].iloc[start["site"][current]] - sites["y"].iloc[site]).to_numpy() <= RELOCATION_KM * 1000]
            options += [("move", int(slot), stage_cost(evaluator.evaluate(move_station(start, slot, site, inputs), demand), target)) for slot in near]
            kind, slot, cost = min(options, key=lambda option: option[2])
            rows.append({"site": site, "w": w, "best action": kind, "slot": slot, "reduction": base - cost})
    scores = pd.DataFrame(rows).sort_values("reduction", ascending=False).drop_duplicates("site").reset_index(drop=True)
    chosen = [int(np.flatnonzero(sites["kind"] == "station 9 plan")[0])]
    for site in scores["site"]:
        if len(chosen) == CANDIDATE_SITES:
            break
        gaps = np.hypot(sites["x"].iloc[chosen] - sites["x"].iloc[site], sites["y"].iloc[chosen] - sites["y"].iloc[site])
        if (gaps >= CANDIDATE_SPACING_KM * 1000).all():
            chosen.append(int(site))
    if verbose:
        show_table("candidate screening (best action per pool node, largest reductions first)", scores.round(4), verbose)
        print(f"candidate sites (rows of sites): {chosen}\n")
    return np.array(chosen), scores


def prepare(world, generator, verbose=False):
    """Load the frozen planning setup: inputs, the calibrated evaluator and the candidate sites (from SETUP_FILE).

    Output
        (inputs, evaluator, candidates). SETUP_FILE is written by .aris/traces/planning_rl/validate_evaluator.py.
    Example
        inputs, evaluator, candidates = prepare(world, generator)
    """
    import json
    setup = json.loads(SETUP_FILE.read_text())
    inputs = build_planning_inputs(world, generator, verbose)
    evaluator = FastEvaluator(world, inputs, setup["scale"], setup["travel_scale"], setup["fire_travel_scale"])
    return inputs, evaluator, np.array(setup["candidates"])


# ----------------------------------------------------------------------------------------------------------------------
# The planning environment
# ----------------------------------------------------------------------------------------------------------------------

class PlanningEnv(gym.Env):
    """Four five-year stages; in each, one station action and then one vehicle action, then the next year is evaluated.

    Actions (one discrete space with masks): 0 no-op; 1..K build at candidate k; then (slot, candidate) moves; then
    add an engine at a slot; then add a quint at a slot. The reward after each vehicle action is -stage_cost of the
    evaluated year (fire inequality plus the penalised excess of the fire and other-call means over 2025); station
    actions get reward 0. Observation per tract: observed population multiplier, fire mean and all-call mean at the last
    evaluation, 2025 fire share and 2025 call share. The episode's growth draw comes from options["draw"] = (path, w) or
    is sampled with PATH_WEIGHTS and w uniform.

    Input (constructor)
        inputs, evaluator : from build_planning_inputs and FastEvaluator
        candidates        : the candidate site rows (array from screen_candidates)
        seed              : seed of the growth draws (int)
    Output
        A gymnasium environment; info at the end of an episode holds unshaped_return, decisions, deviations (actions
        other than no-op) and history (one dict per evaluated year: year, M_F, D_F, M_O, M, D, D_fixed, waiting,
        waiting_fire, violation_fire, violation_other, violation (either), actions, tract_fire_mean, config).
    Example
        env = PlanningEnv(inputs, evaluator, candidates); observation, info = env.reset(options={"draw": ("High", 1.0)})
    """

    def __init__(self, inputs, evaluator, candidates, seed=0):
        super().__init__()
        self.inputs, self.evaluator, self.candidates = inputs, evaluator, np.asarray(candidates)
        self.start = start_configuration(inputs)
        self.slots, self.k = len(self.start["site"]), len(self.candidates)
        self.targets = targets_of(evaluator, inputs)
        sites = inputs["sites"]
        self.center = np.array([sites["x"].mean(), sites["y"].mean()])
        self.site_xy = sites[["x", "y"]].to_numpy()
        self.share = evaluator.base_tract_rate / evaluator.base_tract_rate.sum()
        self.fire_share = evaluator.fire_tract_rate / evaluator.fire_tract_rate.sum()
        self.n_build, self.n_move = self.k, self.slots * self.k
        self.action_space = gym.spaces.Discrete(1 + self.n_build + self.n_move + 2 * self.slots)
        size = self.slots * 7 + self.k * 4 + evaluator.tracts * 5 + 6
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.first_result = evaluator.evaluate(self.start, np.ones(evaluator.tracts))

    def decode(self, action):
        """Return (kind, slot, candidate) of an action: kind is noop, build, move, engine or quint."""
        if action == 0:
            return "noop", -1, -1
        if action <= self.n_build:
            return "build", -1, action - 1
        if action <= self.n_build + self.n_move:
            slot, k = divmod(action - 1 - self.n_build, self.k)
            return "move", slot, k
        index = action - 1 - self.n_build - self.n_move
        return ("engine", index, -1) if index < self.slots else ("quint", index - self.slots, -1)

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        active = self.config["site"] >= 0
        if self.sub == 0:
            used = set(self.config["site"][active].tolist())
            free = np.array([site not in used for site in self.candidates])
            if (~active).any():
                mask[1:1 + self.k] = free
            for slot in np.flatnonzero(active):
                gaps = np.hypot(*(self.site_xy[self.candidates] - self.site_xy[self.config["site"][slot]]).T)
                start = 1 + self.n_build + slot * self.k
                mask[start:start + self.k] = free & (gaps <= RELOCATION_KM * 1000)
        else:
            base = 1 + self.n_build + self.n_move
            mask[base:base + self.slots] = active
            mask[base + self.slots:base + 2 * self.slots] = active
        return mask

    def apply(self, config, action):
        """Return the configuration after an action (the masks decide which actions are allowed)."""
        kind, slot, k = self.decode(action)
        if kind == "build":
            return move_station(config, int(np.flatnonzero(config["site"] < 0)[0]), int(self.candidates[k]), self.inputs)
        if kind == "move":
            return move_station(config, slot, int(self.candidates[k]), self.inputs)
        if kind in ("engine", "quint"):
            return add_vehicle(config, slot, kind == "quint")
        return config

    def observe(self):
        config, result = self.config, self.last
        site = config["site"]
        active = site >= 0
        xy = np.where(active[:, None], (self.site_xy[np.maximum(site, 0)] - self.center) / 30000.0, 0.0)
        stations = np.column_stack([active, xy, config["gfr"], config["engines"] / 3.0, config["quints"] / 2.0, self.slot_busy])
        used = set(site[active].tolist())
        candidate_xy = (self.site_xy[self.candidates] - self.center) / 30000.0
        candidates = np.column_stack([[c not in used for c in self.candidates], candidate_xy, self.inputs["sites"]["gfr"].to_numpy()[self.candidates]])
        tracts = np.column_stack([(self.multipliers[self.stage] - 1) * 5.0, np.nan_to_num(result["tract_fire_mean"]) / 10.0,
                                  np.nan_to_num(result["tract_mean"]) / 10.0, self.fire_share * 10.0, self.share * 10.0])
        flags = np.concatenate([np.eye(4)[min(self.stage, 3)], np.eye(2)[self.sub]])
        return np.concatenate([stations.ravel(), candidates.ravel(), tracts.ravel(), flags]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        draw = (options or {}).get("draw")
        if draw is None:
            draw = (self.rng.choice(list(PATH_WEIGHTS), p=list(PATH_WEIGHTS.values())), float(self.rng.random()))
        self.draw = (str(draw[0]), float(draw[1]))
        self.multipliers = growth_multipliers(self.inputs, *self.draw)
        self.config, self.stage, self.sub = copy.deepcopy(self.start), 0, 0
        self.last, self.history, self.actions, self.total, self.deviations = self.first_result, [], [], 0.0, 0
        self.slot_busy = slot_busy(self.start, self.first_result["busy"])
        return self.observe(), {"draw": self.draw}

    def step(self, action):
        action = int(action)
        self.config = self.apply(self.config, action)
        self.actions.append(self.decode(action))
        self.deviations += action != 0
        reward, terminated = 0.0, False
        if self.sub == 0:
            self.sub = 1
        else:
            result = self.evaluator.evaluate(self.config, self.multipliers[self.stage + 1])
            reward = -stage_cost(result, self.targets)
            fire_over, other_over = violations(result, self.targets)
            self.history.append({"year": STAGE_YEARS[self.stage], "M_F": result["M_F"], "D_F": result["D_F"], "M_O": result["M_O"], "M": result["M"],
                                 "D": result["D"], "D_fixed": result["D_fixed"], "waiting": result["waiting"], "waiting_fire": result["waiting_fire"],
                                 "violation_fire": fire_over, "violation_other": other_over, "violation": fire_over or other_over, "actions": self.actions[-2:],
                                 "tract_fire_mean": result["tract_fire_mean"], "config": copy.deepcopy(self.config)})
            self.last, self.stage, self.sub = result, self.stage + 1, 0
            self.slot_busy = slot_busy(self.config, result["busy"])
            terminated = self.stage == len(STAGE_YEARS)
        self.total += reward
        info = {}
        if terminated:
            info = {"unshaped_return": self.total, "decisions": 2 * len(STAGE_YEARS), "deviations": self.deviations,
                    "history": self.history, "draw": self.draw}
        return self.observe(), float(reward), terminated, False, info


# ----------------------------------------------------------------------------------------------------------------------
# Baselines and validation draws
# ----------------------------------------------------------------------------------------------------------------------

def play(env, choose, draw):
    """Play one growth draw with a policy function choose(env, observation) -> action; return the final info."""
    observation, _ = env.reset(options={"draw": draw})
    while True:
        observation, _, terminated, _, info = env.step(choose(env, observation))
        if terminated:
            return info


def do_nothing(env, observation):
    """Baseline: never act."""
    return 0


def city_plan(env, observation):
    """Baseline: move the existing Station 9 with its vehicles to its documented site in the first stage, nothing else."""
    if env.stage == 0 and env.sub == 0:
        slot = env.start["name"].index(STATION9)
        return 1 + env.n_build + slot * env.k + 0                     # candidate 0 is the documented site
    return 0


def expected_next_multipliers(env):
    """The demand the stage greedy rule plans for: the expected end-of-period multipliers given the observed population.

    In 2025 nothing is observed: the expectation over PATH_WEIGHTS with w = 0.5 (the multipliers are linear in w). Later,
    for each path the w in [0, 1] that fits the observed multipliers best (least squares, closed form since the
    multipliers are linear in w) is found, and the path with the smallest residual gives the next year's multipliers.
    """
    if env.stage == 0:
        return sum(weight * growth_multipliers(env.inputs, path, 0.5)[1] for path, weight in PATH_WEIGHTS.items())
    observed = env.multipliers[env.stage]
    fits = []
    for path in PATH_WEIGHTS:
        low, high = growth_multipliers(env.inputs, path, 0.0)[env.stage], growth_multipliers(env.inputs, path, 1.0)[env.stage]
        span = high - low
        w = float(np.clip((observed - low) @ span / (span @ span), 0.0, 1.0)) if span @ span > 0 else 0.5
        fits.append((float(((low + w * span - observed) ** 2).sum()), path, w))
    _, path, w = min(fits)
    return growth_multipliers(env.inputs, path, w)[env.stage + 1]


class StageGreedy:
    """Baseline: at each stage, the station action and its best vehicle action with the lowest stage cost at the expected
    end-of-period demand (expected_next_multipliers); the pair is chosen at the station sub-step."""

    def __init__(self):
        self.vehicle_action = 0

    def __call__(self, env, observation):
        if env.sub == 1:
            return self.vehicle_action
        demand = expected_next_multipliers(env)
        best = (np.inf, 0, 0)
        station_mask = env.action_masks()
        for station_action in np.flatnonzero(station_mask):
            after = env.apply(env.config, int(station_action))
            env_config, env_sub = env.config, env.sub
            env.config, env.sub = after, 1
            vehicle_mask = env.action_masks()
            env.config, env.sub = env_config, env_sub
            for vehicle_action in np.flatnonzero(vehicle_mask):
                cost = stage_cost(env.evaluator.evaluate(env.apply(after, int(vehicle_action)), demand), env.targets)
                if cost < best[0]:
                    best = (cost, int(station_action), int(vehicle_action))
        self.vehicle_action = best[2]
        return best[1]


def validation_draws(count, seed):
    """Fixed growth draws for validation: a list of (path, w), paths by PATH_WEIGHTS and w uniform."""
    rng = np.random.default_rng(seed)
    return [(str(rng.choice(list(PATH_WEIGHTS), p=list(PATH_WEIGHTS.values()))), float(rng.random())) for _ in range(count)]


# ----------------------------------------------------------------------------------------------------------------------
# The bridge to the dispatch simulator
# ----------------------------------------------------------------------------------------------------------------------

def simulator_inputs(world, scenario, inputs, config):
    """Turn a configuration into the world and scenario that environment.simulate needs.

    Moved stations keep their vehicles and agency; new vehicles get new ids; every vehicle is in service throughout. The
    scenario's modeled travel is recomputed from each vehicle's node to each run's call node, with the agency line.

    Input
        world, scenario : the dict from environment.build_world and a scenario from scenarios.build_scenario
        inputs, config  : the planning inputs and a configuration
    Output
        (world, scenario) copies with engines, home_nodes, quint_indicator and travel replaced.
    """
    sites, stations, engines = inputs["sites"], inputs["stations"], world["engines"].set_index("engine_id")
    rows = []
    for slot in np.flatnonzero(config["site"] >= 0):
        node = int(sites["node"].iloc[config["site"][slot]])
        owned = stations["vehicle_ids"].iloc[slot] if slot < len(stations) else []
        kinds = ["E"] * config["engines"][slot] + ["Q"] * config["quints"][slot]
        existing = {kind: [v for v in owned if engines.at[v, "engine_type"] == kind] for kind in ("E", "Q")}
        for kind in ("E", "Q"):
            for index in range(kinds.count(kind)):
                vehicle = existing[kind][index] if index < len(existing[kind]) else f"P{kind}_{config['name'][slot]}_{index + 1}"
                rows.append({"engine_id": vehicle, "engine_type": kind, "engine_agency": "GFR" if config["gfr"][slot] else "non-GFR",
                             "home_station": config["name"][slot], "node": node})
    fleet = pd.DataFrame(rows)
    graph = world["graph"]
    fleet["home_lon"] = [graph.nodes[node]["x"] for node in fleet["node"]]
    fleet["home_lat"] = [graph.nodes[node]["y"] for node in fleet["node"]]
    fleet["in_service_from"], fleet["in_service_to"] = DAY_ZERO, pd.Timestamp("2100-01-01")
    fleet["n_dispatches"], fleet["first_seen"], fleet["last_seen"] = 0, DAY_ZERO, DAY_ZERO
    planned_world = dict(world)
    planned_world["engines"] = fleet.drop(columns="node")[world["engines"].columns].reset_index(drop=True)
    planned_world["home_nodes"] = pd.Series(fleet["node"].to_numpy(), index=fleet["engine_id"]).astype("int64")
    planned_world["quint_indicator"] = pd.Series((fleet["engine_type"] == "Q").astype(float).to_numpy(), index=fleet["engine_id"])
    call_nodes = scenario["call_nodes"]["node"]
    located = call_nodes.notna().to_numpy()
    positions = world["node_position"][call_nodes[located].astype("int64")].to_numpy()
    values = np.full((len(call_nodes), len(fleet)), np.nan)
    for column, node in enumerate(fleet["node"]):
        values[located, column] = world["seconds_from_node"](int(node))[positions]
    travel = pd.DataFrame(world["travel_model"](values), index=scenario["dispatches"].index, columns=fleet["engine_id"])
    planned_scenario = dict(scenario)
    planned_scenario["travel"] = environment.apply_agency_boundaries(travel, scenario["dispatches"], planned_world["engines"],
                                                                     world["aid_protocols"], verbose=False)
    return planned_world, planned_scenario


def grown_scenario(world, generator, inputs, multipliers, seed):
    """A generated October-March scenario whose other calls grow by tract multipliers (study fires unchanged).

    The extra other calls go to residential blocks in proportion to each tract's added calls (its base other-call rate
    times its multiplier minus one), and within a tract in proportion to residents; a declining path drops other calls
    uniformly.

    Output
        The scenario (dict from scenarios.build_scenario) for the current world's engines; simulator_inputs re-prices it
        for a configuration.
    """
    days = range(int(WINDOW_HOURS // 24))
    calls, day_factors = scenarios.generate_calls(generator["pool"], generator["counts"], generator["dispersion"], days, seed, verbose=False)
    other_rate = np.bincount(inputs["points"]["tract"], weights=inputs["points"]["other_rate"], minlength=len(inputs["tracts"]))
    factor = float((other_rate * multipliers).sum() / other_rate.sum())
    if abs(factor - 1) > 1e-9:
        locations = generator["locations"].copy()
        block_tract = block_tracts(generator, inputs)
        residents_in_tract = locations.groupby(block_tract)["residents"].transform("sum")
        added = np.clip(multipliers - 1, 0, None) * other_rate
        weight = locations["residents"] / residents_in_tract * added[block_tract]
        locations["residents"] = weight.fillna(0.0) if factor > 1 else locations["residents"]
        calls = scenarios.add_demand_growth(calls, day_factors, generator["pool"], generator["counts"], locations, factor, seed, verbose=False)
    no_outages = pd.DataFrame(columns=["engine_id", "start_s", "end_s"])
    return scenarios.build_scenario(world, generator, calls, no_outages, seed, verbose=False)


def block_tracts(generator, inputs):
    """The tract row of each residential block of the generator (array aligned with generator["locations"])."""
    locations = generator["locations"]
    points = gpd.GeoDataFrame(index=locations.index, geometry=gpd.points_from_xy(locations["x"], locations["y"]), crs=PROJECTED_CRS)
    joined = gpd.sjoin(points, inputs["tracts"][["geometry"]], predicate="within", how="left")["index_right"]
    joined = joined[~joined.index.duplicated()].reindex(locations.index)
    return joined.fillna(joined.mode().iloc[0]).astype(int).to_numpy()


def simulated_outcome(world, scenario, simulated, inputs):
    """Measure one simulated run as the fast evaluator does: the first-arriving vehicle's travel per incident, split into
    study fire calls and other calls, kept as sums and counts by tract so that several seeds can be pooled.

    Output
        A dict: fire_sum, fire_count, other_sum, other_count (arrays over tracts: travel minutes summed over the arriving,
        tract-located incidents, and their number), fire_travel and other_travel (arrays of those incidents' travel),
        fire_waited and other_waited (how many of them had their first run dispatched later than in the copied record
        because no eligible vehicle was free), and busy (Series: share of the window each vehicle was busy).
    """
    dispatches = scenario["dispatches"]
    runs = simulated.assign(incident_id=dispatches["incident_id"].to_numpy(), lon=dispatches["incident_lon"].to_numpy(),
                            lat=dispatches["incident_lat"].to_numpy(), fire=dispatches["protocol"].isin(STUDY_FIRE_PROTOCOLS).fillna(False).to_numpy())
    first = runs.dropna(subset=["arrived_at", "enroute_at", "lon"]).sort_values("arrived_at").groupby("incident_id").head(1)
    travel = (first["arrived_at"] - first["enroute_at"]).to_numpy() / 60
    x, y = pyproj.Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform(first["lon"].to_numpy(), first["lat"].to_numpy())
    located = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=PROJECTED_CRS)
    tract = gpd.sjoin(located, inputs["tracts"][["geometry"]], predicate="within", how="left")["index_right"]
    tract = tract[~tract.index.duplicated()].reindex(located.index).to_numpy()
    keep = ~np.isnan(tract)
    tract, travel, fire = tract[keep].astype(int), travel[keep], first["fire"].to_numpy()[keep]
    ids = first["incident_id"].to_numpy()[keep]
    n_tracts = len(inputs["tracts"])
    window = WINDOW_HOURS * 3600
    busy = simulated.dropna(subset=["engine_id"]).assign(seconds=lambda t: (t["cleared_at"] - t["dispatched_at"]).clip(lower=0)).groupby("engine_id")["seconds"].sum() / window
    delay = simulated["dispatched_at"] - scenario["runs"]["dispatched_at"]           # seconds a run waited beyond its copied dispatch time
    first_dispatch = runs.assign(delay=delay.to_numpy()).dropna(subset=["dispatched_at", "lon"]).sort_values("dispatched_at").groupby("incident_id").head(1)
    waited = set(first_dispatch.loc[first_dispatch["delay"] > 1.0, "incident_id"])
    waited_mask = np.array([incident in waited for incident in ids], dtype=bool)
    return {"fire_sum": np.bincount(tract[fire], weights=travel[fire], minlength=n_tracts),
            "fire_count": np.bincount(tract[fire], minlength=n_tracts).astype(float),
            "other_sum": np.bincount(tract[~fire], weights=travel[~fire], minlength=n_tracts),
            "other_count": np.bincount(tract[~fire], minlength=n_tracts).astype(float),
            "fire_travel": travel[fire], "other_travel": travel[~fire], "fire_waited": int(waited_mask[fire].sum()),
            "other_waited": int(waited_mask[~fire].sum()), "busy": busy}


def pooled_metrics(outcomes, inputs):
    """Pool simulated outcomes of several seeds (list of dicts from simulated_outcome) and measure them as the evaluator
    does: tract sums and counts are added before the tract means, and D_F uses the tracts' fixed base fire rates.

    Output
        A dict: M_F, D_F, M_O, M (all calls), P90_F (of fire travel), waiting_fire, waiting_other (shares), busy (Series,
        mean over seeds), tract_fire_mean, fire_incidents, and missing_fire_tracts (tracts with a base fire rate but no
        simulated fire incident, left out of D_F).
    """
    fire_sum, fire_count = sum(o["fire_sum"] for o in outcomes), sum(o["fire_count"] for o in outcomes)
    other_sum, other_count = sum(o["other_sum"] for o in outcomes), sum(o["other_count"] for o in outcomes)
    n_tracts = len(inputs["tracts"])
    fire_weight = np.bincount(inputs["points"]["tract"], weights=inputs["points"]["fire_rate"], minlength=n_tracts)
    with np.errstate(invalid="ignore", divide="ignore"):
        tract_fire_mean = fire_sum / fire_count
    measured = fire_count > 0
    weights = np.where(measured, fire_weight, 0.0)
    fire_travel = np.concatenate([o["fire_travel"] for o in outcomes])
    return {"M_F": float(fire_sum.sum() / fire_count.sum()), "D_F": weighted_sd(np.nan_to_num(tract_fire_mean), weights),
            "M_O": float(other_sum.sum() / other_count.sum()), "M": float((fire_sum.sum() + other_sum.sum()) / (fire_count.sum() + other_count.sum())),
            "P90_F": float(np.quantile(fire_travel, 0.9)), "waiting_fire": sum(o["fire_waited"] for o in outcomes) / fire_count.sum(),
            "waiting_other": sum(o["other_waited"] for o in outcomes) / other_count.sum(),
            "busy": pd.concat([o["busy"] for o in outcomes], axis=1).fillna(0.0).mean(axis=1), "tract_fire_mean": tract_fire_mean,
            "fire_incidents": int(fire_count.sum()), "missing_fire_tracts": int(((fire_weight > 0) & ~measured).sum())}


if __name__ == "__main__":
    world = environment.build_world(verbose=False)
    generator = scenarios.fit_generator(world, verbose=False)
    inputs = build_planning_inputs(world, generator)
    evaluator = FastEvaluator(world, inputs)
    start = start_configuration(inputs)
    now = evaluator.evaluate(start, np.ones(len(inputs["tracts"])))
    print(f"current configuration at 2025 demand (uncalibrated): M_F {now['M_F']:.3f} min, D_F {now['D_F']:.3f}, M_O {now['M_O']:.3f}, "
          f"mean busy {now['busy'].mean():.3f}, fire waiting share {now['waiting_fire']:.5f}")
    for path, w in [("Low", 0.5), ("Medium", 0.0), ("Medium", 1.0), ("High", 0.0), ("High", 1.0)]:
        m = growth_multipliers(inputs, path, w)
        result = evaluator.evaluate(start, m[-1])
        print(f"do nothing, {path} w={w}: 2045 county x{(m[-1] * inputs['tracts']['pop2025']).sum() / COUNTY_2025:.3f}; "
              f"M_F {result['M_F']:.3f}, D_F {result['D_F']:.3f}, M_O {result['M_O']:.3f}")
