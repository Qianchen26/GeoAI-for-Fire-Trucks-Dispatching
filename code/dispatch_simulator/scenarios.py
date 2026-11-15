"""The scenario generator: synthetic calls for training and testing dispatch policies.

A synthetic scenario keeps what is real and resamples what is scarce. Roads, stations and engines stay as they are,
and every new call copies a real call from the training months (October to March). Other calls (every call that is
not one of the study's fire calls) are copied
as whole real days: each scenario day takes all the other calls of one real day of the same kind (weekday or weekend)
within two weeks of it. Copying a whole day keeps the real timing of the day's calls relative to each other. Study
fire calls are drawn hour by hour instead. Each scenario day first draws a day factor from a gamma distribution, which
lets some days be busier than chance alone. Each hour then gets a Poisson number of calls, and each call copies a real
study fire call received in the same block of hours within two weeks of the day. Conditions then change a scenario:
more or fewer other calls, short local surges, engines out of service, a closed station, or a large fire each week.

Usage: call fit_generator(world) once. Then generate_calls, the add_* functions and close_station make the calls and
outages, and
build_scenario turns them into a scenario that environment.simulate can run.
"""
import zlib

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import scipy.spatial

from environment import (DAY_ZERO, OUT_OF_SERVICE_PROTOCOL, PROJECT_ROOT, PROJECTED_CRS, READINESS_CELL_M,
                         STUDY_FIRE_PROTOCOLS, TIME_COLUMNS, TRAINING_MONTHS_END, VERBOSE, apply_agency_boundaries,
                         call_fates, estimate_network_travel_seconds, run_attributes, show_table)

CENSUS_BLOCKS_GPKG = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "census2020_blocks_alachua.gpkg"  # [SOURCE] 2020 Census blocks: residents (POP100) and internal points
SCENARIO_WINDOW_DAYS = 14                                                   # [DESIGN] a scenario day's call rates and copied calls come from training days this close to it
SCENARIO_BAND_HOURS = 4                                                     # [DESIGN] a new call copies a real call received in the same block of this many hours
STRUCTURE_FIRE_PROTOCOL = "F69"                                             # [SOURCE] MPDS fire protocol 69: structure fire (where a large fire is placed)
SURGES_PER_DAY = 1                                                          # [DESIGN] short surges per day in the surge condition
SURGE_RADIUS_M = 2500                                                       # [DESIGN] a short surge's calls fall within this distance of its centre (the record's bursts were counted within the same distance)
SURGE_HOURS = 2                                                             # [DESIGN] length of a short surge (the record's bursts were counted over 2 h)
SURGE_EXTRA_CALLS = [6, 12, 24]                                             # [DESIGN] a stress sweep of extra calls per surge; 6 extra calls plus about 1 usual call match the record's largest local burst (7 other calls within 2.5 km in 2 h), and 12 and 24 are two and four times 6
LARGE_FIRES_PER_WEEK = 1                                                    # [DESIGN] large fires per week in the large-fire condition
LARGE_FIRE_ENGINES = 6                                                      # [PLACEHOLDER] engines a large fire holds: a first alarm's 3 engine-class units (ACFR response matrix), doubled for a second alarm; to confirm with GFR/ACFR
LARGE_FIRE_ON_SCENE_S = 4 * 3600                                            # [SOURCE] for each of the 8 largest October-March fires (by engine-hours), take its longest single-engine dispatch-to-clear time; the median of these 8 times is 4.0 h


def build_call_pool(dispatches, verbose=VERBOSE):
    """Collect the real calls that a scenario copies: every training-month incident that has a location.

    Input
        dispatches : the record (DataFrame, one row per run)
        verbose    : whether to print the pool and its counts (bool)
    Output
        The pool (DataFrame indexed by incident_id, one row per incident): received_s (seconds since DAY_ZERO), day
        (days since DAY_ZERO), hour (hour of the day, 0-23), band (the block of SCENARIO_BAND_HOURS hours that holds the
        hour, 0-5), weekend (received on a Saturday or Sunday), fire (a study fire call or not), protocol, jurisdiction,
        focal (inside the city), x and y (metres, PROJECTED_CRS), cell_x and cell_y (the call's cell on a grid of
        READINESS_CELL_M-metre squares), run_rows (the record rows of its runs), runs (the number of its runs), and
        arrival_row (the record row of its first run to arrive, -1 when none arrived).
    Example
        pool = build_call_pool(dispatches)
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
    """Collect the places where demand growth and surges put new calls: every 2020 Census block with residents.

    More residents means more calls, so a new call goes to a block drawn in proportion to its residents. Each block is
    placed at its internal point, snapped to the road network. It takes the agency side and city flag of the nearest
    training-month call. Its first-due station is the home station of the closest engine of its own agency side,
    measured in free-flow network seconds from the engine's station to the block. An engine without a station id counts
    as its own station.

    Input
        pool              : the call pool from build_call_pool
        graph             : the road network
        home_nodes        : the road node of each engine's home station (Series indexed by engine id)
        seconds_from_node : the shortest-path function from make_seconds_from_node
        node_position     : the position of each road node in the arrays that seconds_from_node returns (Series indexed
                            by road node id)
        engines           : the engines (DataFrame)
        verbose           : whether to print the blocks and their first-due station areas (bool)
    Output
        Two tables. locations (DataFrame, one row per block with residents): residents, lon, lat, x, y, cell_x, cell_y,
        node and snap_m (the nearest road node and the metres to it), jurisdiction, focal and first_due_station.
        location_seconds (DataFrame, rows: blocks, columns: engines): free-flow network seconds from each engine's
        station to the block, as network_from_stations gives for a call.
    Example
        locations, location_seconds = build_resident_locations(pool, graph, home_nodes, seconds_from_node, node_position, engines)
        locations["residents"].sum() -> 278468
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


def fit_call_rates(pool, verbose=VERBOSE):
    """Count how many calls arrive per training day and hour, and how much busier some days are than chance alone.

    The day-to-day spread is measured by phi, the dispersion of a negative binomial (variance = mean + phi x mean^2).
    phi is fitted by the method of moments: each day's total is compared with the mean total of the other days of the
    same kind (weekday or weekend) within SCENARIO_WINDOW_DAYS of it. phi is 0 when the spread is no wider than chance.

    Input
        pool    : the call pool from build_call_pool
        verbose : whether to print the counts and the spread (bool)
    Output
        Two objects. counts (DataFrame, rows: training days since DAY_ZERO; columns: (fire, hour) pairs, where fire is
        False for other calls and True for study fire calls and hour runs from 0 to 23): calls per day and hour.
        dispersion (dict): phi (float) for each fire flag.
    Example
        counts, dispersion = fit_call_rates(pool)
        counts.shape -> (182, 48); dispersion -> about {False: 0.01, True: 0.06}
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


def fit_generator(world, verbose=VERBOSE):
    """Fit the generator's inputs once: the call pool, the residential blocks and the call rates.

    Input
        world   : the world (dict from environment.build_world)
        verbose : whether every step prints its tables (bool)
    Output
        The generator (dict): pool from build_call_pool, locations and location_seconds from build_resident_locations,
        and counts and dispersion from fit_call_rates.
    Example
        generator = fit_generator(world, verbose=False)
        len(generator["pool"]) -> incidents received October to March with a location
    """
    pool = build_call_pool(world["dispatches"], verbose)
    locations, location_seconds = build_resident_locations(pool, world["graph"], world["home_nodes"], world["seconds_from_node"],
                                                           world["node_position"], world["engines"], verbose)
    counts, dispersion = fit_call_rates(pool, verbose)
    return {"pool": pool, "locations": locations, "location_seconds": location_seconds, "counts": counts, "dispersion": dispersion}


def expected_hourly_counts(counts, day):
    """Compute the mean number of calls per hour expected on a scenario day, from training days like it.

    Each (fire, hour) count is averaged over the training days of the same kind (weekday or weekend) within
    SCENARIO_WINDOW_DAYS of the day.

    Input
        counts : calls per training day and hour (DataFrame from fit_call_rates)
        day    : the scenario day, days since DAY_ZERO (int)
    Output
        The expected number of calls for each fire flag and hour (Series with 48 entries, indexed by (fire, hour)).
    Example
        expected_hourly_counts(counts, 0)[(False, 12)] -> about 4.9      # other calls, noon to 1 pm, Tuesday 1 October
    """
    days = counts.index.to_numpy()
    weekend = (DAY_ZERO + pd.to_timedelta(days, unit="D")).dayofweek >= 5
    near = (np.abs(days - day) <= SCENARIO_WINDOW_DAYS) & (weekend == ((DAY_ZERO + pd.Timedelta(days=int(day))).dayofweek >= 5))
    return counts[near].mean()


def draw_source_calls(pool, day, fire, band, size, rng, candidates):
    """Draw the real calls that new calls copy: same group (fire or not), same block of hours, within two weeks.

    The real calls are drawn with replacement among the pool's calls received within SCENARIO_WINDOW_DAYS of the day,
    on any kind of day (weekday or weekend).

    Input
        pool       : the call pool from build_call_pool
        day        : the scenario day, days since DAY_ZERO (int)
        fire       : whether the new calls are study fire calls (bool)
        band       : the block of SCENARIO_BAND_HOURS hours, counted from midnight (int, 0-5), e.g. 3 for 12:00-15:59
        size       : how many real calls to draw (int)
        rng        : the random generator to draw with (numpy Generator)
        candidates : a cache that the caller keeps across calls of this function (dict that maps (day, fire, band) to
                     the positions of the matching calls in the pool)
    Output
        The incident ids of the drawn real calls (Index of str).
    Example
        draw_source_calls(pool, 0, False, 3, 2, np.random.default_rng(1), {}) -> two incident ids received 12:00-15:59
        within 14 days of 1 October, e.g. Index(['100824-0213', '101124-0287'])
    """
    key = (day, fire, band)
    if key not in candidates:
        candidates[key] = np.flatnonzero((np.abs(pool["day"].to_numpy() - day) <= SCENARIO_WINDOW_DAYS) & (pool["fire"].to_numpy() == fire)
                                         & (pool["band"].to_numpy() == band))
    return pool.index[rng.choice(candidates[key], size=size)]


def generate_calls(pool, counts, dispersion, days, seed, verbose=VERBOSE):
    """Generate the calls of a nominal scenario (no conditions) for the given days.

    Other calls (those that are not study fire calls): each day copies all the other calls of one real training day, drawn at random among the
    training days of the same kind (weekday or weekend) within SCENARIO_WINDOW_DAYS of the day. Each copied call keeps
    its time of day, type, location and fate. Copying a whole real day keeps the timing of its calls relative to each
    other; calls drawn one by one would lose it. Study fire calls: each day draws a day factor from a gamma distribution
    with mean 1 and variance phi, which lets some days be busier than chance alone. Each hour then gets a Poisson number
    of calls, with a mean equal to the day factor times the hour's expected count. Each call comes at a whole second
    drawn uniformly within the hour, and it copies the type, location and fate of a real study fire call received in
    the same block of hours within SCENARIO_WINDOW_DAYS.

    Input
        pool       : the call pool from build_call_pool
        counts     : calls per training day and hour (DataFrame from fit_call_rates)
        dispersion : phi for each fire flag (dict from fit_call_rates)
        days       : the scenario days, days since DAY_ZERO within the training months (range)
        seed       : the scenario's seed (int)
        verbose    : whether to print the first week, the first day by hour, and checks (bool)
    Output
        Two tables. calls (DataFrame, one row per new call, sorted by time): incident_id, received_s (seconds since
        DAY_ZERO), source_incident (the incident id of the real call it copies), fire, large_fire, kind ("nominal") and
        location (-1, which means the call keeps its source call's location). day_factors (DataFrame, rows: days;
        columns: the fire flag): for other calls, the copied day's total divided by the expected total; for study fire
        calls, the gamma day factor.
    Example
        calls, day_factors = generate_calls(pool, counts, dispersion, range(7), seed=101)
        calls.iloc[0].tolist() -> ['syn101-000000', 1873.0, '100424-0016', False, False, 'nominal', -1]   # day 0 copies day 3
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
    """Scale a scenario's other calls (those that are not study fire calls) by a factor; study fire calls stay as they are.

    When the factor is below 1, each other call is dropped with probability 1 - factor. When it is above 1, each day
    and hour gets a Poisson number of extra other calls, with a mean of (factor - 1) times the day's other-call factor
    times the hour's expected count. Each extra call comes at a whole second drawn uniformly within the hour, and it
    goes where people live: a residential block drawn in proportion to its 2020 residents. Its type and fate come from
    a real other call drawn by draw_source_calls: one received in the same block of hours within SCENARIO_WINDOW_DAYS.

    Input
        calls       : the scenario's calls (DataFrame from generate_calls)
        day_factors : the day factors from generate_calls (DataFrame)
        pool        : the call pool
        counts      : calls per training day and hour (DataFrame from fit_call_rates)
        locations   : the residential blocks from build_resident_locations
        factor      : the demand factor for other calls (float), e.g. 1.3
        seed        : the scenario's seed (int)
        verbose     : whether to print the counts, the share of other calls in the record's 20 busiest cells, and three
                      extra calls (bool)
    Output
        The calls with the change (DataFrame, sorted by time). Extra calls have kind "growth" and a location of 0 or
        more: the row of their residential block in locations.
    Example
        grown = add_demand_growth(calls, day_factors, pool, counts, locations, 1.3, seed=101)
        (~grown["fire"]).sum() / (~calls["fire"]).sum() -> about 1.30
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
    """Add short local surges of other calls (calls that are not study fire calls) to a scenario, as a stress test.

    The record holds no local burst beyond chance: at most 7 other calls within SURGE_RADIUS_M in SURGE_HOURS. A surge
    adds extra_calls other calls over SURGE_HOURS, inside a disk of radius SURGE_RADIUS_M around a centre. The surge
    starts at a whole second drawn uniformly within its day, and each of its calls comes at a whole second drawn
    uniformly within the surge. There are SURGES_PER_DAY surges a day. Surges visit the first-due station areas in
    turn, so every area is tested equally often (as in the risk map of Usanov et al. 2019); the order of the areas is
    drawn once for each seed. The centre is a residential block of the visited area, drawn in proportion to residents.
    Each call's block is drawn the same way among the blocks inside the disk, and each call copies its type and fate
    from a real other call (draw_source_calls). Each surge draws max(SURGE_EXTRA_CALLS) calls from its own random
    stream ([seed, 2, surge number]) and keeps the first extra_calls of them. A smaller surge is therefore the start of
    a larger one at the same place and time. A call's incident id names its surge and its rank in the surge, so the
    call keeps its travel quantiles at every surge size.

    Input
        calls       : the scenario's calls (DataFrame from generate_calls)
        pool        : the call pool
        locations   : the residential blocks from build_resident_locations
        days        : the scenario days (range)
        extra_calls : the number of extra calls per surge (int), one of SURGE_EXTRA_CALLS
        seed        : the scenario's seed (int)
        verbose     : whether to print the surges, the first surge's calls, and checks (bool)
    Output
        Two tables. The calls with the surges (DataFrame, sorted by time; extra calls have kind "surge"). The surges
        (DataFrame, one row per surge): day, start_s and end_s (seconds since DAY_ZERO), first_due_station, centre_block
        (the centre's row in locations), cell_x, cell_y, focal, extra_calls and usual_other_calls_in_disk (the record's
        mean number of other calls in the disk per SURGE_HOURS over the training months, for comparison with the surge).
    Example
        surged, surges = add_short_surges(calls, pool, locations, range(182), 12, seed=101)
        len(surges) -> 182      # one surge a day
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
            incident_ids.extend(f"{seed_label}-s{number:05d}-{rank:02d}" for rank in range(extra_calls))   # a call keeps its id at every surge size, so it keeps its travel quantiles
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


def add_outages(engines, days, engine_count, seed, verbose=VERBOSE):
    """Take engines out of service: each day, engine_count engines drawn at random are out from midnight to midnight.

    Only engines in service for the whole day can be drawn.

    Input
        engines      : the engines (DataFrame with in_service_from and in_service_to)
        days         : the scenario days (range)
        engine_count : engines out each day (int), e.g. 2
        seed         : the scenario's seed (int)
        verbose      : whether to print the first days and the engines out most often (bool)
    Output
        The outages (DataFrame, one row per engine-day): engine_id, start_s and end_s in seconds since DAY_ZERO.
    Example
        add_outages(engines, range(2), 2, seed=101) -> 4 rows, e.g. E24 0-86,400 s, E5 0-86,400 s, then two on day 1
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
    """Close a station: every engine housed there is out of service from the first scenario day to the end of the last.

    Input
        engines : the engines (DataFrame with home_station)
        station : the station id (str), e.g. "GFR_03"
        days    : the scenario days (range)
        verbose : whether to print the outages (bool)
    Output
        The outages (DataFrame): engine_id, start_s and end_s in seconds since DAY_ZERO.
    Example
        close_station(engines, "GFR_03", range(182)) -> one row: E3, 0 to 15,724,800 s
    """
    outages = pd.DataFrame({"engine_id": engines.loc[engines["home_station"].eq(station), "engine_id"].to_numpy(),
                            "start_s": min(days) * 86400.0, "end_s": (max(days) + 1) * 86400.0})
    if verbose:
        print(f"--- station {station} closed:\n{outages.to_string()}\n")
    return outages


def add_large_fires(calls, pool, days, seed, verbose=VERBOSE):
    """Add LARGE_FIRES_PER_WEEK large fires each week, at the locations of real structure fires.

    Each large fire comes at a whole second drawn uniformly within its week. Its location is that of a real
    training-month structure fire (STRUCTURE_FIRE_PROTOCOL, with an engine on scene), drawn at random. build_scenario
    gives it LARGE_FIRE_ENGINES runs, which all arrive and stay on scene for LARGE_FIRE_ON_SCENE_S.

    Input
        calls   : the scenario's calls (DataFrame from generate_calls)
        pool    : the call pool
        days    : the scenario days (range); weeks are blocks of 7 days counted from DAY_ZERO
        seed    : the scenario's seed (int)
        verbose : whether to print the first three large fires (bool)
    Output
        The calls with the large fires (DataFrame, sorted by time; large fires have kind "large fire").
    Example
        add_large_fires(calls, pool, range(182), seed=101)["large_fire"].sum() -> 26
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


def build_scenario(world, generator, calls, outages, seed, verbose=VERBOSE):
    """Turn a scenario's calls and outages into the inputs that simulate needs, by copying real runs.

    Each new call copies the runs of its real incident: the record's 32 columns and the repaired times. All the copied
    times move by the same whole-second shift (the new call time minus the real one), and the copied runs take the new
    incident id. A call placed at a residential block (calls["location"] >= 0) takes the block's position, agency side,
    city flag, road node and network seconds instead of its source's. A large fire makes LARGE_FIRE_ENGINES copies of
    the first run that arrived at its structure fire; all the copies arrive and stay on scene for
    LARGE_FIRE_ON_SCENE_S. Each outage becomes a stand-in run with no location (protocol OUT_OF_SERVICE_PROTOCOL) that
    holds its engine from start to end. Like any call without a location, the stand-in gets that engine when it is
    free and otherwise waits for it. A waiting stand-in gets the engine ahead of every waiting call, and it still ends
    at its scheduled end.

    Input
        world     : the world (dict from environment.build_world) with the record, engines, network seconds, travel
                    model, aid call types and quint penalties
        generator : the generator (dict from fit_generator) with pool, locations and location_seconds
        calls     : the scenario's calls (DataFrame from generate_calls and the add_* conditions)
        outages   : engine outages (DataFrame with engine_id, start_s and end_s; empty for none)
        seed      : the scenario's seed (int), which also seeds each run's travel quantile
        verbose   : whether to print the table and checks (bool)
    Output
        The scenario (dict), with the keys simulate reads and a few more: dispatches (the record's 32 columns), runs
        (the runs, with times in seconds since DAY_ZERO), source_row (the record row each run copies; -1 for a
        stand-in), large_fire (whether the run belongs to a large fire), travel (modeled travel from every station, with
        the agency line drawn in), call_nodes (each run's road node and snap distance), spread_quantile (each run's
        travel quantile, drawn from the seed and its incident id; a call therefore keeps its travel quantiles when a
        condition adds calls), fates, inside_city, wait_rank, quint_penalty, brush and low_acuity.
    Example
        scenario = build_scenario(world, generator, calls, no_outages, seed=101)
        scenario["dispatches"].shape -> about (18500, 32) for a nominal October-March
    """
    dispatches, record_runs, engines = world["dispatches"], world["runs"], world["engines"]
    pool, locations, location_seconds = generator["pool"], generator["locations"], generator["location_seconds"]
    time_columns = ["call_received_at"] + TIME_COLUMNS
    copied = []                                                  # (source row, new incident id, shift in seconds, large fire, block)
    for call in calls.itertuples(index=False):
        rows = [pool.at[call.source_incident, "arrival_row"]] * LARGE_FIRE_ENGINES if call.large_fire else pool.at[call.source_incident, "run_rows"]
        shift_s = call.received_s - pool.at[call.source_incident, "received_s"]
        copied += [(row, call.incident_id, shift_s, call.large_fire, call.location) for row in rows]
    copied = pd.DataFrame(copied, columns=["source_row", "incident_id", "shift_s", "large_fire", "location"])
    table = dispatches.loc[copied["source_row"]].reset_index(drop=True)
    runs = record_runs.loc[copied["source_row"]].reset_index(drop=True)
    for column in time_columns:
        table[column] = table[column] + pd.to_timedelta(copied["shift_s"], unit="s")
        runs[column] = runs[column] + copied["shift_s"]
    table["incident_id"] = copied["incident_id"].astype(dispatches["incident_id"].dtype)
    runs["incident_id"] = copied["incident_id"].to_numpy()
    copied_placed = copied["location"].to_numpy() >= 0
    if copied_placed.any():                                      # placed calls take the block's position, agency side and city flag
        block = locations.loc[copied.loc[copied_placed, "location"]]
        for column, source in [("incident_lon", "lon"), ("incident_lat", "lat"), ("jurisdiction", "jurisdiction"), ("focal", "focal")]:
            table.loc[copied_placed, column] = pd.Series(block[source].to_numpy(), index=table.index[copied_placed], dtype=dispatches[column].dtype)
    large = copied["large_fire"].to_numpy(dtype=bool)
    if large.any():                                              # every engine of a large fire arrives and stays on scene for LARGE_FIRE_ON_SCENE_S
        table.loc[large, "cleared_at"] = table.loc[large, "arrived_at"] + pd.Timedelta(seconds=LARGE_FIRE_ON_SCENE_S)
        runs.loc[large, "cleared_at"] = runs.loc[large, "arrived_at"] + LARGE_FIRE_ON_SCENE_S
        table.loc[large, "on_scene_s"] = LARGE_FIRE_ON_SCENE_S
        table.loc[large, "n_engines_sent"] = LARGE_FIRE_ENGINES
        table.loc[large, "arrival_order"] = table[large].groupby("incident_id").cumcount().to_numpy() + 1
        table.loc[large, "cancelled"] = False
    if len(outages):                                             # stand-in runs: a real row is the template, so every column keeps the record's data type
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
    seconds_rows = world["network_seconds"].reindex(source_row.to_numpy()).set_axis(table.index)
    seconds_rows.loc[placed] = location_seconds.loc[location[placed], seconds_rows.columns].to_numpy()
    nodes = world["call_nodes"].reindex(source_row.to_numpy()).set_axis(table.index)
    nodes.loc[placed, "node"] = locations.loc[location[placed], "node"].to_numpy()
    nodes.loc[placed, "snap_m"] = locations.loc[location[placed], "snap_m"].to_numpy()
    run_number = runs.groupby("incident_id", sort=False).cumcount().to_numpy()
    spread_quantile = [np.random.default_rng([seed, zlib.crc32(str(incident).encode())]).random(number + 1)[number]
                       for incident, number in zip(runs["incident_id"], run_number)]
    travel = apply_agency_boundaries(estimate_network_travel_seconds(table, engines, seconds_rows, world["travel_model"], verbose=False),
                                     table, engines, world["aid_protocols"], verbose=False)
    scenario = {"dispatches": table, "runs": runs, "source_row": source_row,
                "large_fire": pd.Series(copied["large_fire"].to_numpy(dtype=bool)[order], index=table.index), "travel": travel,
                "call_nodes": nodes, "spread_quantile": pd.Series(spread_quantile, index=table.index), "fates": call_fates(runs),
                **run_attributes(table, world["quint_penalty_by_type"])}
    if verbose:
        show_table(f"scenario dispatch table (seed {seed})", table, verbose)
        stand_in_row = table["protocol"].eq(OUT_OF_SERVICE_PROTOCOL).fillna(False)
        copies = (source_row.to_numpy() >= 0) & ~placed
        record = dispatches.loc[source_row[copies & ~scenario["large_fire"].to_numpy()]]
        kept = table[copies & ~scenario["large_fire"].to_numpy()]
        duration_change = ((kept["cleared_at"] - kept["dispatched_at"]).to_numpy() - (record["cleared_at"] - record["dispatched_at"]).to_numpy())
        record_travel = apply_agency_boundaries(estimate_network_travel_seconds(dispatches.loc[source_row[copies]], engines,
                                                                                world["network_seconds"].loc[source_row[copies]],
                                                                                world["travel_model"], verbose=False),
                                                dispatches.loc[source_row[copies]], engines, world["aid_protocols"], verbose=False)
        service_changed = (travel[copies].isna().to_numpy() != record_travel.isna().to_numpy()).any(axis=1)
        print(f"calls {table['incident_id'].nunique():,} ({calls['kind'].value_counts().to_dict()}), runs {len(table):,}, stand-in runs "
              f"{int(stand_in_row.sum())}, large-fire runs {int(scenario['large_fire'].sum())}, runs placed at a residential block {int(placed.sum())}\n"
              f"checks: copied rows all from the training months: {bool((dispatches.loc[source_row[copies], 'call_received_at'] < TRAINING_MONTHS_END).all())}; "
              f"largest change of a copied run's dispatch-to-clear time: {np.nanmax(np.abs(duration_change.astype('timedelta64[s]').astype(float))) if len(kept) else 0:.0f} s; "
              f"copied runs whose engines in service differ from their source's (moved across an engine's first or last day): {int(service_changed.sum())}\n"
              f"multi-run calls with call fates: {len(scenario['fates']):,}; low-acuity runs {int(scenario['low_acuity'].sum()):,}\n")
    return scenario
