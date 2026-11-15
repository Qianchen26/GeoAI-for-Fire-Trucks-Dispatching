"""Build the FY2025 engine dispatch dataset for the Gainesville–Alachua dispatch simulation.

One row = one engine (E or Q unit) dispatched to one incident: what the incident was, how urgent,
where it was, where the engine started, and how long it took to get rolling, drive, and finish.

Inputs (read only):
    data/raw/gnv_alachua/acfr_gfr_unit_responses_fy2025.parquet    county unit responses, FY2025
    data/raw/gnv_alachua/acfr_gfr_automatic_aid_fy2025.parquet     automatic-aid responses, FY2025
    data/raw/gnv_alachua/gfr_stations_2022.gpkg                    GFR stations
    data/raw/gnv_alachua/acdps_fire_ems_stations_2025.gpkg         county fire and EMS stations
    data/raw/gnv_alachua/gainesville_city_limits_2026.gpkg         Gainesville city limits
    data/derived/gfr_soc_2024/dashboard/appendix_e_crosswalk.csv   GFR SOC 2024 Appendix E (code -> risk level)

Outputs (data/derived/engine_dispatches_fy2025/):
    engine_dispatches.parquet   one row per engine dispatch
    engines.csv                 one row per engine
    stations.csv                one row per fire station

Run from the project root:
    python code/data_collection/build_engine_dispatches_fy2025.py
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]                       # [DESIGN] this file lives in code/data_collection/
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gnv_alachua"                  # [DESIGN]
RESPONSES_PARQUET = RAW_DIR / "acfr_gfr_unit_responses_fy2025.parquet"   # [SOURCE] county FY2025 unit responses
AUTO_AID_PARQUET = RAW_DIR / "acfr_gfr_automatic_aid_fy2025.parquet"     # [SOURCE] FY2025 automatic aid
GFR_STATIONS_GPKG = RAW_DIR / "gfr_stations_2022.gpkg"                   # [SOURCE] GFR stations 1-5, 7-9
COUNTY_STATIONS_GPKG = RAW_DIR / "acdps_fire_ems_stations_2025.gpkg"     # [SOURCE] county stations, incl. GFR station 6
CITY_LIMITS_GPKG = RAW_DIR / "gainesville_city_limits_2026.gpkg"         # [SOURCE] Gainesville city limits
SOC_CROSSWALK_CSV = PROJECT_ROOT / "data" / "derived" / "gfr_soc_2024" / "dashboard" / "appendix_e_crosswalk.csv"  # [SOURCE] SOC Appendix E
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived" / "engine_dispatches_fy2025"  # [DESIGN]

ENGINE_UNIT_PATTERN = r"^(E|Q)\d+$"        # [DESIGN] engine-class units: engines (E) and quints (Q)
GFR_MAX_ENGINE_NUMBER = 9                  # [SOURCE] GFR stations are numbered 1-9, so E1-E9 and Q1-Q9 are GFR units
COORDINATE_SCALE = 1e6                     # [SOURCE] integer coordinate columns are degrees x 1e6 (README)
FIRE_STATION_TYPES = ["Fire Response", "Fire/EMS Response"]  # [SOURCE] county layer types that house fire units
METRIC_CRS = "EPSG:26917"                  # [STANDARD] NAD83 / UTM zone 17N in metres
AT_STATION_RADIUS_M = 150.0                # [DESIGN] an engine within this distance of a station counts as at that station
HOME_GRID_M = 100.0                        # [DESIGN] grid used to find each engine's most frequent dispatch position
DST_SWITCHES_LOCAL = ["2024-11-03 02:00", "2025-03-09 02:00"]  # [STANDARD] America/New_York clock changes in FY2025
DST_MARGIN = pd.Timedelta(hours=1)         # [STANDARD] wall-clock intervals touching this window may be off by one hour
MAX_PLAUSIBLE_SPEED_KMH = 130.0            # [DESIGN] straight-line distance / travel time above this cannot be driven
MAX_PLAUSIBLE_DISTANCE_KM = 50.0           # [DESIGN] the county is about 60 km across; farther engine positions are errors
PRIORITY_WORDS = {"low": "Low", "mod": "Moderate", "moderate": "Moderate", "high": "High", "special": "Special"}  # [DESIGN] first level word in an Appendix E entry
FALLBACK_CALL_TYPES = {"E": "Medical", "F": "Fire"}  # [DESIGN] call type by protocol letter when Appendix E has no match
VERBOSE = True                             # [DESIGN] print every loaded and derived table


# ---------------------------------------------------------------------------------------------
# Reading and small helpers
# ---------------------------------------------------------------------------------------------
def show_table(name: str, table: pd.DataFrame, verbose: bool = VERBOSE) -> None:
    """Print a table's shape and first five rows.

    Example:
        >>> show_table("demo", pd.DataFrame({"a": [1, 2]}))   # prints "demo (2, 1)" and the rows; returns None
    """
    if verbose:
        print(f"--- {name} {table.shape}")
        print(table.head(5).to_string(), "\n")


def parse_local_time(column: pd.Series) -> pd.Series:
    """Parse source timestamps: they carry a +00:00 label but are Gainesville wall-clock time, so keep the clock.

    Example:
        >>> parse_local_time(pd.Series(["2024-10-01T00:18:45+00:00", None])).tolist()
        [Timestamp('2024-10-01 00:18:45'), NaT]
    """
    return pd.to_datetime(column, format="ISO8601", utc=True, errors="coerce").dt.tz_localize(None)


def seconds_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Whole seconds from one time column to another; missing when either is missing.

    Example:
        >>> seconds_between(pd.Series(pd.to_datetime(["2024-10-01 00:19:32"])),
        ...                 pd.Series(pd.to_datetime(["2024-10-01 00:18:45"]))).tolist()   # [47]: list[int]
    """
    return (later - earlier).dt.total_seconds().round().astype("Int64")


def read_engine_rows(verbose: bool = VERBOSE) -> pd.DataFrame:
    """Read the county FY2025 unit responses and keep the engine-class rows (E and Q units).

    Example:
        >>> engine_rows = read_engine_rows(verbose=False)
        >>> engine_rows.shape, engine_rows["Unit"].iloc[0]       # ((37194, 24), 'E62')
    """
    responses = pd.read_parquet(RESPONSES_PARQUET)
    show_table("raw unit responses", responses, verbose)
    engine_rows = responses[responses["Unit"].str.match(ENGINE_UNIT_PATTERN)].reset_index(drop=True)
    show_table("engine rows (E and Q units)", engine_rows, verbose)
    return engine_rows


# ---------------------------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------------------------
def build_station_table(verbose: bool = VERBOSE) -> pd.DataFrame:
    """One row per fire station: GFR stations from the GFR layer, all other fire stations from the county layer.

    Station numbers repeat across agencies (GFR 3 and county 3), so ids carry a prefix: GFR_03, CO_21, CO_31S.

    Example:
        >>> stations = build_station_table(verbose=False)
        >>> stations[["station_id", "name"]].iloc[0].tolist()    # ['CO_12', 'Fire Station 12 (Mobile DRC)']
    """
    gfr = gpd.read_file(GFR_STATIONS_GPKG)
    show_table("GFR station layer", pd.DataFrame(gfr.drop(columns="geometry")), verbose)
    gfr_rows = pd.DataFrame({"station_id": "GFR_" + gfr["FACILITYID"].str.strip().str.zfill(2), "name": gfr["NAME"],
                             "listed_jurisdiction": "Gainesville", "lon": gfr.geometry.x, "lat": gfr.geometry.y,
                             "source_layer": GFR_STATIONS_GPKG.name})

    county = gpd.read_file(COUNTY_STATIONS_GPKG)
    show_table("county station layer", pd.DataFrame(county.drop(columns="geometry")), verbose)
    county = county[county["type"].isin(FIRE_STATION_TYPES)].reset_index(drop=True)
    number = county["name"].str.extract(r"Station (\d+)", expand=False).str.zfill(2)
    prefix = np.where(county["jurisdicti"].eq("Gainesville"), "GFR_", "CO_")
    suffix = np.where(county["name"].str.contains("Substation"), "S", "")
    county_rows = pd.DataFrame({"station_id": prefix + number + suffix, "name": county["name"],
                                "listed_jurisdiction": county["jurisdicti"], "lon": county.geometry.x,
                                "lat": county.geometry.y, "source_layer": COUNTY_STATIONS_GPKG.name})
    # The county layer repeats most GFR stations; keep its copy only where the GFR layer has none (station 6, airport).
    county_rows = county_rows[~county_rows["station_id"].isin(gfr_rows["station_id"])]

    stations = pd.concat([gfr_rows, county_rows]).sort_values("station_id").reset_index(drop=True)
    assert stations["station_id"].is_unique, "station ids must be unique"
    show_table("stations", stations, verbose)
    return stations


def to_metric_points(lon: pd.Series, lat: pd.Series) -> gpd.GeoSeries:
    """Longitude/latitude columns as projected points in metres (missing coordinates stay empty).

    Example:
        >>> to_metric_points(pd.Series([-82.32]), pd.Series([29.65])).x.round().tolist()   # [372240.0]: list[float]
    """
    return gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326", index=lon.index).to_crs(METRIC_CRS)


def nearest_station_within(lon: pd.Series, lat: pd.Series, stations: pd.DataFrame, radius_m: float) -> pd.Series:
    """Id of the nearest station within radius_m of each point; missing when none is that close or the point is missing.

    Example:
        >>> nearest_station_within(pd.Series([-82.3245]), pd.Series([29.6475]), stations, 150.0).tolist()   # ['GFR_01']
    """
    usable = lon.notna() & lat.notna()
    points = gpd.GeoDataFrame(geometry=to_metric_points(lon[usable], lat[usable]))
    station_points = gpd.GeoDataFrame(stations[["station_id"]],
                                      geometry=to_metric_points(stations["lon"], stations["lat"]).values)
    joined = gpd.sjoin_nearest(points, station_points, how="left", max_distance=radius_m)
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["station_id"].reindex(lon.index).astype("string")


# ---------------------------------------------------------------------------------------------
# Call type and priority
# ---------------------------------------------------------------------------------------------
def split_problem_code(problem: pd.Series) -> pd.DataFrame:
    """Split the CAD problem text into protocol (card number), determinant level letter and full determinant code.

    Example:
        >>> split_problem_code(pd.Series(["E17A4 - FALLS - LIFT ASSIST", "F52 Com Alarm/", "HCF TRANSFER"])).values.tolist()
        [['E17', 'A', 'E17A4'], ['F52', <NA>, 'F52'], [<NA>, <NA>, <NA>]]
    """
    parts = problem.str.extract(r"^(?P<protocol>[EF]\d{2})(?P<determinant_level>[A-EO])?(?P<determinant_tail>\d*)")
    determinant_code = (parts["protocol"] + parts["determinant_level"].fillna("") + parts["determinant_tail"].fillna(""))
    return pd.DataFrame({"protocol": parts["protocol"].astype("string"),
                         "determinant_level": parts["determinant_level"].astype("string"),
                         "determinant_code": determinant_code.astype("string")})


def normalize_code_text(text: pd.Series) -> pd.Series:
    """Uppercase, trim, drop a trailing slash and squeeze spaces, so CAD text and Appendix E entries compare equal.

    Example:
        >>> normalize_code_text(pd.Series(["F52 Com Alarm/ ", "e06d"])).tolist()     # ['F52 COM ALARM', 'E06D']
    """
    return (text.astype("string").str.upper().str.strip().str.rstrip("/").str.strip()
            .str.replace(r"\s+", " ", regex=True))


def read_soc_priority_lookup(verbose: bool = VERBOSE) -> pd.DataFrame:
    """Appendix E of the GFR SOC 2024 as a lookup: normalized code -> call type and priority (Low/Moderate/High/Special).

    The priority is the first level word in the Appendix E entry, so "High/Special" becomes High and
    "High EMS D" (Delta and Echo calls) becomes High; the determinant letter keeps the finer detail.

    Example:
        >>> lookup = read_soc_priority_lookup(verbose=False)
        >>> lookup.loc["E10D"].tolist()          # ['Medical', 'High']
    """
    crosswalk = pd.read_csv(SOC_CROSSWALK_CSV)
    show_table("SOC Appendix E crosswalk", crosswalk, verbose)
    level_word = crosswalk["risk_level"].astype("string").str.lower().str.extract(
        r"(moderate|mod|low|high|special)", expand=False)
    lookup = pd.DataFrame({"code_key": normalize_code_text(crosswalk["code"]),
                           "call_type": crosswalk["risk_type"].astype("string"),
                           "priority": level_word.map(PRIORITY_WORDS).astype("string")})
    lookup = lookup.drop_duplicates("code_key").set_index("code_key")
    if verbose:
        print("priority mapping from Appendix E levels:")
        print(pd.crosstab(crosswalk["risk_level"].fillna("(blank)"), level_word.map(PRIORITY_WORDS).fillna("(none)")), "\n")
    return lookup


def assign_call_type_and_priority(problem: pd.Series, codes: pd.DataFrame, lookup: pd.DataFrame,
                                  verbose: bool = VERBOSE) -> pd.DataFrame:
    """Match each problem to Appendix E: full text first, then the determinant code, code + level, protocol alone.

    Unmatched rows get a call type from the protocol letter (E Medical, F Fire; transfers Medical) and no priority.

    Example:
        >>> assign_call_type_and_priority(pd.Series(["E10D - CHEST PAIN"]), split_problem_code(pd.Series(["E10D - CHEST PAIN"])),
        ...                               lookup, verbose=False).values.tolist()
        [['Medical', 'High', True]]
    """
    candidates = [normalize_code_text(problem.str.split(" - ").str[0]), normalize_code_text(problem),
                  codes["determinant_code"], codes["protocol"] + codes["determinant_level"].fillna(""), codes["protocol"]]
    matched_key = pd.Series(pd.NA, index=problem.index, dtype="string")
    for candidate in candidates:
        found = matched_key.isna() & candidate.isin(lookup.index)
        matched_key[found] = candidate[found]
    matched = matched_key.notna()
    call_type = matched_key.map(lookup["call_type"]).astype("string")
    fallback = codes["protocol"].str[0].map(FALLBACK_CALL_TYPES).astype("string")
    fallback[problem.str.contains("TRANSFER", na=False)] = "Medical"
    call_type = call_type.fillna(fallback)
    priority = matched_key.map(lookup["priority"]).astype("string")
    if verbose:
        print(f"Appendix E match: {matched.mean():.1%} of rows; with a priority: {priority.notna().mean():.1%}")
        print("unmatched problems (top 10):", problem[~matched].value_counts().head(10).to_dict(), "\n")
    return pd.DataFrame({"call_type": call_type, "priority": priority, "soc_matched": matched})


# ---------------------------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------------------------
def locate_engine_homes(engine_id: pd.Series, lon: pd.Series, lat: pd.Series, verbose: bool = VERBOSE) -> pd.DataFrame:
    """Each engine's home position: the mean of its dispatch positions in its most frequent 100 m grid cell.

    Example:
        >>> homes = locate_engine_homes(dispatches["engine_id"], dispatches["engine_lon_at_dispatch"],
        ...                             dispatches["engine_lat_at_dispatch"], verbose=False)
        >>> homes.columns.tolist()               # ['engine_id', 'home_lon', 'home_lat']
    """
    usable = lon.notna() & lat.notna()
    points = to_metric_points(lon[usable], lat[usable])
    positions = pd.DataFrame({"engine_id": engine_id[usable], "x": points.x, "y": points.y,
                              "cell_x": (points.x // HOME_GRID_M).astype(int), "cell_y": (points.y // HOME_GRID_M).astype(int)})
    cell_counts = positions.groupby(["engine_id", "cell_x", "cell_y"]).size().rename("dispatches").reset_index()
    modal_cells = cell_counts.sort_values("dispatches", ascending=False).drop_duplicates("engine_id")
    in_modal_cell = positions.merge(modal_cells[["engine_id", "cell_x", "cell_y"]], on=["engine_id", "cell_x", "cell_y"])
    centres = in_modal_cell.groupby("engine_id")[["x", "y"]].mean()
    centre_points = gpd.GeoSeries(gpd.points_from_xy(centres["x"], centres["y"]), crs=METRIC_CRS).to_crs("EPSG:4326")
    homes = pd.DataFrame({"engine_id": centres.index, "home_lon": centre_points.x.values, "home_lat": centre_points.y.values})
    show_table("engine home positions", homes, verbose)
    return homes


def build_engine_table(dispatches: pd.DataFrame, stations: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """One row per engine: type, agency (GFR by number 1-9), home position and station, activity span.

    Example:
        >>> engines = build_engine_table(dispatches, stations, verbose=False)
        >>> engines.loc[engines.engine_id == "E3", ["engine_agency", "home_station"]].values.tolist()   # [['GFR', 'GFR_03']]
    """
    homes = locate_engine_homes(dispatches["engine_id"], dispatches["engine_lon_at_dispatch"],
                                dispatches["engine_lat_at_dispatch"], verbose)
    homes["home_station"] = nearest_station_within(homes["home_lon"], homes["home_lat"], stations, AT_STATION_RADIUS_M)
    activity = dispatches.groupby("engine_id").agg(n_dispatches=("incident_id", "size"),
                                                   first_seen=("dispatched_at", "min"),
                                                   last_seen=("dispatched_at", "max")).reset_index()
    engines = activity.merge(homes, on="engine_id", how="left")
    number = engines["engine_id"].str.extract(r"(\d+)$", expand=False).astype(int)
    engines.insert(1, "engine_type", engines["engine_id"].str[0])
    engines.insert(2, "engine_agency", np.where(number <= GFR_MAX_ENGINE_NUMBER, "GFR", "non-GFR"))
    engines = (engines.assign(engine_number=number).sort_values(["engine_agency", "engine_number", "engine_type"])
               .drop(columns="engine_number").reset_index(drop=True))
    show_table("engines", engines, verbose)
    return engines


# ---------------------------------------------------------------------------------------------
# The dispatch table
# ---------------------------------------------------------------------------------------------
def build_dispatch_table(engine_rows: pd.DataFrame, stations: pd.DataFrame, lookup: pd.DataFrame,
                         verbose: bool = VERBOSE) -> pd.DataFrame:
    """One row per engine dispatch with the agreed columns (engine agency and at_home are added later).

    Example:
        >>> dispatches = build_dispatch_table(engine_rows, stations, lookup, verbose=False)
        >>> dispatches.loc[0, ["incident_id", "engine_id", "travel_s"]].tolist()    # ['100124-0043', 'E62', 397]
    """
    times = {name: parse_local_time(engine_rows[source]) for name, source in [
        ("call_received_at", "Time_FirstCallTakingKeystroke"), ("dispatched_at", "Time_Assigned"),
        ("enroute_at", "Time_Enroute"), ("arrived_at", "Time_Arrived_At_Scene"), ("cleared_at", "Time_Call_Cleared")]}
    codes = split_problem_code(engine_rows["Problem"])
    calls = assign_call_type_and_priority(engine_rows["Problem"], codes, lookup, verbose)

    incident_lat = engine_rows["GIS_LAT"].where(engine_rows["GIS_LAT"] != 0)
    incident_lon = engine_rows["GIS_LONG"].where(engine_rows["GIS_LONG"] != 0)
    engine_lat = (engine_rows["Latitude_At_Assign_Time"] / COORDINATE_SCALE).where(engine_rows["Latitude_At_Assign_Time"] != 0)
    engine_lon = (-engine_rows["Longitude_At_Assign_Time"].abs() / COORDINATE_SCALE).where(engine_rows["Longitude_At_Assign_Time"] != 0)

    city = gpd.read_file(CITY_LIMITS_GPKG).to_crs(METRIC_CRS).union_all()
    usable = incident_lon.notna()
    focal = pd.Series(pd.NA, index=engine_rows.index, dtype="boolean")
    focal[usable] = to_metric_points(incident_lon[usable], incident_lat[usable]).within(city).to_numpy()

    auto_aid = pd.read_parquet(AUTO_AID_PARQUET)
    show_table("FY2025 automatic aid", auto_aid, verbose)
    aid_pairs = set(zip(auto_aid["Master_Incident_Number"], auto_aid["Radio_Name"]))

    dispatches = pd.DataFrame({
        "incident_id": engine_rows["Incident_Number"].astype("string"),
        "call_received_at": times["call_received_at"],
        "problem": engine_rows["Problem"].astype("string"),
        "call_type": calls["call_type"],
        "protocol": codes["protocol"],
        "determinant_level": codes["determinant_level"],
        "determinant_code": codes["determinant_code"],
        "priority": calls["priority"],
        "n_engines_sent": engine_rows.groupby("Incident_Number")["Unit"].transform("size").astype("Int64"),
        "incident_lon": incident_lon,
        "incident_lat": incident_lat,
        "jurisdiction": engine_rows["Jurisdiction"].astype("string"),
        "focal": focal,
        "engine_id": engine_rows["Unit"].astype("string"),
        "engine_type": engine_rows["Unit"].str[0].astype("string"),
        "engine_lon_at_dispatch": engine_lon,
        "engine_lat_at_dispatch": engine_lat,
        "from_station": nearest_station_within(engine_lon, engine_lat, stations, AT_STATION_RADIUS_M),
        "dispatched_at": times["dispatched_at"],
        "enroute_at": times["enroute_at"],
        "arrived_at": times["arrived_at"],
        "cleared_at": times["cleared_at"],
        "turnout_s": seconds_between(times["enroute_at"], times["dispatched_at"]),
        "travel_s": seconds_between(times["arrived_at"], times["enroute_at"]),
        "on_scene_s": seconds_between(times["cleared_at"], times["arrived_at"]),
        "staged": engine_rows["Time_Staged"].notna().astype("boolean"),
        "cancelled": times["arrived_at"].isna().astype("boolean"),
        "auto_aid": pd.Series([pair in aid_pairs for pair in zip(engine_rows["Incident_Number"], engine_rows["Unit"])],
                              index=engine_rows.index, dtype="boolean"),
    })
    dispatches.insert(dispatches.columns.get_loc("on_scene_s") + 1, "arrival_order",
                      dispatches.groupby("incident_id")["arrived_at"].rank(method="first").astype("Int64"))
    dispatches["qc_note"] = build_quality_notes(dispatches, calls["soc_matched"])
    show_table("engine dispatches (before engine agency and at_home)", dispatches, verbose)
    return dispatches


def build_quality_notes(dispatches: pd.DataFrame, soc_matched: pd.Series) -> pd.Series:
    """Semicolon-separated notes on data problems in each row; empty when the row has none.

    Example:
        >>> build_quality_notes(dispatches, soc_matched).value_counts().head(2)   # '' for most rows, then the commonest note
    """
    switches = pd.to_datetime(DST_SWITCHES_LOCAL)
    first_time = dispatches[["dispatched_at", "enroute_at", "arrived_at", "cleared_at"]].min(axis=1)
    last_time = dispatches[["dispatched_at", "enroute_at", "arrived_at", "cleared_at"]].max(axis=1)
    near_switch = pd.Series(False, index=dispatches.index)
    for switch in switches:
        near_switch |= (first_time <= switch + DST_MARGIN) & (last_time >= switch - DST_MARGIN)
    distance_km = to_metric_points(dispatches["engine_lon_at_dispatch"], dispatches["engine_lat_at_dispatch"]).distance(
        to_metric_points(dispatches["incident_lon"], dispatches["incident_lat"])) / 1000
    implied_speed_kmh = distance_km / (dispatches["travel_s"].astype("Float64") / 3600)
    checks = {
        "incident_coordinates_missing": dispatches["incident_lon"].isna(),
        "engine_position_missing": dispatches["engine_lon_at_dispatch"].isna(),
        "engine_position_or_travel_implausible": ((implied_speed_kmh > MAX_PLAUSIBLE_SPEED_KMH).fillna(False)
                                                  | (distance_km > MAX_PLAUSIBLE_DISTANCE_KM)),
        "engine_position_equals_incident": (dispatches["engine_lon_at_dispatch"].round(6).eq(dispatches["incident_lon"].round(6))
                                            & dispatches["engine_lat_at_dispatch"].round(6).eq(dispatches["incident_lat"].round(6))).fillna(False),
        "negative_turnout": dispatches["turnout_s"].lt(0).fillna(False),
        "negative_travel": dispatches["travel_s"].lt(0).fillna(False),
        "negative_on_scene": dispatches["on_scene_s"].lt(0).fillna(False),
        "near_dst_change": near_switch,
        "priority_not_in_soc_appendix_e": ~soc_matched,
    }
    notes = pd.Series("", index=dispatches.index, dtype="string")
    for label, flagged in checks.items():
        flagged = flagged.astype(bool)
        notes[flagged] = (notes[flagged] + ";" + label).str.lstrip(";")
    return notes


def add_engine_columns(dispatches: pd.DataFrame, engines: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """Add engine_agency and at_home (dispatched within 150 m of the engine's own home position).

    Example:
        >>> final = add_engine_columns(dispatches, engines, verbose=False)
        >>> final[["engine_agency", "at_home"]].iloc[0].tolist()      # ['non-GFR', True]
    """
    with_home = dispatches.merge(engines[["engine_id", "engine_agency", "home_lon", "home_lat"]], on="engine_id", how="left")
    distance = to_metric_points(with_home["engine_lon_at_dispatch"], with_home["engine_lat_at_dispatch"]).distance(
        to_metric_points(with_home["home_lon"], with_home["home_lat"]))
    at_home = pd.Series(distance <= AT_STATION_RADIUS_M, dtype="boolean")
    at_home[with_home["engine_lon_at_dispatch"].isna()] = pd.NA
    final = dispatches.copy()
    final.insert(final.columns.get_loc("engine_type") + 1, "engine_agency", with_home["engine_agency"].astype("string").values)
    final.insert(final.columns.get_loc("from_station") + 1, "at_home", at_home.values)
    show_table("engine dispatches (final)", final, verbose)
    return final


# ---------------------------------------------------------------------------------------------
# Checks and writing
# ---------------------------------------------------------------------------------------------
def check_dispatch_table(dispatches: pd.DataFrame, engine_rows: pd.DataFrame, verbose: bool = VERBOSE) -> None:
    """Assert the invariants the simulation relies on and print the main counts.

    Example:
        >>> check_dispatch_table(final, engine_rows)   # raises AssertionError if an invariant breaks; returns None
    """
    assert len(dispatches) == len(engine_rows), "one row per engine response"
    per_incident = dispatches.groupby("incident_id")[["call_received_at", "incident_lon", "jurisdiction", "problem"]].nunique(dropna=False)
    assert (per_incident <= 1).all().all(), "incident fields must agree across the rows of an incident"
    both = dispatches["turnout_s"].notna() & dispatches["travel_s"].notna()
    total = seconds_between(dispatches["arrived_at"], dispatches["dispatched_at"])
    assert (dispatches.loc[both, "turnout_s"] + dispatches.loc[both, "travel_s"] == total[both]).all(), "turnout + travel = dispatch to arrival"
    assert dispatches.loc[dispatches["cancelled"], "arrival_order"].isna().all(), "cancelled rows have no arrival order"
    if verbose:
        print(f"rows {len(dispatches):,} | incidents {dispatches['incident_id'].nunique():,} | engines {dispatches['engine_id'].nunique()}")
        print("priority:", dispatches["priority"].value_counts(dropna=False).to_dict())
        print("call type:", dispatches["call_type"].value_counts(dropna=False).to_dict())
        print("determinant level:", dispatches["determinant_level"].value_counts(dropna=False).to_dict())
        print(f"from a station: {dispatches['from_station'].notna().mean():.1%} | at home: {dispatches['at_home'].mean():.1%}"
              f" | cancelled: {dispatches['cancelled'].mean():.1%} | staged: {dispatches['staged'].mean():.2%}"
              f" | auto aid: {int(dispatches['auto_aid'].sum())} | focal: {dispatches['focal'].mean():.1%}")
        notes = dispatches["qc_note"].str.split(";").explode()
        print("quality notes:", notes[notes != ""].value_counts().to_dict(), "\n")


def write_outputs(dispatches: pd.DataFrame, engines: pd.DataFrame, stations: pd.DataFrame, verbose: bool = VERBOSE) -> None:
    """Write the three tables to OUTPUT_DIR.

    Example:
        >>> write_outputs(final, engines, stations, verbose=False)   # writes engine_dispatches.parquet, engines.csv, stations.csv
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dispatches.to_parquet(OUTPUT_DIR / "engine_dispatches.parquet", index=False)
    engines.to_csv(OUTPUT_DIR / "engines.csv", index=False)
    stations.to_csv(OUTPUT_DIR / "stations.csv", index=False)
    if verbose:
        for path in sorted(OUTPUT_DIR.glob("*")):
            print(f"wrote {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1024:,.0f} KB)")


def main(verbose: bool = VERBOSE) -> None:
    """Build and write the engine dispatch dataset.

    Example:
        >>> main(verbose=False)     # writes data/derived/engine_dispatches_fy2025/*
    """
    engine_rows = read_engine_rows(verbose)
    stations = build_station_table(verbose)
    lookup = read_soc_priority_lookup(verbose)
    dispatches = build_dispatch_table(engine_rows, stations, lookup, verbose)
    engines = build_engine_table(dispatches, stations, verbose)
    final = add_engine_columns(dispatches, engines, verbose)
    check_dispatch_table(final, engine_rows, verbose)
    write_outputs(final, engines, stations, verbose)


if __name__ == "__main__":
    main()
