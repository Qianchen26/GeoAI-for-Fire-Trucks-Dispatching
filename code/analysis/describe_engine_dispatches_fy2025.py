"""Descriptive statistics of the FY2025 engine dispatch dataset, plus the number of engines sent per incident.

Input:
    data/derived/engine_dispatches_fy2025/engine_dispatches.parquet   one row per engine dispatch
    data/raw/gnv_alachua/alachua_county_boundary.gpkg                 map extent for the location grids

Outputs (data/reports/engine_dispatches_fy2025/):
    overview.csv                       one row: dispatches, incidents, engines, period, travel median and P90
    numeric_summary.csv                count, missing, mean, sd, min, quartiles, P90, max, skewness of numeric columns
    histogram_bins.csv                 histogram bins of turnout, travel and on-scene seconds
    category_counts.csv                count and share of every value of the categorical and yes/no columns
    time_profile.csv                   dispatches and incidents by month, hour of day and weekday
    durations_by_group.csv             turnout, travel and on-scene median and P90 by engine agency and by focal area
    location_grid.csv                  incidents and engine dispatch positions counted on a 0.01-degree grid
    engines_per_incident.csv           incidents by number of engines sent and number that arrived
    engines_per_incident_by_group.csv  engines per incident by call type, priority, focal area and protocol
    response_curves.csv                share of incidents that reached each stage within t minutes of the call

Identifier columns (incident_id, engine_id) are not summarized.

Run from the project root:
    python code/analysis/describe_engine_dispatches_fy2025.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]                                   # [DESIGN] this file lives in code/analysis/
DISPATCHES_PARQUET = PROJECT_ROOT / "data" / "derived" / "engine_dispatches_fy2025" / "engine_dispatches.parquet"  # [DESIGN]
COUNTY_BOUNDARY_GPKG = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "alachua_county_boundary.gpkg"  # [SOURCE] map extent
OUTPUT_DIR = PROJECT_ROOT / "data" / "reports" / "engine_dispatches_fy2025"          # [DESIGN]
NUMERIC_COLUMNS = ["turnout_s", "travel_s", "on_scene_s", "n_engines_sent", "arrival_order",
                   "incident_lon", "incident_lat", "engine_lon_at_dispatch", "engine_lat_at_dispatch"]  # [DESIGN]
CATEGORY_COLUMNS = ["call_type", "priority", "determinant_level", "protocol", "jurisdiction", "focal", "engine_type",
                    "engine_agency", "from_station", "at_home", "arrival_order", "staged", "cancelled", "auto_aid"]  # [DESIGN] no id columns
DURATION_COLUMNS = ["turnout_s", "travel_s", "on_scene_s"]   # [DESIGN] the three intervals the simulation uses
QUANTILES = [0.25, 0.5, 0.75, 0.9]                           # [STANDARD] quartiles plus P90, the fire-service benchmark
HISTOGRAM_BINS = 30                                          # [DESIGN] equal-width bins per duration histogram
HISTOGRAM_LOW_QUANTILE = 0.005                               # [DESIGN] values below this quantile form one underflow bar
HISTOGRAM_HIGH_QUANTILE = 0.995                              # [DESIGN] values above this quantile form one overflow bar
GRID_DEGREES = 0.01                                          # [DESIGN] map cell, about 1.1 km north-south and 1 km east-west
MAP_MARGIN_DEGREES = 0.02                                    # [DESIGN] margin around the county bounding box
ENGINE_COUNT_TOP = 5                                         # [DESIGN] incidents with this many engines or more share one bucket ("5+")
MIN_INCIDENTS_PER_PROTOCOL = 200                             # [DESIGN] protocols with fewer incidents are not listed separately
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]  # [STANDARD]
DURATION_GROUPS = [("engine_agency", "GFR", "GFR engines"), ("engine_agency", "non-GFR", "non-GFR engines"),
                   ("focal", True, "incident inside city"), ("focal", False, "incident outside city")]  # [DESIGN]
RESPONSE_GRID_MINUTES = np.round(np.arange(0, 20.0001, 0.1), 1)   # [DESIGN] response curves: 0-20 min after the call, every 6 s
COMPLETION_GRID_MINUTES = np.arange(0, 241, 1.0)                   # [DESIGN] completion curves: 0-4 h after the call, every minute
SOC_TARGETS_SECONDS = {"urban": 410, "rural": 770}                 # [SOURCE] GFR SOC 2024 first-due total response P90: 6:50 and 12:50
VERBOSE = True                                               # [DESIGN] print every derived table


def show_table(name: str, table: pd.DataFrame, verbose: bool = VERBOSE, rows: int = 5) -> None:
    """Print a table's shape and first rows.

    Example:
        >>> show_table("demo", pd.DataFrame({"a": [1, 2]}))   # prints "demo (2, 1)" and the rows; returns None
    """
    if verbose:
        print(f"--- {name} {table.shape}")
        print(table.head(rows).to_string(), "\n")


def read_dispatches(verbose: bool = VERBOSE) -> pd.DataFrame:
    """Read the engine dispatch table.

    Example:
        >>> dispatches = read_dispatches(verbose=False)
        >>> dispatches.shape                       # (37194, 32): tuple[int, int]
    """
    dispatches = pd.read_parquet(DISPATCHES_PARQUET)
    show_table("engine dispatches", dispatches, verbose)
    return dispatches


# ---------------------------------------------------------------------------------------------
# Conventional descriptive statistics
# ---------------------------------------------------------------------------------------------
def summarize_numeric(dispatches: pd.DataFrame, columns: list[str], verbose: bool = VERBOSE) -> pd.DataFrame:
    """Count, missing, mean, sd, min, quartiles, P90, max and skewness of each numeric column.

    Example:
        >>> summarize_numeric(pd.DataFrame({"x": [1, 2, 3, None]}), ["x"], verbose=False).loc[0, ["count", "missing", "p50"]].tolist()
        [3, 1, 2.0]
    """
    rows = []
    for column in columns:
        values = dispatches[column].astype("Float64").dropna().astype(float)
        quantiles = values.quantile(QUANTILES)
        rows.append({"column": column, "count": int(values.size), "missing": int(dispatches[column].isna().sum()),
                     "mean": values.mean(), "sd": values.std(), "min": values.min(),
                     **{f"p{round(q * 100)}": quantiles[q] for q in QUANTILES},
                     "max": values.max(), "skewness": values.skew()})
    summary = pd.DataFrame(rows)
    show_table("numeric summary", summary, verbose, rows=len(summary))
    return summary


def bin_durations(dispatches: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """Equal-width histogram bins between the 0.5% and 99.5% quantiles, plus one bar for each tail beyond them.

    Example:
        >>> bins = bin_durations(dispatches, verbose=False)
        >>> bins[bins.column == "travel_s"]["count"].sum()     # 30477: every travel value is in exactly one bar
    """
    pieces = []
    for column in DURATION_COLUMNS:
        values = dispatches[column].dropna().astype(float)
        low, high = values.quantile([HISTOGRAM_LOW_QUANTILE, HISTOGRAM_HIGH_QUANTILE]).tolist()
        edges = np.linspace(low, high, HISTOGRAM_BINS + 1)
        counts, _ = np.histogram(values[(values >= low) & (values <= high)], bins=edges)
        rows = [{"x0": left, "x1": right, "count": int(count), "tail": ""} for left, right, count in zip(edges[:-1], edges[1:], counts)]
        below, above = int((values < low).sum()), int((values > high).sum())
        if below:
            rows.insert(0, {"x0": float(values.min()), "x1": low, "count": below, "tail": "low"})
        if above:
            rows.append({"x0": high, "x1": float(values.max()), "count": above, "tail": "high"})
        table = pd.DataFrame(rows)
        table.insert(0, "i", range(len(table)))
        table.insert(0, "column", column)
        table["share"] = table["count"] / len(values)
        pieces.append(table)
    bins = pd.concat(pieces, ignore_index=True)
    assert all(bins[bins.column == column]["count"].sum() == dispatches[column].notna().sum() for column in DURATION_COLUMNS)
    show_table("histogram bins", bins, verbose)
    return bins


def count_categories(dispatches: pd.DataFrame, columns: list[str], verbose: bool = VERBOSE) -> pd.DataFrame:
    """Count and share of every value (missing shown as "(missing)") of each categorical or yes/no column.

    Example:
        >>> count_categories(pd.DataFrame({"c": ["a", "a", None]}), ["c"], verbose=False)[["value", "count"]].values.tolist()
        [['a', 2], ['(missing)', 1]]
    """
    pieces = []
    for column in columns:
        counts = dispatches[column].astype("string").fillna("(missing)").value_counts()
        pieces.append(pd.DataFrame({"column": column, "value": counts.index, "count": counts.values,
                                    "share": counts.values / len(dispatches)}))
    table = pd.concat(pieces, ignore_index=True)
    show_table("category counts", table, verbose)
    return table


def profile_over_time(dispatches: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """Dispatches (by dispatch time) and incidents (by call time) per month, hour of day and weekday.

    Example:
        >>> profile = profile_over_time(dispatches, verbose=False)
        >>> profile[profile.unit == "hour"].shape[0]      # 24: int
    """
    incidents = dispatches.drop_duplicates("incident_id")
    keys = {
        "month": (dispatches["dispatched_at"].dt.strftime("%Y-%m"), incidents["call_received_at"].dt.strftime("%Y-%m")),
        "hour": (dispatches["dispatched_at"].dt.hour, incidents["call_received_at"].dt.hour),
        "weekday": (dispatches["dispatched_at"].dt.dayofweek.map(dict(enumerate(WEEKDAY_NAMES))),
                    incidents["call_received_at"].dt.dayofweek.map(dict(enumerate(WEEKDAY_NAMES)))),
    }
    pieces = []
    for unit, (dispatch_key, incident_key) in keys.items():
        counts = pd.DataFrame({"dispatches": dispatch_key.value_counts(), "incidents": incident_key.value_counts()})
        order = WEEKDAY_NAMES if unit == "weekday" else sorted(counts.index)
        counts = counts.reindex(order).rename_axis("value").reset_index()
        counts.insert(0, "unit", unit)
        pieces.append(counts)
    profile = pd.concat(pieces, ignore_index=True)
    show_table("time profile", profile, verbose)
    return profile


def summarize_durations_by_group(dispatches: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """Median and P90 of turnout, travel and on-scene seconds for GFR and non-GFR engines and inside and outside the city.

    Example:
        >>> table = summarize_durations_by_group(dispatches, verbose=False)
        >>> table["group"].tolist()       # ['GFR engines', 'non-GFR engines', 'incident inside city', 'incident outside city']
    """
    rows = []
    for column, value, label in DURATION_GROUPS:
        subset = dispatches[dispatches[column] == value]
        row = {"group": label, "dispatches": len(subset)}
        for duration in DURATION_COLUMNS:
            row[f"{duration}_median"] = subset[duration].median()
            row[f"{duration}_p90"] = subset[duration].astype("Float64").quantile(0.9)
        rows.append(row)
    table = pd.DataFrame(rows)
    show_table("durations by group", table, verbose, rows=len(table))
    return table


def grid_locations(dispatches: pd.DataFrame, verbose: bool = VERBOSE) -> tuple[pd.DataFrame, dict[str, int]]:
    """Count incident locations and engine dispatch positions on a grid over the county; also count points off the map.

    Example:
        >>> grid, outside = grid_locations(dispatches, verbose=False)
        >>> sorted(grid["layer"].unique()), sorted(outside)    # (['engine', 'incident'], ['engine', 'incident'])
    """
    west, south, east, north = gpd.read_file(COUNTY_BOUNDARY_GPKG).total_bounds
    west, south, east, north = west - MAP_MARGIN_DEGREES, south - MAP_MARGIN_DEGREES, east + MAP_MARGIN_DEGREES, north + MAP_MARGIN_DEGREES
    incidents = dispatches.drop_duplicates("incident_id")
    layers = {"incident": (incidents["incident_lon"], incidents["incident_lat"]),
              "engine": (dispatches["engine_lon_at_dispatch"], dispatches["engine_lat_at_dispatch"])}
    pieces, outside = [], {}
    for layer, (lon, lat) in layers.items():
        present = lon.notna() & lat.notna()
        on_map = present & lon.between(west, east) & lat.between(south, north)
        outside[layer] = int((present & ~on_map).sum())
        cells = pd.DataFrame({"cell_x": ((lon[on_map] - west) // GRID_DEGREES).astype(int),
                              "cell_y": ((lat[on_map] - south) // GRID_DEGREES).astype(int)})
        counts = cells.value_counts().rename("count").reset_index()
        counts["lon"] = west + (counts["cell_x"] + 0.5) * GRID_DEGREES
        counts["lat"] = south + (counts["cell_y"] + 0.5) * GRID_DEGREES
        counts.insert(0, "layer", layer)
        pieces.append(counts[["layer", "lon", "lat", "count"]])
    grid = pd.concat(pieces, ignore_index=True)
    show_table("location grid", grid, verbose)
    if verbose:
        print("points off the map:", outside, "\n")
    return grid, outside


def summarize_overview(dispatches: pd.DataFrame, outside: dict[str, int], verbose: bool = VERBOSE) -> pd.DataFrame:
    """One-row table of headline counts, period, travel median and P90, and points off the map.

    Example:
        >>> summarize_overview(dispatches, {"incident": 0, "engine": 9}, verbose=False).loc[0, "engines"]   # 29
    """
    overview = pd.DataFrame([{
        "dispatches": len(dispatches), "incidents": dispatches["incident_id"].nunique(),
        "engines": dispatches["engine_id"].nunique(),
        "first_call": dispatches["call_received_at"].min().strftime("%Y-%m-%d"),
        "last_call": dispatches["call_received_at"].max().strftime("%Y-%m-%d"),
        "travel_median_s": dispatches["travel_s"].median(),
        "travel_p90_s": dispatches["travel_s"].astype("Float64").quantile(0.9),
        "incidents_off_map": outside["incident"], "engine_positions_off_map": outside["engine"],
    }])
    show_table("overview", overview, verbose)
    return overview


# ---------------------------------------------------------------------------------------------
# Engines per incident
# ---------------------------------------------------------------------------------------------
def build_incident_table(dispatches: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """One row per incident: engines sent, engines that arrived, and the incident's attributes.

    Example:
        >>> incidents = build_incident_table(dispatches, verbose=False)
        >>> incidents.shape[0], incidents["n_engines_sent"].min()     # (33156, 1)
    """
    incidents = dispatches.groupby("incident_id").agg(
        n_engines_sent=("engine_id", "size"),
        n_engines_arrived=("cancelled", lambda cancelled: int((~cancelled).sum())),
        call_type=("call_type", "first"), priority=("priority", "first"), focal=("focal", "first"),
        jurisdiction=("jurisdiction", "first"), protocol=("protocol", "first")).reset_index()
    assert (incidents["n_engines_sent"] == dispatches.groupby("incident_id")["n_engines_sent"].first().values).all()
    show_table("incidents", incidents, verbose)
    return incidents


def bucket_engine_count(counts: pd.Series) -> pd.Series:
    """Engine counts as labels, with ENGINE_COUNT_TOP and above pooled.

    Example:
        >>> bucket_engine_count(pd.Series([0, 1, 4, 7])).tolist()    # ['0', '1', '4', '5+']
    """
    return counts.where(counts < ENGINE_COUNT_TOP, ENGINE_COUNT_TOP).astype(int).astype(str).replace(
        {str(ENGINE_COUNT_TOP): f"{ENGINE_COUNT_TOP}+"})


def count_engines_per_incident(incidents: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """Incidents by number of engines sent and by number that arrived (count and share).

    Example:
        >>> table = count_engines_per_incident(incidents, verbose=False)
        >>> table.columns.tolist()     # ['engines', 'incidents_by_sent', 'share_by_sent', 'incidents_by_arrived', 'share_by_arrived']
    """
    sent = incidents["n_engines_sent"].value_counts().sort_index()
    arrived = incidents["n_engines_arrived"].value_counts().sort_index()
    labels = sorted(set(sent.index) | set(arrived.index))
    table = pd.DataFrame({"engines": labels,
                          "incidents_by_sent": sent.reindex(labels, fill_value=0).values,
                          "incidents_by_arrived": arrived.reindex(labels, fill_value=0).values})
    table.insert(2, "share_by_sent", table["incidents_by_sent"] / len(incidents))
    table["share_by_arrived"] = table["incidents_by_arrived"] / len(incidents)
    show_table("engines per incident", table, verbose, rows=len(table))
    return table


def engines_per_incident_by_group(incidents: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """For each call type, priority, focal area and common protocol: incidents, mean engines sent, share by engine count.

    Example:
        >>> table = engines_per_incident_by_group(incidents, verbose=False)
        >>> table.loc[table.group_column == "call_type", "group"].tolist()[:2]    # e.g. ['Medical', 'Fire']
    """
    common_protocols = incidents["protocol"].value_counts()
    common_protocols = common_protocols[common_protocols >= MIN_INCIDENTS_PER_PROTOCOL].index
    pieces = []
    for group_column in ["call_type", "priority", "focal", "jurisdiction", "protocol"]:
        subset = incidents if group_column != "protocol" else incidents[incidents["protocol"].isin(common_protocols)]
        group = subset[group_column].astype("string").fillna("(missing)")
        shares = pd.crosstab(group, bucket_engine_count(subset["n_engines_sent"]), normalize="index")
        summary = subset.groupby(group).agg(incidents=("incident_id", "size"), mean_engines_sent=("n_engines_sent", "mean"),
                                            mean_engines_arrived=("n_engines_arrived", "mean"))
        table = summary.join(shares.add_prefix("share_sent_")).sort_values("incidents", ascending=False)
        table = table.rename_axis("group").reset_index()
        table.insert(0, "group_column", group_column)
        pieces.append(table)
    result = pd.concat(pieces, ignore_index=True)
    share_columns = [column for column in result.columns if column.startswith("share_sent_")]
    result[share_columns] = result[share_columns].fillna(0.0)
    result.insert(0, "row_key", result["group_column"] + "=" + result["group"])
    show_table("engines per incident by group", result, verbose, rows=len(result))
    return result


def compute_response_curves(dispatches: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """Share of incidents that reached each stage within t minutes of the call (each incident's own call is t = 0).

    Stages: first engine dispatched, first engine rolling, first engine on scene, last engine cleared.
    The denominator is every incident in the group, so incidents no engine reached keep the curve below 100%.

    Example:
        >>> curves = compute_response_curves(dispatches, verbose=False)
        >>> curves.query("curve == 'stages' and series == 'first_on_scene' and minute == 20.0")["share"].round(3).tolist()   # [0.87...]
    """
    per_incident = dispatches.groupby("incident_id").agg(
        call=("call_received_at", "min"), dispatched=("dispatched_at", "min"), en_route=("enroute_at", "min"),
        first_on_scene=("arrived_at", "min"), last_cleared=("cleared_at", "max"),
        focal=("focal", "first"), priority=("priority", "first"))
    minutes = {stage: (per_incident[stage] - per_incident["call"]).dt.total_seconds() / 60
               for stage in ["dispatched", "en_route", "first_on_scene", "last_cleared"]}
    area = per_incident["focal"].map({True: "inside city", False: "outside city"})
    curve_specs = [
        ("stages", "dispatched", minutes["dispatched"], RESPONSE_GRID_MINUTES),
        ("stages", "en_route", minutes["en_route"], RESPONSE_GRID_MINUTES),
        ("stages", "first_on_scene", minutes["first_on_scene"], RESPONSE_GRID_MINUTES),
        ("completion", "first_on_scene", minutes["first_on_scene"], COMPLETION_GRID_MINUTES),
        ("completion", "last_cleared", minutes["last_cleared"], COMPLETION_GRID_MINUTES),
    ]
    curve_specs += [("first_on_scene_by_area", label, minutes["first_on_scene"][area == label], RESPONSE_GRID_MINUTES)
                    for label in ["inside city", "outside city"]]
    curve_specs += [("first_on_scene_by_priority", label, minutes["first_on_scene"][per_incident["priority"].astype("string").fillna("(missing)") == label],
                     RESPONSE_GRID_MINUTES) for label in ["Low", "Moderate", "High", "(missing)"]]
    pieces = []
    for curve, series, values, grid in curve_specs:
        reached = np.sort(values.dropna().to_numpy())
        share = np.searchsorted(reached, grid, side="right") / len(values)
        pieces.append(pd.DataFrame({"curve": curve, "series": series, "minute": grid, "share": share, "incidents": len(values)}))
    curves = pd.concat(pieces, ignore_index=True)
    show_table("response curves", curves, verbose)
    if verbose:
        first = minutes["first_on_scene"]
        for label, mask in [("all", slice(None)), ("inside city", area == "inside city"), ("outside city", area == "outside city")]:
            subset = first[mask]
            within = {name: f"{(subset <= seconds / 60).sum() / len(subset):.1%}" for name, seconds in SOC_TARGETS_SECONDS.items()}
            print(f"first engine on scene, {label}: n={len(subset):,}, reached {subset.notna().mean():.1%}, "
                  f"median {subset.median():.2f} min, P90 of reached {subset.quantile(0.9):.2f} min, within SOC targets {within}")
        print(f"last engine cleared within 60 / 120 min: {(minutes['last_cleared'] <= 60).mean():.1%} / {(minutes['last_cleared'] <= 120).mean():.1%}\n")
    return curves


def main(verbose: bool = VERBOSE) -> None:
    """Compute every table and write it to OUTPUT_DIR.

    Example:
        >>> main(verbose=False)     # writes nine CSV files under data/reports/engine_dispatches_fy2025/
    """
    dispatches = read_dispatches(verbose)
    incidents = build_incident_table(dispatches, verbose)
    grid, outside = grid_locations(dispatches, verbose)
    tables = {
        "overview": summarize_overview(dispatches, outside, verbose),
        "numeric_summary": summarize_numeric(dispatches, NUMERIC_COLUMNS, verbose),
        "histogram_bins": bin_durations(dispatches, verbose),
        "category_counts": count_categories(dispatches, CATEGORY_COLUMNS, verbose),
        "time_profile": profile_over_time(dispatches, verbose),
        "durations_by_group": summarize_durations_by_group(dispatches, verbose),
        "location_grid": grid,
        "engines_per_incident": count_engines_per_incident(incidents, verbose),
        "engines_per_incident_by_group": engines_per_incident_by_group(incidents, verbose),
        "response_curves": compute_response_curves(dispatches, verbose),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        if verbose:
            print(f"wrote {(OUTPUT_DIR / f'{name}.csv').relative_to(PROJECT_ROOT)} ({len(table):,} rows)")


if __name__ == "__main__":
    main()
