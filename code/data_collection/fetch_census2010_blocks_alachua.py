"""Download the 2010 Census blocks of Alachua County from Census TIGERweb, attributes only: residents (POP100), housing
units (HU100), urban or rural (UR) and each block's internal point. Used to compare 2010 and 2020 residents on a common
grid (block boundaries changed between the two censuses, so blocks are compared through their internal points).

Output:
    data/raw/gnv_alachua/census2010_blocks_alachua.csv   one row per 2010 block

Run from the project root:
    python code/data_collection/fetch_census2010_blocks_alachua.py
"""
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]                          # [DESIGN] this file lives in code/data_collection/
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "gnv_alachua" / "census2010_blocks_alachua.csv"  # [DESIGN]
BLOCKS_LAYER = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2010/MapServer/18"  # [SOURCE] 2010 Census blocks
ALACHUA_WHERE = "STATE='12' AND COUNTY='001'"                               # [SOURCE] Florida, Alachua County
FIELDS = ["GEOID", "STATE", "COUNTY", "TRACT", "BLOCK", "UR", "POP100", "HU100", "AREALAND", "INTPTLON", "INTPTLAT"]  # [DESIGN] no geometry
PAGE_SIZE = 5000                                                            # [DESIGN] rows per request (the layer allows 100,000)
TIMEOUT_SECONDS = 120                                                       # [DESIGN] per request
VERBOSE = True                                                              # [DESIGN] print inputs and outputs


def query_layer(params, verbose=VERBOSE):
    """One query of the TIGERweb block layer; returns the parsed JSON and stops on a service error.

    Example: query_layer({"where": ALACHUA_WHERE, "returnCountOnly": "true", "f": "json"}) -> {'count': 7382}
    """
    response = requests.get(f"{BLOCKS_LAYER}/query", params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    page = response.json()
    if "error" in page:
        raise RuntimeError(f"{BLOCKS_LAYER}: {page['error']}")
    if verbose and "features" in page:
        print(f"rows in this page: {len(page['features']):,} (offset {params.get('resultOffset', 0):,})")
    return page


def download_block_attributes(verbose=VERBOSE):
    """Every 2010 Census block of Alachua County as a table, page by page, checked against the layer's own count.

    Example: download_block_attributes().shape -> (7382, 11)
    """
    expected = query_layer({"where": ALACHUA_WHERE, "returnCountOnly": "true", "f": "json"}, verbose)["count"]
    rows = []
    while True:
        page = query_layer({"where": ALACHUA_WHERE, "outFields": ",".join(FIELDS), "returnGeometry": "false",
                            "orderByFields": "GEOID", "resultOffset": len(rows), "resultRecordCount": PAGE_SIZE, "f": "json"}, verbose)
        batch = [feature["attributes"] for feature in page["features"]]
        rows.extend(batch)
        if not batch or (len(batch) < PAGE_SIZE and not page.get("exceededTransferLimit")):
            break
    blocks = pd.DataFrame(rows)[FIELDS]
    assert len(blocks) == expected and blocks["GEOID"].is_unique, f"got {len(blocks)} rows for {expected} blocks"
    if verbose:
        print(f"--- 2010 Census blocks, Alachua County {blocks.shape}\n{blocks.head(5).to_string()}\n"
              f"residents {blocks.POP100.sum():,}, housing units {blocks.HU100.sum():,}, "
              f"urban / rural blocks {blocks.UR.value_counts().to_dict()}\n")
    return blocks


if __name__ == "__main__":
    blocks_2010 = download_block_attributes()
    blocks_2010.to_csv(OUTPUT_CSV, index=False)
    print(f"wrote {OUTPUT_CSV.relative_to(PROJECT_ROOT)} ({len(blocks_2010):,} rows, {OUTPUT_CSV.stat().st_size / 1e6:.2f} MB)")
