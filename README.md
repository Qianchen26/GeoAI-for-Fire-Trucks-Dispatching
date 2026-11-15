# GeoAI-for-Fire-Trucks-Dispatching

Code for a SimPy simulator of fire-engine dispatch in Gainesville and Alachua County, Florida. It replays the FY2025
engine dispatches of Gainesville Fire Rescue and Alachua County Fire Rescue, then generates synthetic scenarios for
training a dispatch policy. The data are not stored here; the first script downloads them.

## Requirements

Python 3.13 with simpy, pandas, numpy, pyarrow, geopandas, shapely, pyproj, osmnx, networkx, scipy, scikit-learn and
requests. The notebook also needs jupyter and matplotlib. Tested with simpy 4.1.2, pandas 3.0.1, numpy 2.4.2,
geopandas 1.1.3, osmnx 2.1.0, networkx 3.6.1, scipy 1.15.3 and scikit-learn 1.6.1.

Run every command from the repository root. The scripts read and write under `data/`, which they create.

## Steps

**1. Download the inputs** into `data/raw/gnv_alachua/`:

    python -I code/data_collection/fetch_gnv_alachua_layers.py

With no arguments it downloads every layer not yet on disk, then writes a README in that folder with the sources and
any layer that failed. To download only the files the steps below use, name them:

    python -I code/data_collection/fetch_gnv_alachua_layers.py \
        acfr_gfr_unit_responses_fy2025.parquet acfr_gfr_automatic_aid_fy2025.parquet \
        acfr_gfr_automatic_aid_fy2024.parquet gfr_stations_2022.gpkg acdps_fire_ems_stations_2025.gpkg \
        gainesville_city_limits_2026.gpkg alachua_county_boundary.gpkg census2020_blocks_alachua.gpkg \
        alachua_drive.graphml

**2. Build the engine dispatch table:**

    python code/data_collection/build_engine_dispatches_fy2025.py

Writes `engine_dispatches.parquet` (one row per engine dispatch), `engines.csv` and `stations.csv` to
`data/derived/engine_dispatches_fy2025/`. It also reads `data/derived/gfr_soc_2024/dashboard/appendix_e_crosswalk.csv`
(call code to risk level, from Appendix E of the GFR Standards of Cover 2024), which no script here produces.

**3. Describe the 2025 dispatches:**

    python code/analysis/describe_engine_dispatches_fy2025.py

Writes descriptive tables to `data/reports/engine_dispatches_fy2025/`, including `response_curves.csv`, which the
simulator's Section 1 compares byte for byte.

**4. Run the simulator:**

    python code/reproduce_dispatch_simulator.py

Runs Sections 1–13 in order (about one hour) and writes every output to `data/reports/reproduce_dispatch_simulator/`.
The sections cover replaying the record and testing the dispatch rule (1–5), the counterfactual simulator checked
against the record (6–10), first-due coverage and fire readiness (11), an opportunity-cost dispatch rule (12), and
synthetic scenarios with stress conditions (13). The module docstring describes each section and output file.

**5. Step log (optional):**

    jupyter notebook code/reproduce_dispatch_simulator_steps.ipynb

The same steps, cell by cell, with each function's output on concrete cases. Before running it, set `PROJECT_ROOT` in
the first code cell to your copy of the repository. A full run takes about two hours; the last cell runs the
simulator script on its own as a final check.

## Lombok base model

    python code/reproduce_simulator.py

The dispatch simulator started from this minimal SimPy version of the Lombok queue-recovery model (Alisjahbana &
Kiremidjian 2021, github.com/ialsjbn/queue-recovery). It needs `lombok_data.pkl` from that repository: set
`BNPB_PICKLE` at the top of the file to its path. It runs in about 5.5 minutes and writes to
`data/reports/reproduce_simulator/`.
