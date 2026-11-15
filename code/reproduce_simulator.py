"""Minimal SimPy version of Irene Alisjahbana's Lombok queue-recovery model (Alisjahbana & Kiremidjian 2021,
github.com/ialsjbn/queue-recovery): the part that reproduces the BNPB-observed rebuilding history.

Every damaged house asks for a contractor (simpy.PriorityResource: major damage first; minor and moderate are
shuffled together behind it; in West Lombok the minor and moderate houses ask first). The contractor stays with the
house through the pre-construction delay and the construction, both lognormal. Ten 600-day simulations per region,
simulation k seeded with np.random.seed(SEED + k); the mean daily completion curve is scored against the BNPB
reports with RMSE, overall and per damage level. Irene's own code, seeded the same way, gives the same finish
times, curves and RMSE digit for digit (checked 2026-11-05).

Outputs (data/reports/reproduce_simulator/):
    table_rmse.csv       RMSE; rows overall/minor/moderate/major, columns regions + average (as Irene's error table)
    recovery_curves.csv  region, level, day 1-599, simulated mean completion, BNPB observed on report days

Run from the project root (about 5.5 minutes):
    python code/reproduce_simulator.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import simpy
from sklearn.metrics import mean_squared_error

BNPB_PICKLE = Path("/Users/qianchenyu/Documents/pyrebuild/queue-recovery-main/lombok_data.pkl")  # [SOURCE] Irene's repo: BNPB finished and target houses per report date
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports" / "reproduce_simulator"  # [DESIGN]
SEED = 0                                       # [DESIGN] simulation k runs on np.random.seed(SEED + k)
N_SIMULATIONS = 10                             # [SOURCE] Irene's num_sim
SIM_DAYS = 600                                 # [SOURCE] Irene's SIM_TIME
DAYS = np.arange(1, SIM_DAYS)                  # [SOURCE] Irene's days: curves cover days 1-599
EARTHQUAKE_DATE = pd.to_datetime("20180819")   # [SOURCE] day 0
DELAY_MEDIAN_DAYS = np.array([275, 300, 225])  # [SOURCE] Irene's DELAY_MED: minor, moderate, major
DELAY_SIGMA = np.array([0.3, 0.3, 0.3])        # [SOURCE] Irene's BETA_DELAY
BUILD_MEDIAN_DAYS = np.array([30, 40, 50])     # [SOURCE] Irene's CONS_MED
BUILD_SIGMA = 0.4                              # [SOURCE] Irene's BETA_CONS
MAJOR = 2                                      # [SOURCE] damage code 0 minor, 1 moderate, 2 major; major asks with priority 0, the rest 1
LEVELS = ["overall", "minor", "moderate", "major"]  # [DESIGN] curve rows: all houses, then damage code 0, 1, 2
BNPB_SUFFIX = ["rr", "rs", "rb"]               # [SOURCE] BNPB column suffix of minor, moderate, major
MINOR_MODERATE_FIRST = {"lombokbarat"}         # [SOURCE] West Lombok repaired minor and moderate first (Alisjahbana & Kiremidjian 2021)
# region: (houses [minor, moderate, major], contractors) — [SOURCE] Irene's damage_dist and NUM_CONTRACTOR,
# in the column order of her error table
REGIONS = {
    "mataram":      ((9500, 3672, 1345),    9917),
    "sumbawa":      ((9652, 2756, 1374),    10360),
    "sumbawabarat": ((13078, 3803, 1283),   10200),
    "lombokbarat":  ((45218, 13556, 14069), 45208),
    "lombokutara":  ((8889, 4772, 42049),   22996),
    "lomboktimur":  ((12209, 4657, 10104),  15404),
    "lomboktengah": ((16639, 3096, 4483),   15048),
}


def simulate(region, seed):
    """One run of the contractor queue; returns the houses finished before day 600, in finish order.

    Example: simulate("mataram", 0)[:2] -> [(0, 115.134...), (0, 116.028...)]  # (damage code: int, finish day: float)
    """
    houses, contractors = REGIONS[region]
    np.random.seed(seed)
    minor_moderate = np.concatenate((np.zeros(houses[0]), np.ones(houses[1])))
    np.random.shuffle(minor_moderate)                              # one priority, random order
    major = np.ones(houses[2]) * MAJOR
    ask_order = np.concatenate((minor_moderate, major) if region in MINOR_MODERATE_FIRST else (major, minor_moderate))
    env = simpy.Environment()
    contractor_pool = simpy.PriorityResource(env, contractors)
    finished = []

    def house(damage):
        with contractor_pool.request(priority=0 if damage == MAJOR else 1) as contractor:
            yield contractor                                       # held until the house is rebuilt
            yield env.timeout(np.random.lognormal(np.log(DELAY_MEDIAN_DAYS[damage]), DELAY_SIGMA[damage]))
            yield env.timeout(np.random.lognormal(np.log(BUILD_MEDIAN_DAYS[damage]), BUILD_SIGMA))
            finished.append((int(damage), env.now))

    for damage_code in ask_order.astype(int):
        env.process(house(damage_code))
    env.run(until=SIM_DAYS)
    return finished


def mean_curves(runs, houses):
    """Share of houses finished by each day, averaged over runs; a house finished at day t counts from day ceil(t).

    Example: mean_curves(runs["mataram"], (9500, 3672, 1345)) -> float array (4, 599); rows LEVELS, column i = day i + 1
    """
    completion = np.zeros((len(runs), len(LEVELS), len(DAYS)))
    for k, run in enumerate(runs):
        damage = np.array([house_damage for house_damage, _ in run])
        done_day = np.ceil([finish for _, finish in run])
        completion[k, 0] = np.searchsorted(np.sort(done_day), DAYS, side="right") / sum(houses)
        for code in range(MAJOR + 1):
            completion[k, code + 1] = np.searchsorted(np.sort(done_day[damage == code]), DAYS, side="right") / houses[code]
    return completion.mean(axis=0)


def observed_recovery(bnpb, region):
    """BNPB share of houses finished on each report day.

    overall = finished houses summed over the three levels (an unreported level counts as 0) / all damaged houses;
    each level = finished houses / BNPB's last target for that level (NaN where BNPB did not report it).

    Example: observed_recovery(bnpb, "mataram") -> DataFrame; index day since the quake (int), columns LEVELS (float)
    """
    houses = REGIONS[region][0]
    finished_all = bnpb[[f"{region}_comp_{suffix}" for suffix in BNPB_SUFFIX[::-1]]].sum(axis=1)
    observed = pd.DataFrame({"overall": finished_all / sum(houses)})
    for level, suffix in zip(LEVELS[1:], BNPB_SUFFIX):
        observed[level] = bnpb[f"{region}_comp_{suffix}"] / bnpb[f"{region}_target_{suffix}"].iloc[-1]
    observed.index = (bnpb.index - EARTHQUAKE_DATE).days
    return observed


def rmse_table(observed, curves):
    """RMSE of the mean curve on BNPB report days, per region and level, plus the average over regions.

    Example: rmse_table(observed, curves).loc["overall", "mataram"] -> 0.16413777696647902 (float)
    """
    rmse = {}
    for region in REGIONS:
        rmse[region] = []
        for row, level in enumerate(LEVELS):
            share = observed[region][level]
            reported = share.notna().to_numpy()                    # overall is never NaN: every report day counts
            simulated = curves[region][row, share.index.to_numpy()[reported] - 1]
            rmse[region].append(np.sqrt(mean_squared_error(share[reported], simulated)))
    table = pd.DataFrame(rmse, index=LEVELS)
    table["average"] = table.mean(axis=1)
    return table


def write_outputs(observed, curves, table):
    """Save the RMSE table and the daily curves (simulated mean next to BNPB on report days).

    BNPB has four reports for 2019-01-30 (day 164); the RMSE scores all four, so that day gets four rows.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "table_rmse.csv", index_label="level")
    pieces = []
    for region in REGIONS:
        for row, level in enumerate(LEVELS):
            simulated = pd.DataFrame({"region": region, "level": level, "day": DAYS, "simulated_mean": curves[region][row]})
            reports = observed[region][level].rename("observed").rename_axis("day").reset_index()
            pieces.append(simulated.merge(reports, on="day", how="left"))
    curves_long = pd.concat(pieces)
    curves_long.to_csv(OUTPUT_DIR / "recovery_curves.csv", index=False)
    print(f"wrote {OUTPUT_DIR}/table_rmse.csv and recovery_curves.csv ({len(curves_long):,} rows)")


if __name__ == "__main__":
    bnpb = pd.read_pickle(BNPB_PICKLE)
    print(f"BNPB reports {bnpb.shape}, {bnpb.index.min().date()} to {bnpb.index.max().date()}\n{bnpb.head(5)}\n")
    runs, curves = {}, {}
    for region, (houses, contractors) in REGIONS.items():
        runs[region] = [simulate(region, SEED + k) for k in range(N_SIMULATIONS)]
        curves[region] = mean_curves(runs[region], houses)
        print(f"{region}: {sum(houses):,} houses, {contractors:,} contractors, seeds {SEED}-{SEED + N_SIMULATIONS - 1}; "
              f"mean share finished by day {DAYS[-1]}: {curves[region][0, -1]:.4f}")
    observed = {region: observed_recovery(bnpb, region) for region in REGIONS}
    table = rmse_table(observed, curves)
    print(f"\nRMSE (rows: level; columns: region)\n{table.round(4).to_string()}")
    write_outputs(observed, curves, table)
