"""Format the existing evaluations and the two completed experiments for the paper."""
import json
import pickle
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"do nothing": "No action", "city plan": "Modeled Station 9 relocation",
         "greedy": "Stage greedy", "learned": "Frozen PPO"}
ORDER = list(NAMES)
METRICS = ("D_F", "M_F", "M_O")
captions, values = [], {}


def write_table(name, caption, headers, rows, specification):
    def tex_cell(value):
        value = str(value)
        match = re.fullmatch(r"([+-][\d.]+) (\[[^\]]+\])", value)
        return "\\shortstack[r]{" + match[1] + "\\\\\\strut " + match[2] + "}" if match else value
    text = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    text += "\\caption{" + caption + "}\n\\label{tab:" + name + "}\n"
    text += "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{" + specification + "}\n\\toprule\n"
    text += " & ".join(headers) + " \\\\\n\\midrule\n"
    text += "\n".join(" & ".join(tex_cell(value) for value in row) + " \\\\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (OUT / f"{name}.tex").write_text(text)
    readable_caption = caption.replace("\\%", "%").replace("\\_", "_")
    captions.append("### " + name + "\n\n" + readable_caption + "\n\n| "
                    + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers)
                    + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows))


def interval(row, metric):
    def number(value):
        return f"{value:+.6f}" if 0 < abs(value) < 0.0001 else f"{value:+.4f}"
    return f"{number(row[f'delta_{metric}'])} [{number(row[f'low_{metric}'])}, {number(row[f'high_{metric}'])}]"


data = pd.read_csv(HERE.parent / "data/models/planning/final/evaluation_draws.csv")
data = data[data.kind == "draw"]
assert data.groupby("policy").size().to_dict() == {p: 48 for p in ORDER}
summary = data.groupby("policy").agg(**{
    "D_F": ("mean D_F", "mean"), "M_F": ("mean M_F", "mean"), "M_O": ("mean M_O", "mean"),
    "actions": ("actions", "mean"), "violations": ("violations", "sum"), "return": ("return", "mean")}).loc[ORDER]
rows = [[NAMES[p], *[f"{r[m]:.3f}" for m in METRICS], f"{r['return']:.3f}", f"{r.actions:.1f}", f"{int(r.violations)}/192"]
        for p, r in summary.iterrows()]
write_table("fast_policy_levels",
    "Policy outcomes from the approximate queueing evaluator on 48 new growth draws. "
    "Each draw contains four evaluations (2030, 2035, 2040, and 2045); outcomes are averaged over those years and draws. "
    "$D_F$ is the fixed-fire-rate-weighted standard deviation of tract mean fire travel times; "
    "$M_F$ and $M_O$ are regional mean travel times for study-fire and other calls. All three are in minutes. "
    "Return sums the four original stage rewards; higher is better. Each stage cost is $D_F$ plus ten times "
    "the positive excess of each mean above its 2025 target (6.6962 min for fire and 5.9992 min for other calls). "
    "The penalty starts at zero excess. Actions count station and vehicle planning changes per draw. A target violation occurs when either mean "
    "exceeds its own 2025 baseline by more than 0.05 min. No action retains the initial configuration; "
    "the Station 9 policy moves that station once; stage greedy chooses the feasible action pair with the lowest "
    "next-stage cost; frozen PPO applies the trained proximal policy optimization policy. "
    "Greedy and PPO share stage action limits. Gains against no action combine location changes and fleet additions.",
    ["Policy", "$D_F$", "$M_F$", "$M_O$", "Return", "Actions", "Violations"], rows, "lrrrrrr")
values["fast_policy_levels"] = summary.reset_index().to_dict("records")

execution = HERE / "results/execution_summary.json"
if execution.exists() and json.loads(execution.read_text()).get("status") == "complete":
    paired = pd.read_csv(HERE / "results/paired_comparisons.csv")
    e1 = paired[paired.experiment == "E1"]
    assert len(e1) == 6
    rows = [[f"{r['path']}, $w={r['w']:g}$", str(r['year']), *[interval(r, m) for m in METRICS]]
            for _, r in e1.iterrows()]
    write_table("e1_policy_contrasts",
        "Direct frozen-PPO minus stage-greedy contrasts in discrete-event simulation (E1). "
        "Negative values favor PPO. Entries are differences in minutes with 95\\% paired bootstrap intervals. "
        "$D_F$ measures fixed-fire-rate-weighted dispersion of tract mean fire travel; $M_F$ and $M_O$ measure "
        "regional mean fire and other-call travel. Both policies share the same 15 incident streams (seeds 101--115); "
        "2,000 bootstrap draws resample whole streams and recompute pooled tract means and dispersion. "
        "Medium and High denote county population paths; spatial growth mix $w=0$ allocates growth in proportion "
        "to population and $w=1$ follows positive historical growth shares. Years are evaluation years. "
        "Zero intervals at identical configurations express deterministic equality, with no extra simulation. "
        "Intervals describe incident-stream variation within each stated scenario.",
        ["Growth scenario", "Year", "$\\Delta D_F$ [95\\% CI]", "$\\Delta M_F$ [95\\% CI]", "$\\Delta M_O$ [95\\% CI]"],
        rows, "llrrr")
    values["e1_policy_contrasts"] = e1.to_dict("records")
    plan = pickle.loads((HERE / "results/plan.pkl").read_bytes())
    resource_rows, resource_values = [], []
    for case in plan["comparisons"]:
        if case["experiment"] != "E1":
            continue
        row = [f"{case['path']}, $w={case['w']:g}$", str(case["year"])]
        detail = {key: case[key] for key in ("path", "w", "year", "identical")}
        for side in ("a", "b"):
            config = plan["configs"][case[side+"_config"]]
            counts = [sum(int(site >= 0) for site in config["site"]),
                      int(sum(config["engines"])), int(sum(config["quints"]))]
            row.append(" / ".join(map(str, counts)))
            detail["PPO" if side == "a" else "greedy"] = dict(zip(("locations", "engines", "quints"), counts))
        fleet_delta = sum(detail["PPO"][k]-detail["greedy"][k] for k in ("engines", "quints"))
        row.append(f"{fleet_delta:+d}")
        detail["delta_total_vehicles"] = fleet_delta
        resource_rows.append(row)
        resource_values.append(detail)
    write_table("e1_resource_configurations",
        "Resource configurations used in E1. Each entry gives active station/base locations, engines (E), and quints (Q) "
        "in that order. Both policies start with 25 locations, 23 engines and 5 quints. Each five-year stage allows "
        "at most one station action (build, relocate, or hold) and one vehicle addition (engine or quint, or hold). "
        "Building includes one engine, so equal action limits permit different realized fleet sizes and compositions. "
        "The final column is PPO minus greedy in total vehicles. Policy contrasts include the consequences of these resource choices. "
        "Medium and High are county population paths; $w=0$ allocates growth in proportion to population and "
        "$w=1$ follows positive historical growth. Years denote evaluation years.",
        ["Growth scenario", "Year", "PPO: locations / E / Q", "Greedy: locations / E / Q", "$\\Delta$ vehicles"], resource_rows, "llrrr")
    values["e1_resource_configurations"] = resource_values
    e2 = pd.read_csv(HERE / "results/e2_remaining_horizon.csv")
    rows = []
    for _, r in e2.iterrows():
        per_year = paired[(paired.experiment == "E2") & (paired.w == r["w"])].sort_values("year")
        for _, yearly in per_year.iterrows():
            rows.append([f"$w={r['w']:g}$", r["chosen"], str(int(yearly["year"])), *[interval(yearly, m) for m in METRICS]])
        rows.append([f"$w={r['w']:g}$", r["chosen"], "Mean", *[interval(r, m) for m in METRICS]])
    write_table("e2_action_contrasts",
        "Alternative engine-addition destinations after the common 2035 station action (E2). "
        "Action A adds an engine at the modeled E62 base; action B adds one at CO\\_29. "
        "The table reports A minus B for each evaluation year and their arithmetic mean, in minutes; negative values favor A. "
        "Chosen denotes the frozen PPO policy's observed selection. Both states use the Medium county population "
        "path and the same resource configuration: $w=0$ uses population-proportional growth and $w=1$ uses "
        "positive historical-growth shares. All branches continue with the same frozen policy. "
        "$D_F$ is fixed-fire-rate-weighted dispersion of tract mean fire travel; $M_F$ and $M_O$ are regional "
        "mean fire and other-call travel. The 95\\% intervals use 2,000 paired whole-stream bootstrap draws "
        "over 15 streams, with the same resampled seed indices across the two years. "
        "These contrasts evaluate the specified alternative actions and continuation rule.",
        ["Growth", "Chosen", "Year", "$\\Delta D_F$ [95\\% CI]", "$\\Delta M_F$ [95\\% CI]", "$\\Delta M_O$ [95\\% CI]"],
        rows, "lllrrr")
    rows, objective_values = [], []
    for _, r in e2.iterrows():
        penalties = []
        for action in ("A", "B"):
            history = plan["fast_histories"][(action, ("Medium", r["w"]))]["history"][2:]
            penalties.append(10*sum(max(0, h[m]-plan["targets_fast"][m]) for h in history for m in ("M_F", "M_O")))
        rows.append([f"$w={r['w']:g}$", r["chosen"], f"{r['fast_delta_D_F']:+.5f}",
            *[f"{penalty:.5f}" for penalty in penalties],
            f"{r['fast_remaining_return_A']:.5f}", f"{r['fast_remaining_return_B']:.5f}"])
        objective_values.append({"w": r["w"], "chosen": r["chosen"], "penalty_A": penalties[0], "penalty_B": penalties[1]})
    write_table("e2_fast_objective",
        "Approximate-evaluator outcomes for the same E2 branches. The dispersion contrast is A minus B "
        "averaged over 2040 and 2045; negative favors A. Remaining return sums the original stage rewards "
        "over those two years; higher is better. Each reward is minus the sum of fire-travel dispersion "
        "and the original penalty for regional fire and other-call means above their 2025 targets. "
        "Penalty A/B sums these penalties over both years; each is ten times positive mean excess, with no "
        "0.05-min deadband. The 0.05-min tolerance applies only to violation reporting. "
        "Return and penalty use minute-based objective units, not additional physical travel time. "
        "A adds one engine at E62 and B at CO\\_29, followed by the frozen policy. "
        "The two states share the Medium county growth path and post-relocation branch configuration; "
        "$w=0$ is population-proportional growth and $w=1$ follows positive historical-growth shares.",
        ["Growth", "Chosen", "$\\Delta D_F$ (min)", "Penalty A", "Penalty B", "Return A", "Return B"], rows, "llrrrrr")
    values["e2_action_contrasts"] = e2.to_dict("records")
    values["e2_objective_components"] = objective_values

    fast_tracts = pd.read_csv(HERE / "results/fast_tract_outcomes.csv")
    simulated_tracts = pd.read_csv(HERE / "results/spatial_contrasts.csv")
    spatial_rows, spatial_values = [], []
    for source in ("Approximate evaluator", "Event simulator"):
        for path, w in (("Medium", 1), ("High", 1), ("High", 0)):
            if source == "Approximate evaluator":
                subset = fast_tracts[(fast_tracts.path == path) & (fast_tracts.w == w) & (fast_tracts.year == 2045)]
                a = subset[subset.policy == "learned"].set_index("tract")
                b = subset[subset.policy == "do nothing"].set_index("tract")
                assert a.index.equals(b.index)
                delta, weight, reference = a.fire_mean-b.fire_mean, a.fire_weight, b.fire_mean
                comparison = "PPO--no action (approx.)"
            else:
                subset = simulated_tracts[(simulated_tracts.experiment == "E1") & (simulated_tracts.path == path)
                    & (simulated_tracts.w == w) & (simulated_tracts.year == 2045)].set_index("tract")
                delta, weight, reference = subset.a_minus_b, subset.fire_weight, subset.b_mean
                comparison = "PPO--greedy (sim.)"
            positive = weight > 0
            assert len(weight[positive]) == 57 and abs(weight.sum()-1) < 1e-10
            assert delta[positive].notna().all() and reference[positive].notna().all()
            faster, slower = (delta < -1e-10) & positive, (delta > 1e-10) & positive
            unchanged = positive & ~faster & ~slower
            shares = [float(weight[mask].sum()) for mask in (faster, slower, unchanged)]
            detail = {"source": source, "path": path, "w": w, "year": 2045,
                "policy": "PPO", "comparator": "no action" if source == "Approximate evaluator" else "stage greedy",
                "faster_fire_weight": shares[0], "slower_fire_weight": shares[1], "unchanged_fire_weight": shares[2],
                "faster_tracts": int(faster.sum()), "slower_tracts": int(slower.sum()),
                "unchanged_tracts": int(unchanged.sum()), "unchanged_tolerance_min": 1e-10}
            spatial_values.append(detail)
            spatial_rows.append([f"{path}, $w={w}$", comparison, *[f"{100*s:.1f}\\%" for s in shares]])
    write_table("spatial_consequences",
        "Distribution of local fire-travel changes in 2045. Shares sum the fixed baseline fire-demand weights "
        "of the 57 tracts with positive fire rates; each row totals 100\\% apart from rounding. "
        "Faster and slower identify lower and higher tract mean travel under frozen PPO than under the stated "
        "reference policy. Approximate-evaluator rows compare PPO with no action; event-simulator rows compare "
        "PPO with stage greedy using the E1 pooled outputs. Unchanged means equality within numerical precision "
        "($10^{-10}$ min). Categories describe point-estimate signs, rather than tract-level significance tests. "
        "Medium and High are county population paths; $w=0$ uses population-proportional growth and $w=1$ "
        "uses positive historical-growth shares. Changes in spatial dispersion and regional means can coexist "
        "with longer travel times in some tracts.",
        ["Growth scenario", "Comparison", "Faster", "Slower", "Unchanged"], spatial_rows, "llrrr")
    values["spatial_consequences"] = spatial_values

(HERE / "tables_for_review.md").write_text("# Paper tables\n\n" + "\n\n".join(captions) + "\n")
(HERE / "results/paper_table_values.json").write_text(json.dumps(values, indent=2, allow_nan=False))
print(f"Formatted {len(captions)} tables; E1/E2 included: {len(captions) > 1}")
