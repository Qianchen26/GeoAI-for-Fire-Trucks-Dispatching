"""Verify the archived E1/E2 comparisons in a checkout with the frozen inputs.

Four deterministic policy replays verify the recorded actions. The archived
receipt identifies the 225 simulation files by path, byte count and SHA-256.
"""
from datetime import datetime, timezone
import ast
import inspect
import json
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CEUS = HERE.parents[1]
ROOT = CEUS.parent
sys.path.insert(0, str(CEUS))
import run_experiments as R
import numpy as np
import pandas as pd


def describe_decoded(action, names):
    kind, slot, candidate = action
    if kind == "build":
        return f"build at C{candidate}"
    if kind == "move":
        return f"move {names[slot]} to C{candidate}"
    if kind in ("engine", "quint"):
        return f"+{'E' if kind == 'engine' else 'Q'} at {names[slot]}"
    return "-"


def source_anchor(path, text):
    lines = path.read_text().splitlines()
    hits = [{"line": n, "text": line.strip()} for n, line in enumerate(lines, 1) if text in line]
    assert hits, (path, text)
    return {"path": str(path.relative_to(ROOT)), "matches": hits}


def main():
    manifest_path = CEUS / "inputs_manifest.json"
    manifest_hash = R.sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checked = [{**entry, "matches": R.sha(ROOT / entry["path"]) == entry["sha256"]}
               for entry in manifest["inputs"]]
    assert all(entry["matches"] for entry in checked)
    plan = pickle.loads((R.RESULTS / "plan.pkl").read_bytes())
    weights = np.asarray(plan["fire_weight"])
    evidence = {"audit_utc": datetime.now(timezone.utc).isoformat(),
                "audit_script_sha256": R.sha(Path(__file__)), "release_manifest_sha256": manifest_hash,
                "original_inputs": checked, "event_simulations_added": 0, "training_runs_added": 0}

    # Reconstruct existing inputs and replay the policy, without generating event
    # streams or calling simulate(), experiment prepare(), or a cache writer.
    _, _, inputs, evaluator, candidates = R.load_setup()
    model = R.MaskablePPO.load(R.MODEL_DIR / "final" / "model.zip", device="cpu")
    expected = pd.read_csv(R.MODEL_DIR / "final" / "rq3_actions.csv")
    checks, metric_errors, configuration_matches = [], [], []
    draws = [draw for policy, draw in plan["fast_histories"] if policy == "learned"]
    for draw in draws:
        env = R.P.PlanningEnv(inputs, evaluator, candidates)
        replay = R.P.play(env, lambda e, obs: R.predict(model, e, obs), draw)
        rows = expected[(expected.path == draw[0]) & (expected.w == draw[1])].sort_values("year")
        assert rows.year.is_unique
        history = sorted(replay["history"], key=lambda h: h["year"])
        stored = sorted(plan["fast_histories"][("learned", draw)]["history"], key=lambda h: h["year"])
        for h, old_h, (_, row) in zip(history, stored, rows.iterrows(), strict=True):
            assert h["year"] == old_h["year"] == row["year"]
            actions = [describe_decoded(a, h["config"]["name"]) for a in h["actions"]]
            assert actions == [row["station action"], row["vehicle action"]]
            errors = [abs(h[m] - row[m]) for m in R.METRICS]
            assert max(errors) < 1e-7
            assert all(abs(h[m] - old_h[m]) < 1e-12 for m in R.METRICS)
            same_config = R.configuration_id(h["config"]) == R.configuration_id(old_h["config"])
            assert same_config
            configuration_matches.append(same_config)
            metric_errors.extend(errors)
            checks.append({"path": draw[0], "w": draw[1], "year": h["year"],
                           "station_action": actions[0], "vehicle_action": actions[1],
                           "matches_original_actions_and_metrics": True})
    replay_path = R.RESULTS / "frozen_replay_checks.json"
    replay_bytes = (json.dumps(checks, indent=2) + "\n").encode()
    assert replay_bytes == replay_path.read_bytes(), "the regenerated replay receipt differs"
    evidence["replay"] = {"fresh_deterministic_episodes": len(draws), "stages": len(checks),
                          "actions_verified": len(checks) * 2, "metrics_verified": len(metric_errors),
                          "max_abs_metric_difference": max(metric_errors),
                          "stored_configurations_match": all(configuration_matches),
                          "expected_and_replayed_histories_sorted_by_evaluation_year": True,
                          "decision_year_rule": "decision_year = evaluation_year - 5",
                          "archived_receipt_sha256": R.sha(replay_path),
                          "regenerated_receipt_matches_archive": True}

    # Both implementations use exactly this original demand-point bincount.
    fire_rate = np.bincount(inputs["points"]["tract"], weights=inputs["points"]["fire_rate"],
                            minlength=len(inputs["tracts"]))
    normalized = fire_rate / fire_rate.sum()
    assert np.array_equal(fire_rate, evaluator.fire_tract_rate)
    assert np.array_equal(normalized, weights)
    assert inputs["tracts"]["tract"].tolist() == plan["tract_ids"]
    evidence["fixed_weights"] = {"positive_weight_tracts": int((weights > 0).sum()),
                                 "tracts": len(weights), "evaluator_bincount_exact": True,
                                 "saved_weight_bincount_exact": True, "tract_order_exact": True}

    paired = pd.read_csv(R.RESULTS / "paired_comparisons.csv")
    verification = pd.read_csv(R.MODEL_DIR / "final" / "verification.csv")
    cohort_checks, reproduction = [], []
    for case in plan["comparisons"]:
        if case["identical"]:
            assert case["a_config"] == case["b_config"]
        else:
            aa, bb = R.load_arrays(case, "a"), R.load_arrays(case, "b")
            check = {k: case[k] for k in ("experiment", "path", "w", "year")}
            for kind in ("fire", "other"):
                a, b = aa[kind + "_count"], bb[kind + "_count"]
                assert a.shape == b.shape == (15, 58)
                assert np.array_equal(a, b), (case, kind)
                check[kind + "_seed_by_tract_count_exact"] = True
            cohort_checks.append(check)
        if case["experiment"] != "E1":
            continue
        new_row = paired[(paired.experiment == "E1") & (paired.path == case["path"])
                         & (paired.w == case["w"]) & (paired.year == case["year"])].iloc[0]
        for side in ("a", "b"):
            old = verification[(verification.draw == f"{case['path']} w={case['w']:g}")
                               & (verification.year == case["year"]) & (verification.policy == case[side])]
            assert len(old) == 1
            measured = {m: new_row[side + "_" + m] for m in R.METRICS} if case["identical"] else R.pooled(R.load_arrays(case, side), weights)
            for metric in R.METRICS:
                error = abs(measured[metric] - old.iloc[0][metric + " sim"])
                assert error < 1e-12
                assert abs(measured[metric] - new_row[side + "_" + metric]) < 1e-12
                reproduction.append({"path": case["path"], "w": case["w"], "year": case["year"],
                                     "policy": case[side], "metric": metric, "absolute_difference": error,
                                     "level_source": "configuration identity reuse" if case["identical"] else "raw NPZ reaggregation"})
    evidence["per_seed_cohorts"] = cohort_checks
    evidence["verification_reproduction"] = {"checks": reproduction,
        "max_abs_difference_by_metric": {m: max(r["absolute_difference"] for r in reproduction if r["metric"] == m) for m in R.METRICS}}
    targets = {}
    for metric, excess in (("M_F", "fire over 2025 sim"), ("M_O", "other over 2025 sim")):
        values = verification[metric + " sim"] - verification[excess]
        difference = float(np.max(np.abs(values - plan["target_sim"][metric])))
        assert difference < 1e-12
        targets[metric] = {"reference": plan["target_sim"][metric], "rows": len(values),
                           "minimum": float(values.min()), "maximum": float(values.max()),
                           "max_abs_difference_from_saved_reference": difference}
    evidence["simulator_targets"] = targets

    tasks = R.tasks_from(plan)
    expected_raw = {R.raw_path(key, seed, ident) for key, seed, ids in tasks for ident in ids}
    assert expected_raw == set(R.RAW.glob("*.npz")) and len(expected_raw) == 225
    logs = [json.loads(line) for line in (CEUS / "logs" / "experimental_simulations.jsonl").read_text().splitlines()]
    expected_groups = {(R.label(key[:2], key[2]), seed): len(ids) for key, seed, ids in tasks}
    assert len(logs) == len(expected_groups) == 105
    assert {(x["case"], x["seed"]) for x in logs} == set(expected_groups)
    assert all(x["simulated"] == expected_groups[(x["case"], x["seed"])] for x in logs)
    assert sum(x["simulated"] for x in logs) == 225
    receipt_path = HERE / "raw_hash_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["original_manifest_sha256"] == manifest["original_manifest_sha256"]
    assert receipt["manifest_created_utc"] == manifest["created_utc"]
    raw_receipt = {row["file"]: row for row in receipt["files"]}
    assert len(raw_receipt) == len(receipt["files"])
    assert set(raw_receipt) == {path.relative_to(CEUS).as_posix() for path in expected_raw}
    waiting = []
    for path in sorted(expected_raw):
        row = raw_receipt[path.relative_to(CEUS).as_posix()]
        assert row["bytes"] == path.stat().st_size, f"File size mismatch: {path}"
        assert row["sha256"] == R.sha(path), f"SHA-256 mismatch: {path}"
        with np.load(path) as data:
            waiting.append((int(data["fire_waited"]), int(data["other_waited"])))
    evidence["raw_provenance"] = {"raw_files": len(raw_receipt), "logged_groups": len(logs),
        "sum_logged_simulated": sum(x["simulated"] for x in logs), "all_groups_generated_every_required_configuration": True,
        "raw_hash_receipt_sha256": R.sha(receipt_path)}

    # The proposed alignment failure is checked against the frozen source contract.
    tree = ast.parse(inspect.getsource(R.environment.simulate))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "simulated" for t in n.targets)]
    assert len(assignments) == 1
    assert any(k.arg == "index" and ast.unparse(k.value) == "runs.index" for k in assignments[0].value.keywords)
    reorders = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "simulated"
                and n.func.attr in ("reset_index", "sort_index", "sort_values", "reindex", "drop", "drop_duplicates")]
    assert not reorders
    assert ast.unparse(tree.body[0].body[-1]) == "return simulated"
    assert (verification["waiting fire sim"] == 0).all()
    assert all(fire == 0 and other == 0 for fire, other in waiting)
    evidence["waiting"] = {"original_verification_fire_rows": len(verification),
        "original_fire_waiting_all_zero": True, "original_other_waiting_column_available": False,
        "raw_fire_and_other_first_dispatch_waiting_all_zero": True, "raw_files_checked": len(waiting),
        "simulated_uses_original_runs_index_and_returns_same_table": True,
        "source_index_reorder_calls": len(reorders),
        "statistic": "First-dispatched run delayed by more than one second, among arriving tract-located incidents.",
        "limit": "Raw NPZ files do not retain run-level timestamps or delay arrays; zero saved counts do not establish absence of every queued run."}
    planning = ROOT / "code" / "dispatch_simulator" / "planning.py"
    environment = ROOT / "code" / "dispatch_simulator" / "environment.py"
    scenarios = ROOT / "code" / "dispatch_simulator" / "scenarios.py"
    evidence["source_anchors"] = [
        source_anchor(planning, "KAPPA ="), source_anchor(planning, "TARGET_TOLERANCE_MIN ="),
        source_anchor(planning, "self.fire_tract_rate ="), source_anchor(planning, "fire_weight = np.bincount"),
        source_anchor(environment, "simulated = pd.DataFrame(np.nan, index=runs.index"),
        source_anchor(environment, "return simulated"), source_anchor(planning, "delay = simulated["),
        source_anchor(planning, "first_dispatch ="), source_anchor(planning, "first_dispatch[\"delay\"] > 1.0"),
        source_anchor(scenarios, "reset_index(drop=True)"), source_anchor(planning, "planned_scenario = dict(scenario)")]
    assert R.P.KAPPA == 10.0 and R.P.TARGET_TOLERANCE_MIN == 0.05
    evidence["reward_constants"] = {"KAPPA": R.P.KAPPA, "tolerance_min": R.P.TARGET_TOLERANCE_MIN,
        "location": "code/dispatch_simulator/planning.py", "KAPPA_in_setup_json": "KAPPA" in json.loads((R.MODEL_DIR / "setup.json").read_text()),
        "interpretation": "Reward penalizes excess above the target; the reporting tolerance is separate."}
    assert R.sha(manifest_path) == manifest_hash
    assert all(R.sha(ROOT / e["path"]) == e["sha256"] for e in manifest["inputs"])
    evidence["status"] = "passed"
    print(json.dumps({k: evidence[k] for k in ("status", "event_simulations_added", "training_runs_added", "replay", "fixed_weights", "raw_provenance")}, indent=2))
    print("Original verification max differences:", evidence["verification_reproduction"]["max_abs_difference_by_metric"])


if __name__ == "__main__":
    main()
