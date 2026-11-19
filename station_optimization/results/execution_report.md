# Execution record: frozen-policy E1 and E2

Status: complete. Date: 2026-11-16. This file is a technical execution record, separate from the manuscript.

## 1. Scope and completed work

The run implemented the fixed design in `../task.md`. It used the existing planning model, setup, action masks, candidate locations, demand generator, dispatch rules and reward. It added zero RL training runs. It added no scenarios, policy baselines, seeds, case searches or research experiments beyond E1 and E2.

- E1 covered the three named draws and the two prescribed years: six checkpoints. Two checkpoint pairs had identical complete configurations and therefore have zero policy contrasts; their levels were reused from the original verification table. Four pairs required 120 configuration–stream evaluations.
- E2 covered the prespecified Medium w=0 and w=1 states, A/B vehicle actions, and 2040/2045 outcomes. Its nominal 120 evaluations shared 15 exact configuration–scenario–year–seed outcomes with E1, adding 105 unique evaluations.
- Total: **225 unique configuration–stream evaluations**, **105 stream groups**, **225 raw NPZ files**, and **15 seeds (101–115)**. The planned upper bound was 300. The log records 225 newly simulated evaluations.
- One process invocation ran to exit code 0. Its exec session was 16260. The preparation step took 17.9 seconds. The simulation log records 10074.4 cumulative worker-seconds across eight workers; this is not elapsed wall time.

The original model and source/input files were read-only. All 31 manifest hashes, including the wrapper and frozen task, matched at completion. `original_inputs_unchanged` is true. The model checkpoint remains `data/models/planning/final/model.zip`.

## 2. Source and interface flow

The command, from the project root, was:

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -u station_optimization/run_experiments.py all > station_optimization/logs/experimental_run.log 2>&1
```

Preparation reused `environment.build_world → scenarios.fit_generator → planning.prepare`, which reconstruct the existing simulator inputs and reads the frozen planning setup. The wrapper loaded `MaskablePPO` for deterministic inference. It read the existing `baselines.pkl` directly and did not invoke the baseline-cache writer. The execution added zero policy-optimization steps; simulator-input preparation is the original source workflow.

Simulation reused `grown_scenario → simulator_inputs → environment.simulate(..., policies.closest_engine) → simulated_outcome`. A scenario was generated once per path, spatial weight, year and seed and reused by the required configurations. Complete configuration hashes plus scenario/year/seed identify raw outputs. The same configurations were deduplicated across E1 and E2. Each raw file was written through a temporary file followed by rename.

The frozen trajectories reproduced all 32 prescribed station/vehicle action names and all 48 D_F/M_F/M_O values across the 16 required learned-policy stages. The direct action/metric check is saved in `frozen_replay_checks.json`. The first newly completed E1 checkpoint also reproduced the original verification levels to floating-point precision.

The E2 branches use direct `env.step()` continuation after the vehicle intervention. They retain the same frozen continuation rule. In this case, all four branches subsequently choose build C12 and add an engine at CO_29. No `play()` reset is used after branching.

## 3. Estimation and coverage

D_F is the weighted standard deviation of tract means, with the original fixed fire-demand weights. There are 58 tract geometries and 57 positive-weight fire tracts. For each branch, the 15 streams’ tract travel sums and incident counts are pooled before tract means and D_F are calculated. M_F and M_O also use pooled sums and counts. They are first-arriving eligible E/Q **travel** times, measured from en route to arrival.

The 2,000 bootstrap samples use RNG seed 7 and resample whole event streams. Each draw applies the same 15 indices to the compared branches; E2 also shares these indices across years and states. Each replicate recomputes pooled tract means and D_F. The 95% intervals are percentile intervals for direct paired contrasts. They describe event-stream variation conditional on the fixed growth scenario and frozen policy.

Every newly simulated pooled branch covered all 57 positive-weight tracts. All 2,000 bootstrap replicates produced finite D_F, M_F and M_O for every comparison: invalid counts are **0**. Individual streams can omit up to 2 positive-weight tracts; those stream-level D_F values remain missing in `paired_seed_metrics.csv`. They are not averaged to obtain the reported D_F or intervals. The zero-weight tract retains missing travel means where applicable.

All 16 newly simulated branch levels met the original mean-target point checks with the 0.05-minute tolerance. The fixed simulator reference levels were M_F=6.698115377 and M_O=6.001014501 minutes. The saved first-dispatch waiting counts were zero; travel time remains the evaluated outcome. This statistic concerns the earliest dispatched row of arriving, tract-located incidents and does not measure every queued unit. The tolerance is a mean-target reporting rule, not a policy-equivalence threshold.

## 4. E1: direct policy comparisons

Negative values indicate lower values under RL. D_F remains in minutes; mean-time differences below are converted to seconds.

| Path | w | Year | D_F: RL−greedy, min [95% CI] | M_F: RL−greedy, s [95% CI] | M_O: RL−greedy, s [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Medium | 1 | 2035 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| Medium | 1 | 2045 | +0.001437 [-0.010699, +0.011046] | -5.251358 [-6.035933, -4.542213] | -2.858617 [-3.057967, -2.671221] |
| High | 1 | 2035 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| High | 1 | 2045 | -0.003346 [-0.022958, +0.011558] | -11.990177 [-13.339925, -10.872554] | -9.475058 [-9.595805, -9.360372] |
| High | 0 | 2035 | +0.001096 [-0.000153, +0.002709] | +0.046621 [-0.005236, +0.118171] | +0.004787 [+0.001537, +0.008662] |
| High | 0 | 2045 | +0.021547 [-0.001059, +0.037603] | -6.656627 [-7.485288, -5.856948] | -4.930981 [-5.092161, -4.772278] |

The two identical-configuration checkpoints have exact zero contrasts by configuration identity. Their zero intervals are not empirical claims about stochastic precision. At each of the four nonidentical checkpoints, the D_F interval includes zero. The three 2045 checkpoints have lower M_F and M_O under RL, with intervals below zero. High w=0 in 2035 has a small M_F contrast whose interval includes zero; its M_O difference is about 0.0048 seconds.

Both policies use the same stage action limits. Their final fleets differ because a new station includes an engine. The comparison therefore includes resource expansion as well as location and vehicle-type choices:

| Path | w | Year | RL: active locations / fleet | Greedy: active locations / fleet |
| --- | --- | --- | --- | --- |
| Medium | 1 | 2035 | 27 / 27E + 5Q | 27 / 27E + 5Q |
| Medium | 1 | 2045 | 28 / 30E + 5Q | 27 / 29E + 5Q |
| High | 1 | 2035 | 27 / 27E + 5Q | 27 / 27E + 5Q |
| High | 1 | 2045 | 29 / 31E + 5Q | 27 / 28E + 6Q |
| High | 0 | 2035 | 27 / 26E + 6Q | 27 / 27E + 5Q |
| High | 0 | 2045 | 29 / 30E + 6Q | 28 / 29E + 6Q |

The E1 results support a comparison of the frozen policies under the implemented action constraints. They do not isolate a location effect at an equal final fleet or equal monetary cost. The six checkpoints do not replace the existing 48-draw fast-evaluator comparison. The tract table preserves areas that improve and areas that become slower.

## 5. E2: observed selection and action consequences

The prespecified selection was reproduced: S0 (Medium, w=0) chose A, an engine at the modeled E62 base; S1 (Medium, w=1) chose B, an engine at CO_29. E62 is a source-derived modeled base label, not a verified official station number. Both states shared the 2025/2030 actions, the 2035 move GFR_01→C18, decision time, resource configuration and county population totals. Their complete spatial growth distributions differed. Both candidate vehicle actions passed the original masks.

The selection difference is separate from the evidence about consequences. The following event-simulator effects use A−B:

| w | Year | D_F: A−B, min [95% CI] | M_F: A−B, min [95% CI] | M_O: A−B, min [95% CI] |
| --- | --- | --- | --- | --- |
| 0 | 2040 | +0.026303 [+0.012163, +0.039720] | +0.016815 [+0.008523, +0.024938] | -0.000819 [-0.001337, -0.000276] |
| 0 | 2045 | -0.009444 [-0.015823, -0.003076] | -0.011305 [-0.015526, -0.007013] | -0.004964 [-0.006075, -0.003906] |
| 1 | 2040 | +0.026214 [+0.012018, +0.039793] | +0.016815 [+0.008567, +0.025041] | +0.000255 [-0.000419, +0.000929] |
| 1 | 2045 | -0.008037 [-0.014608, -0.001643] | -0.010200 [-0.014759, -0.005538] | -0.002349 [-0.002853, -0.001836] |

A has higher D_F in 2040 and lower D_F in 2045 in both states, with the respective intervals entirely above and below zero. These are stage-dependent consequences of the specified action and frozen continuation. The remaining two-stage average effects are:

| w | Selected | Mean remaining D_F: A−B, min [95% CI] | Mean remaining M_F: A−B, min [95% CI] | Mean remaining M_O: A−B, min [95% CI] |
| --- | --- | --- | --- | --- |
| 0 | A | +0.008429 [-0.001299, +0.017483] | +0.002755 [-0.003001, +0.008542] | -0.002891 [-0.003601, -0.002183] |
| 1 | B | +0.009088 [-0.000597, +0.018315] | +0.003308 [-0.002525, +0.009415] | -0.001047 [-0.001606, -0.000500] |

Both states’ mean remaining D_F and M_F intervals include zero. A has lower remaining M_O in both states, with intervals below zero. The between-state contrast, [A−B at w=1]−[A−B at w=0], is D_F +0.000659 [-0.000551, +0.002067], M_F +0.000552 [-0.000516, +0.001608], and M_O +0.001845 [+0.001321, +0.002445] minutes. The D_F and M_F differences between states remain unresolved by these event streams; the M_O difference is small and positive.

### Original fast objective

The original remaining return is the negative sum of the two remaining stage costs. Each stage cost is D_F + 10 × positive excess M_F above its 2025 target + 10 × positive excess M_O above its 2025 target. The reward penalty starts at the target; the separate 0.05-minute violation tolerance is not a reward dead zone.

| w | Fast remaining mean D_F: A−B (min) | Fast return A | Fast return B | Return A−B | Selected action |
| --- | --- | --- | --- | --- | --- |
| 0 | +0.001269 | -3.248128 | -3.303298 | +0.055170 | A |
| 1 | +0.001571 | -3.168153 | -3.165011 | -0.003142 | B |

At w=0 in 2040, fast M_O is 6.007972 minutes for A and 6.013743 for B, above the target of 5.999246. The smaller M_O penalty makes A’s remaining return higher despite its slightly higher mean remaining D_F. At w=1, both branches’ fast M_O remain below that target; B’s lower mean remaining D_F gives B the higher return. All four E2 branches meet the tolerance-based fast point constraints.

Thus, the original fast objective’s preference agrees with the frozen policy’s actual choice in both states. The event simulator confirms stage trade-offs and the remaining M_O direction, while the remaining D_F and M_F contrasts retain intervals spanning zero. The recorded results support a conditional account of the modeled objective and consequences. They do not establish an event-simulator ranking reversal, optimal action values or a neural-feature causal explanation.

## 6. Completion and stopping rule

The fixed 15 streams, two experiments, six E1 checkpoints and single E2 decision family are complete. The paired intervals, spatial contrasts, exact branch records and input hashes are retained. Zero new RL training runs, extra seeds, stress scenarios, SHAP analyses, policy searches or new fairness objectives were added. The results contain both benefits and costs; uncertain remaining effects are retained. The next work is manuscript and figure preparation plus the separately requested experiment review.

The run emitted environment warnings about the installed numexpr version, font cache and sandboxed CPU-information queries. They did not interrupt execution or change the successful source-output reproduction checks.

## 7. File index

Paths below are relative to this report unless marked otherwise.

| File | Role |
| --- | --- |
| ../run_experiments.py | Executable E1/E2 wrapper; all and summarize modes |
| ../task.md | Original fixed design; unchanged |
| ../inputs_manifest.json | 31 SHA-256 entries, package versions, frozen model, seeds and bootstrap settings |
| plan.pkl | Complete comparison plan, configuration dictionaries, all staged histories, fixed weights and reference levels |
| raw/*.npz | 225 raw configuration–stream outcomes; tract sums/counts, event travel arrays, waiting counts and vehicle busy shares |
| comparisons_planned.json | Six E1 and four E2 checkpoint comparisons, IDs and fast differences |
| paired_comparisons.csv | Ten formal checkpoint contrasts; pooled levels, direct paired intervals, fast contrasts and invalid-replicate counts |
| simulated_levels.csv | 16 newly simulated branch levels; waiting shares and original mean-target point checks |
| paired_seed_metrics.csv | 120 branch-pair stream records; per-stream missingness retained |
| spatial_contrasts.csv | 580 tract-by-comparison differences; fixed fire weights and pooled counts |
| e2_remaining_horizon.csv | Two-stage E2 mean effects, paired intervals and original fast remaining returns |
| e2_condition_contrast.csv | Difference between the two states’ A−B effects, using shared bootstrap indices |
| bootstrap_draw_indices.npz | Exact 2,000 × 15 resampling indices, shared across policies, states and years |
| frozen_trajectories.csv | 80 staged action/configuration-ID/metric records |
| fast_tract_outcomes.csv | 4,640 existing fast-evaluator tract records for baseline, RL and E2 branches |
| frozen_replay_checks.json | Direct check of 32 actions and 48 metrics in 16 stored learned-policy stages against rq3_actions.csv |
| e2_state_action_contract.json | Matched branching configurations, observations, masks and actual frozen-policy choices |
| tracts.geojson | 58 tract geometries, population fields, growth shares and fixed fire weights |
| stations.csv; candidate_sites.csv | Baseline resource locations and all candidate coordinates in EPSG:32617 |
| e2_growth_by_tract.csv | S0/S1 population multipliers and populations for 2025–2045 |
| model_architecture.json | Frozen planning policy dimensions, architecture, targets and data counts |
| execution_summary.json | Completion status, counts and input-integrity checks |
| ../logs/experimental_run.log | Full run output, warnings, group progress and final summaries |
| ../logs/experimental_simulations.jsonl | 105 completed groups, exact seeds, new evaluation counts and worker timings |

Reaggregation of persisted raw outputs, without simulator execution or model inference, is available through:

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python station_optimization/run_experiments.py summarize
```

## 8. Post-review execution audit

The requested audit closed the action-replay, per-seed cohort, raw-provenance, original-verification, fixed-weight and target-constant checks. It added no event simulations and no training. The original wrapper, 31-entry manifest and 225 raw outputs remain unchanged. Four fresh deterministic policy replays regenerated the existing action-check receipt byte for byte. Maximum new-versus-original E1 differences were D_F = 4.440892098500626e−16, M_F = 0 and M_O = 0 minutes. Fire and other-call seed-by-tract count arrays matched exactly for all eight nonidentical comparisons.

The evidence, source-level waiting interpretation and review-item resolutions are in [resolution.md](../reviews/experiment/resolution.md), [audit_evidence.json](../reviews/experiment/audit_evidence.json) and [raw_hash_receipt.json](../reviews/experiment/raw_hash_receipt.json). The independent executable audit is [audit_experiments.py](../reviews/experiment/audit_experiments.py). KAPPA = 10 is defined in the frozen `code/dispatch_simulator/planning.py:35`; it is not a field in `setup.json`. No equivalence margin or additional simulator-reward analysis was introduced.
