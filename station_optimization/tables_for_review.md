# Paper tables

### fast_policy_levels

Policy outcomes from the approximate queueing evaluator on 48 new growth draws. Each draw contains four evaluations (2030, 2035, 2040, and 2045); outcomes are averaged over those years and draws. $D_F$ is the fixed-fire-rate-weighted standard deviation of tract mean fire travel times; $M_F$ and $M_O$ are regional mean travel times for study-fire and other calls. All three are in minutes. Return sums the four original stage rewards; higher is better. Each stage cost is $D_F$ plus ten times the positive excess of each mean above its 2025 target (6.6962 min for fire and 5.9992 min for other calls). The penalty starts at zero excess. Actions count station and vehicle planning changes per draw. A target violation occurs when either mean exceeds its own 2025 baseline by more than 0.05 min. No action retains the initial configuration; the Station 9 policy moves that station once; stage greedy chooses the feasible action pair with the lowest next-stage cost; frozen PPO applies the trained proximal policy optimization policy. Greedy and PPO share stage action limits. Gains against no action combine location changes and fleet additions.

| Policy | $D_F$ | $M_F$ | $M_O$ | Return | Actions | Violations |
| --- | --- | --- | --- | --- | --- | --- |
| No action | 1.872 | 6.711 | 6.056 | -10.398 | 0.0 | 113/192 |
| Modeled Station 9 relocation | 1.825 | 6.764 | 6.109 | -14.427 | 1.0 | 185/192 |
| Stage greedy | 1.662 | 6.597 | 5.967 | -6.657 | 8.0 | 0/192 |
| Frozen PPO | 1.667 | 6.558 | 5.942 | -6.675 | 8.0 | 0/192 |

### e1_policy_contrasts

Direct frozen-PPO minus stage-greedy contrasts in discrete-event simulation (E1). Negative values favor PPO. Entries are differences in minutes with 95% paired bootstrap intervals. $D_F$ measures fixed-fire-rate-weighted dispersion of tract mean fire travel; $M_F$ and $M_O$ measure regional mean fire and other-call travel. Both policies share the same 15 incident streams (seeds 101--115); 2,000 bootstrap draws resample whole streams and recompute pooled tract means and dispersion. Medium and High denote county population paths; spatial growth mix $w=0$ allocates growth in proportion to population and $w=1$ follows positive historical growth shares. Years are evaluation years. Zero intervals at identical configurations express deterministic equality, with no extra simulation. Intervals describe incident-stream variation within each stated scenario.

| Growth scenario | Year | $\Delta D_F$ [95\% CI] | $\Delta M_F$ [95\% CI] | $\Delta M_O$ [95\% CI] |
| --- | --- | --- | --- | --- |
| Medium, $w=1$ | 2035 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| Medium, $w=1$ | 2045 | +0.0014 [-0.0107, +0.0110] | -0.0875 [-0.1006, -0.0757] | -0.0476 [-0.0510, -0.0445] |
| High, $w=1$ | 2035 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| High, $w=1$ | 2045 | -0.0033 [-0.0230, +0.0116] | -0.1998 [-0.2223, -0.1812] | -0.1579 [-0.1599, -0.1560] |
| High, $w=0$ | 2035 | +0.0011 [-0.0002, +0.0027] | +0.0008 [-0.000087, +0.0020] | +0.000080 [+0.000026, +0.0001] |
| High, $w=0$ | 2045 | +0.0215 [-0.0011, +0.0376] | -0.1109 [-0.1248, -0.0976] | -0.0822 [-0.0849, -0.0795] |

### e1_resource_configurations

Resource configurations used in E1. Each entry gives active station/base locations, engines (E), and quints (Q) in that order. Both policies start with 25 locations, 23 engines and 5 quints. Each five-year stage allows at most one station action (build, relocate, or hold) and one vehicle addition (engine or quint, or hold). Building includes one engine, so equal action limits permit different realized fleet sizes and compositions. The final column is PPO minus greedy in total vehicles. Policy contrasts include the consequences of these resource choices. Medium and High are county population paths; $w=0$ allocates growth in proportion to population and $w=1$ follows positive historical growth. Years denote evaluation years.

| Growth scenario | Year | PPO: locations / E / Q | Greedy: locations / E / Q | $\Delta$ vehicles |
| --- | --- | --- | --- | --- |
| Medium, $w=1$ | 2035 | 27 / 27 / 5 | 27 / 27 / 5 | +0 |
| Medium, $w=1$ | 2045 | 28 / 30 / 5 | 27 / 29 / 5 | +1 |
| High, $w=1$ | 2035 | 27 / 27 / 5 | 27 / 27 / 5 | +0 |
| High, $w=1$ | 2045 | 29 / 31 / 5 | 27 / 28 / 6 | +2 |
| High, $w=0$ | 2035 | 27 / 26 / 6 | 27 / 27 / 5 | +0 |
| High, $w=0$ | 2045 | 29 / 30 / 6 | 28 / 29 / 6 | +1 |

### e2_action_contrasts

Alternative engine-addition destinations after the common 2035 station action (E2). Action A adds an engine at the modeled E62 base; action B adds one at CO_29. The table reports A minus B for each evaluation year and their arithmetic mean, in minutes; negative values favor A. Chosen denotes the frozen PPO policy's observed selection. Both states use the Medium county population path and the same resource configuration: $w=0$ uses population-proportional growth and $w=1$ uses positive historical-growth shares. All branches continue with the same frozen policy. $D_F$ is fixed-fire-rate-weighted dispersion of tract mean fire travel; $M_F$ and $M_O$ are regional mean fire and other-call travel. The 95% intervals use 2,000 paired whole-stream bootstrap draws over 15 streams, with the same resampled seed indices across the two years. These contrasts evaluate the specified alternative actions and continuation rule.

| Growth | Chosen | Year | $\Delta D_F$ [95\% CI] | $\Delta M_F$ [95\% CI] | $\Delta M_O$ [95\% CI] |
| --- | --- | --- | --- | --- | --- |
| $w=0$ | A | 2040 | +0.0263 [+0.0122, +0.0397] | +0.0168 [+0.0085, +0.0249] | -0.0008 [-0.0013, -0.0003] |
| $w=0$ | A | 2045 | -0.0094 [-0.0158, -0.0031] | -0.0113 [-0.0155, -0.0070] | -0.0050 [-0.0061, -0.0039] |
| $w=0$ | A | Mean | +0.0084 [-0.0013, +0.0175] | +0.0028 [-0.0030, +0.0085] | -0.0029 [-0.0036, -0.0022] |
| $w=1$ | B | 2040 | +0.0262 [+0.0120, +0.0398] | +0.0168 [+0.0086, +0.0250] | +0.0003 [-0.0004, +0.0009] |
| $w=1$ | B | 2045 | -0.0080 [-0.0146, -0.0016] | -0.0102 [-0.0148, -0.0055] | -0.0023 [-0.0029, -0.0018] |
| $w=1$ | B | Mean | +0.0091 [-0.0006, +0.0183] | +0.0033 [-0.0025, +0.0094] | -0.0010 [-0.0016, -0.0005] |

### e2_fast_objective

Approximate-evaluator outcomes for the same E2 branches. The dispersion contrast is A minus B averaged over 2040 and 2045; negative favors A. Remaining return sums the original stage rewards over those two years; higher is better. Each reward is minus the sum of fire-travel dispersion and the original penalty for regional fire and other-call means above their 2025 targets. Penalty A/B sums these penalties over both years; each is ten times positive mean excess, with no 0.05-min deadband. The 0.05-min tolerance applies only to violation reporting. Return and penalty use minute-based objective units, not additional physical travel time. A adds one engine at E62 and B at CO_29, followed by the frozen policy. The two states share the Medium county growth path and post-relocation branch configuration; $w=0$ is population-proportional growth and $w=1$ follows positive historical-growth shares.

| Growth | Chosen | $\Delta D_F$ (min) | Penalty A | Penalty B | Return A | Return B |
| --- | --- | --- | --- | --- | --- | --- |
| $w=0$ | A | +0.00127 | 0.08727 | 0.14497 | -3.24813 | -3.30330 |
| $w=1$ | B | +0.00157 | 0.00000 | 0.00000 | -3.16815 | -3.16501 |

### spatial_consequences

Distribution of local fire-travel changes in 2045. Shares sum the fixed baseline fire-demand weights of the 57 tracts with positive fire rates; each row totals 100% apart from rounding. Faster and slower identify lower and higher tract mean travel under frozen PPO than under the stated reference policy. Approximate-evaluator rows compare PPO with no action; event-simulator rows compare PPO with stage greedy using the E1 pooled outputs. Unchanged means equality within numerical precision ($10^{-10}$ min). Categories describe point-estimate signs, rather than tract-level significance tests. Medium and High are county population paths; $w=0$ uses population-proportional growth and $w=1$ uses positive historical-growth shares. Changes in spatial dispersion and regional means can coexist with longer travel times in some tracts.

| Growth scenario | Comparison | Faster | Slower | Unchanged |
| --- | --- | --- | --- | --- |
| Medium, $w=1$ | PPO--no action (approx.) | 62.9\% | 37.1\% | 0.0\% |
| High, $w=1$ | PPO--no action (approx.) | 100.0\% | 0.0\% | 0.0\% |
| High, $w=0$ | PPO--no action (approx.) | 100.0\% | 0.0\% | 0.0\% |
| Medium, $w=1$ | PPO--greedy (sim.) | 52.3\% | 35.8\% | 11.9\% |
| High, $w=1$ | PPO--greedy (sim.) | 65.6\% | 19.3\% | 15.0\% |
| High, $w=0$ | PPO--greedy (sim.) | 51.7\% | 27.3\% | 21.0\% |
