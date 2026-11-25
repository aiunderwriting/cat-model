# Results summary: Hand-checkable compound Poisson golden example

- **Setup:** `golden-poisson` (scope `golden-poisson`)
- **Study period:** 2027-01-01 to 2028-01-01 (forecast as of 2027-01-01; data cutoff 2027-01-01)
- **Scenarios (1):** `golden`
- **Model:** Example model (`model`, revision `r1`, hypothetical)
- **Currency and monetary basis:** USD (USD at study start before recovery)
- **Reference arm:** `reference`
- **Result status:** `partial`; comparison coverage `not_requested`
- **Analysis:** `<analysis>`

## Headline

| Measure | View | EAL | Status |
| --- | --- | --- | --- |
| Gross component loss | `gross` | 14 USD | available |
| Insured payment | `financial` | 5.03 USD | available |

How to read it: EAL is the expected annual loss. P(L > 0) is the probability of any loss in a year. A 1-in-N loss is the annual loss exceeded with probability 1/N in a single year of the predictive law, whose years each read fresh parameter ranks; it is not the loss exceeded once in N years under a persistent but unknown rate. The EAL parameter band shows how uncertain the EAL itself is because the elicited count laws are uncertain. A dash marks a value that is unavailable or not requested; a value in parentheses is withheld because the simulation cannot support it, and is shown only for reference. The last sections give every reason.

## Expected annual loss by scenario

Measure `gross`.

| Scenario | EAL | Share of EAL | Note |
| --- | --- | --- | --- |
| `golden` | — | — | not_requested: The view was not requested. |
| unallocated | — | — | Needs the pooled EAL and every scenario's EAL. |
| total (pooled view) | 14 | 100.0% |  |

Shares divide a scenario's EAL by the pooled EAL when it is available. With every scenario's EAL available, the scenario EALs and the unallocated amount add up to the pooled EAL.

## Annual aggregate loss (AEP), illustrative

The total loss in a year, exceeded with probability 1/N in a single predictive year. The 1-in-N losses are illustrative: they depend on the governed tail shapes, the caps and the conditional independence of scenarios far more than the expected annual loss does.

| View | EAL | P(L > 0) | 1-in-10 | 1-in-100 | 1-in-200 | 1-in-250 |
| --- | --- | --- | --- | --- | --- | --- |
| `financial` | 5.03 | 0.0954 | (0) | 50 | (50) | (100) |
| `gross` | 14 | 0.3961 | (30) | (120) | (130) | (200) |

## Largest single occurrence in a year (OEP), illustrative

The largest single loss in a year, exceeded with probability 1/N in a single predictive year. Occurrence statistics are illustrative: they depend on the governed tail shapes, the caps and the conditional independence of scenarios far more than the expected annual loss does.

| View | EAL | P(L > 0) | 1-in-10 | 1-in-100 | 1-in-200 | 1-in-250 |
| --- | --- | --- | --- | --- | --- | --- |
| `oep_financial` | 4.77 | 0.0954 | (0) | 50 | 50 | 50 |
| `oep_gross` | 12.5 | 0.3961 | (10) | 100 | 100 | 100 |

## EAL parameter band

Report only. Count-law and respondent uncertainty, with the capability forecast integrated: the EAL with each rank group's count law (one scenario at one source status) held at one rank for every year instead of integrated over it, each respondent's trajectory nodes integrated by their weights at those ranks, and the respondents mixed by weight. Its mean is the EAL. The base forecast's rate is integrated; the forecast arms show how the EAL moves under alternative forecast laws. *Independent* ranks are the model's assumption; *comonotone* ranks give every rank group the same rank, which gives the larger mean of the worst 1−p share (TVaR_p) and the smaller mean of the best. A quantile under an intermediate dependence need not lie between the two, and negative dependence between ranks is not covered. The band is not a range of annual losses and is neither a confidence nor a predictive interval; the 1-in-N losses above are single-year predictive statistics, whose ranks are redrawn every year.

Independent quantiles are Monte Carlo over ranks alone (131,072 per respondent, seed 20261029); a value in parentheses has a 95% half-width above 10.0% of it. Comonotone quantiles are exact within their reported error bounds.

| View | EAL | Independent p5 | p50 | p95 | Comonotone p5 | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gross` | 14 | 14 | 14 | 14 | 14 | 14 | 14 |

## Respondent sampling

Unavailable: No state of the setup is a respondent sample.

## Tail value at risk, illustrative

VaR and TVaR are illustrative: they depend on the governed tail shapes, the caps and the conditional independence of scenarios far more than the expected annual loss does.

TVaR is the mean loss in the worst years beyond the VaR. It is estimated as the average of the worst simulated years plus β times the analytic EAL less the simulated years' mean, over one minus the level (a control variate), and is at least the VaR. β is 0 for a view without an analytic EAL (no exact mean), leaving the sample tail mean, which is biased low for heavy tails; 1 when the law has no finite variance, so only the part below the VaR comes from the years; and otherwise the regression coefficient that minimizes the estimate's variance. Years beyond VaR counts the simulated years with loss above the VaR. Intervals are 95% batch-means intervals for simulation error, widened for a numerically integrated EAL by β times its certified error bound over one minus the level, and a TVaR whose half-width exceeds the tail gate's tolerance is withheld.

| View | Level | VaR | VaR interval | TVaR | TVaR interval | Estimator | Years beyond VaR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `financial` | 0.9 | (0) | 0 to 11.9 | 50.3 | 48.1 to 52.4 | β = 0 (no exact mean) | 1908 |
| `financial` | 0.95 | 50 | 50 to 50 | 55.1 | 53.8 to 56.5 | β = 0 (no exact mean) | 98 |
| `financial` | 0.98 | 50 | 50 to 50 | 62.9 | 59.6 to 66.2 | β = 0 (no exact mean) | 98 |
| `financial` | 0.99 | 50 | 50 to 50 | 75.7 | 69.2 to 82.3 | β = 0 (no exact mean) | 98 |
| `financial` | 0.995 | (50) | 36.7 to 63.3 | 101 | 94.6 to 108 | β = 0 (no exact mean) | 98 |
| `financial` | 0.996 | (100) | 86.2 to 114 | (103) | 100 to 109 | β = 0 (no exact mean) | 5 |
| `financial` | 0.999 | (100) | 93.3 to 107 | (112) | 102 to 123 | β = 0 (no exact mean) | 5 |
| `gross` | 0.9 | (30) | 11.9 to 48.1 | 105 | 104 to 106 | β = 0.741 (regression) | 1926 |
| `gross` | 0.95 | 100 | 100 to 100 | 118 | 116 to 120 | β = 0.137 (regression) | 688 |
| `gross` | 0.98 | 110 | 110 to 110 | 137 | 133 to 142 | β = 0.1 (regression) | 209 |
| `gross` | 0.99 | (120) | 117 to 123 | (164) | 156 to 173 | β = 0.0854 (regression) | 118 |
| `gross` | 0.995 | (130) | 109 to 151 | (207) | 198 to 216 | β = 0.0752 (regression) | 99 |
| `gross` | 0.996 | (200) | 179 to 221 | (212) | 202 to 222 | β = 0.0114 (regression) | 43 |
| `gross` | 0.999 | (210) | 197 to 223 | (237) | 219 to 255 | β = 0.0071 (regression) | 13 |
| `oep_financial` | 0.9 | (0) | 0 to 11.9 | 47.7 | 46 to 49.4 | β = 0 (no exact mean) | 1908 |
| `oep_financial` | 0.95 | 50 | 50 to 50 | 50 | 50 to 50 | β = 0 (no exact mean) | 0 |
| `oep_financial` | 0.98 | 50 | 50 to 50 | 50 | 50 to 50 | β = 0 (no exact mean) | 0 |
| `oep_financial` | 0.99 | 50 | 50 to 50 | 50 | 50 to 50 | β = 0 (no exact mean) | 0 |
| `oep_financial` | 0.995 | 50 | 50 to 50 | 50 | 50 to 50 | β = 0 (no exact mean) | 0 |
| `oep_financial` | 0.996 | 50 | 50 to 50 | 50 | 50 to 50 | β = 0 (no exact mean) | 0 |
| `oep_financial` | 0.999 | 50 | 50 to 50 | 50 | 50 to 50 | β = 0 (no exact mean) | 0 |
| `oep_gross` | 0.9 | (10) | 0 to 31.4 | 95.9 | 92.8 to 99 | β = 0 (no exact mean) | 1908 |
| `oep_gross` | 0.95 | 100 | 100 to 100 | 100 | 100 to 100 | β = 0 (no exact mean) | 0 |
| `oep_gross` | 0.98 | 100 | 100 to 100 | 100 | 100 to 100 | β = 0 (no exact mean) | 0 |
| `oep_gross` | 0.99 | 100 | 100 to 100 | 100 | 100 to 100 | β = 0 (no exact mean) | 0 |
| `oep_gross` | 0.995 | 100 | 100 to 100 | 100 | 100 to 100 | β = 0 (no exact mean) | 0 |
| `oep_gross` | 0.996 | 100 | 100 to 100 | 100 | 100 to 100 | β = 0 (no exact mean) | 0 |
| `oep_gross` | 0.999 | 100 | 100 to 100 | 100 | 100 to 100 | β = 0 (no exact mean) | 0 |

## Sensitivity

No executed sensitivity is ranked (comparison coverage `not_requested`).

Dependence bracket unavailable: The reference capture lacks an available view for every scenario; no partial portfolio dependence reference is substituted.

## Plausibility checks

Report only. Welfare EAL as a share of world output, 135T USD in 2027 (`OK`; IMF World Economic Outlook database, October 2026 edition (GDP at current prices in US dollars, World), read from the database on 2026-11-11: USD 111.906 trillion in 2024, 118.485 trillion in 2025, 128.004 trillion in 2026 and 135.093 trillion in 2027. The 2027 value is the IMF's projection, not an outturn. USD 130 to 140 trillion is a reporting sensitivity, not a forecast interval.). A scenario above 0.1% or a total above 1% is flagged for review (thresholds `INITIAL_ESTIMATE`); a flag changes no result.

| Scenario | Welfare EAL | Share of world output | Flag |
| --- | --- | --- | --- |
| Total | — | — |  |
| `golden` | — | — |  |

## Unavailable, withheld and not requested

- Statistic withheld (6): `financial tvar::0.996`, `financial tvar::0.999`, `financial var::0.996`, `financial var::0.999`, `gross tvar::0.996`, `gross var::0.996` — Unconverged at 20000 years per state: the sparsest of 16 batches has 3 years at or above the VaR, and a converged statistic needs at least 10 in each.
- Statistic withheld (2): `financial var::0.9`, `oep_financial var::0.9` — Unconverged at 20000 years per state: its 95% batch-means half-width is unbounded of the estimate, above the 10% tolerance.
- Statistic withheld (1): `financial var::0.995` — Unconverged at 20000 years per state: its 95% batch-means half-width is ±26.6% of the estimate, above the 10% tolerance.
- Statistic withheld (2): `gross tvar::0.99`, `gross var::0.99` — Unconverged at 20000 years per state: the sparsest of 16 batches has 8 years at or above the VaR, and a converged statistic needs at least 10 in each.
- Statistic withheld (2): `gross tvar::0.995`, `gross var::0.995` — Unconverged at 20000 years per state: the sparsest of 16 batches has 4 years at or above the VaR, and a converged statistic needs at least 10 in each.
- Statistic withheld (2): `gross tvar::0.999`, `gross var::0.999` — Unconverged at 20000 years per state: the sparsest of 16 batches has 1 years at or above the VaR, and a converged statistic needs at least 10 in each.
- Statistic withheld (1): `gross var::0.9` — Unconverged at 20000 years per state: its 95% batch-means half-width is ±60.5% of the estimate, above the 10% tolerance.
- Statistic withheld (1): `oep_gross var::0.9` — Unconverged at 20000 years per state: its 95% batch-means half-width is ±214.5% of the estimate, above the 10% tolerance.
- View not requested (2): `welfare`, `net_resource_cost` — The execution plan does not request this view.
- Sensitivity family not requested (18): `epistemic_tail_width`, `tail_shape`, `large_loss_anchor`, `very_large_loss_anchor`, `large_loss_anchor_ignored`, `uncapped_severity`, `severity_cap`, `severity_endpoint_stress`, `lab_loss_cap`, `lab_loss_provider_only`, `model_pooling_leave_one_out`, `epistemic_spread_calibration`, `common_frequency_bias`, `count_family`, `model_pooling`, `exposure_scale`, `safeguard_level`, `public_forecast` — The execution plan froze no requirement for this family.

## Limitations

1. Hand-checkable software example; every number is stipulated, none is an empirical estimate. _(setup)_
2. No sensitivity arm was requested, so this result is a base case only; report it beside at least one executed perturbation of each material assumption. See `docs/limitations.md#4-the-main-pipeline-challenges-only-some-of-its-assumptions`. _(run)_
3. Tail shapes, caps, quantile extrapolation and spread factors are judgments. VaR, TVaR and the occurrence statistics depend on them far more than EAL does and are illustrative. The EAL parameter band shows how the count laws' and respondents' uncertainty moves the EAL, with the capability forecast integrated; it is not a range of annual losses and is neither a confidence nor a predictive interval. See `docs/limitations.md#3-tail-shapes-caps-and-spread-factors-are-judgments`. _(model)_
4. Where a capability forecast conditions the result, it rests on few frontier releases and a judgmental pace shape. See `docs/limitations.md#6-the-capability-forecast-rests-on-few-releases`. _(model)_
5. Only the setup's populations are quantified; populations outside it are unquantified, not counted as zero. See `docs/limitations.md#7-the-model-quantifies-selected-populations-only`. _(model)_
6. Numerical standard errors and intervals cover simulation error only; they exclude parameter, elicitation, forecast and model uncertainty. See `docs/limitations.md#9-numerical-precision-at-default-budgets-is-modest`. _(model)_
7. Results are research estimates, not actuarial work product. See `docs/limitations.md#10-results-are-research-estimates-not-actuarial-work-product`. _(model)_
