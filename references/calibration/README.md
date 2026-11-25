# Analog evidence

Empirical and analog evidence behind the tail-shape judgments. No command reads these values; a study that relies on one captures its source in its source bundle and states its population, dates, transformations and limitations.

## NPDB malpractice payment tail

The reference analysis uses 109,015 type-P `TOTALPMT` payment reports from 2015-2024 in the NPDB public-use release that holds reports through March 31, 2026. Published dollar codes are treated as intervals, the `$105m` code is right-censored above `$100m`, and each band is on-leveled by `ORIGYEAR` to 2025 annual-average CPI-U dollars. Bands that straddle a candidate threshold are excluded, and the grouped likelihood conditions on the first complete band for each report year. Unknown policy limits are not imputed as censoring, because the public file does not identify them.

At a `$2m` threshold, 2,861 complete-band observations give `xi = 0.534` and `sigma = $1.01m`; the seeded 90% nonparametric bootstrap interval for `xi` is `[0.488, 0.578]`. Threshold estimates are `0.370` at `$0.5m`, `0.635` at `$1m`, `0.534` at `$2m` and `0.451` at `$5m`, a spread that the governed shape alternatives, 0.20 either side of 0.5, cover. At the `$2m` threshold, `PAYMENT` gives `xi = 0.510`, single-payment `TOTALPMT` gives `0.536` and single-payment `PAYMENT` gives `0.511`, so the payment field and payment count matter much less than the threshold. The fit's report, [`npdb_medical_severity.md`](npdb_medical_severity.md), records its inputs and why it cannot be reproduced from the repository alone.

NPDB rows are payer-practitioner payment reports: the public file has no case identifier, and the amounts are neither ground-up patient losses nor policy occurrences. Policy limits and state damage caps compress payments relative to gross compensatory loss. The fit is therefore per-claim liability evidence for the `liability_enforcement` tail shape, and context for one patient's harm in `output_reliance_health`. That scenario counts one identified defect as one incident and takes the `cyber_operational` shape, because its largest incidents merge the injuries of everyone who relied on the defect.

## Published tail-index evidence

The tail classes in [`scenarios/severity_governance_policies.yaml`](../../scenarios/severity_governance_policies.yaml) cite these analog estimates. None is same-unit AI loss experience, and each depends on its threshold, data source and selection.

| Domain | Estimate | Source and limits |
|---|---|---|
| Malpractice payments | ξ = 0.534 at USD 2m (90% bootstrap 0.488–0.578; 0.37–0.63 across thresholds) | NPDB payment reports ([above](#npdb-malpractice-payment-tail)); limits and damage caps compress payments |
| Bank operational losses | ξ 1.01–1.39 in six of eight Basel business lines | Moscadelli (2004), Table 5, also reproduced in [arXiv:2403.20171](https://arxiv.org/abs/2403.20171); see Nešlehová, Embrechts and Chavez-Demoulin (2006) |
| US cyber-event losses | Infinite-mean tails across sectors and risk categories | Malavasi et al. (2022), Advisen data |
| Deaths per terrorist attack | Density exponent 2.4 ± 0.2, so ξ = 1/(α − 1) ≈ 0.6–0.8 | Clauset and Wiegel (2010) |
| Pandemic deaths | ξ ≈ 1.6 for an unbounded dual variable mapped to deaths bounded by world population | Cirillo and Taleb (2020); Corral (2020) finds the power-law evidence inconclusive |

Dollar-loss estimates at or above 1 imply an infinite uncapped mean, which a governed tail can represent only with a finite endpoint. The casualty-count estimates describe deaths, not economic loss.

- Cirillo, P., and Taleb, N. N. (2020). Tail risk of contagious diseases. *Nature Physics* 16, 606–613.
- Clauset, A., and Wiegel, F. W. (2010). A generalized aggregation-disintegration model for the frequency of severe terrorist attacks. *Journal of Conflict Resolution* 54(1).
- Corral, Á. (2020). Scientific comment on "Tail risk of contagious diseases". arXiv:2007.06876.
- Malavasi, M., Peters, G. W., Shevchenko, P. V., Trück, S., Jang, J., and Sofronov, G. (2022). Cyber risk frequency, severity and insurance viability. *Insurance: Mathematics and Economics* 106, 90–114.
- Moscadelli, M. (2004). The modelling of operational risk: experience with the analysis of the data collected by the Basel Committee. Banca d'Italia, Temi di discussione 517.
- Nešlehová, J., Embrechts, P., and Chavez-Demoulin, V. (2006). Infinite mean models and the LDA for operational risk. *Journal of Operational Risk* 1(1), 3–25.
