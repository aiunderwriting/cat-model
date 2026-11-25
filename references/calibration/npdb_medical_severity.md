# NPDB payment-report tail evidence

## Purpose and scope

This page reports an interval-censored GPD fit to NPDB malpractice payment reports. It is analog evidence for the `liability_enforcement` tail class: the observable unit is a payer-practitioner payment report, an imperfect analog of any scenario's incident severity.

## Provenance

The fit comes from an interval-censored GPD analysis that is not part of this repository. The raw NPDB file is restricted by HRSA's data use agreement and is not in the repository either, so the analysis cannot be reproduced from the repository alone. Its inputs were:

- the NPDB public-use file `NPDB2601.CSV`, SHA-256 `1409fc66686b637033cf1d61d4e478e46fb5e6ed308a61c34f793a7b6b4f044b`, read with the [public-use file format specifications](https://npdb.hrsa.gov/resources/puf/pufFormatSpecifications.jsp);
- annual-average CPI-U for the U.S. city average, all items (`CUUR0000SA0`, 1982-84=100), 2004 through 2025, retrieved from the [BLS Public Data API](https://www.bls.gov/developers/) on 2026-07-11, SHA-256 `13f614d1148c89b839aec382d7409ef5d5a3343943da4613f6a01a76d0a82490`.

## Data and treatment

- Input: `data/external/npdb/NPDB2601.CSV`
- Window: 2015-2024; amounts on-leveled to 2025 annual CPI-U
- Amount field: TOTALPMT
- Selected reports: 109,015
- Published bands: interval-censored; $105m code: right-censored above $100m
- Unobserved policy limits: not inferred as censoring

## Base fit

- Threshold: $2,000,000, the lowest candidate above the $1m spike, which balances threshold stability and sample size
- Complete-band observations: 2,861
- GPD xi: 0.5343
- GPD sigma: $1,010,461
- 90% bootstrap xi interval: [0.4876, 0.5778]
- Bootstrap seed: 20260711; succeeded: 500/500

The bootstrap interval reflects resampling variation at the fixed $2m threshold. It does not reflect threshold choice, policy-limit compression or unit transfer; the threshold spread below is wider.

## Threshold stability

| Threshold | N | xi | sigma |
|---:|---:|---:|---:|
| $500,000 | 32,427 | 0.3705 | $414,309 |
| $1,000,000 | 13,803 | 0.6348 | $349,627 |
| $2,000,000 | 2,861 | 0.5343 | $1,010,461 |
| $5,000,000 | 570 | 0.4511 | $2,333,305 |

## Payment-basis sensitivities

| Amount field | Single-payment only | N | xi | sigma |
|---|:---:|---:|---:|---:|
| PAYMENT | no | 2,550 | 0.5100 | $1,049,132 |
| TOTALPMT | yes | 2,556 | 0.5356 | $1,021,296 |
| PAYMENT | yes | 2,399 | 0.5114 | $1,049,700 |

## Reliance and limitations

The fit informs a tail-shape judgment; no code reads this page. It does not bridge a payment report to a patient occurrence, does not identify case clusters, and does not recover ground-up loss censored by unknown policy limits or state damage caps. Any use as more than a prior for a tail shape needs a separately supported bridge from reports to patient occurrences, and sensitivity tests.
