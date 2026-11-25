# Golden Poisson example

This is the smallest end-to-end example whose results can be checked by hand. Every input is stipulated, so it checks the software and says nothing about risk. `tests/test_golden_poisson.py` runs the commands below and compares the engine's outputs with the values on this page, which it recomputes without engine code.

## Inputs

| Input | Value |
| --- | --- |
| Annual incident count `N` | Poisson, rate λ = 0.5 per year |
| Ground-up loss per incident `X` | 10 with probability 0.8, 100 with probability 0.2, independent of `N` and of other incidents |
| Policy terms | per-occurrence retention 20, per-occurrence limit 50, no annual terms |
| Payout per incident `Y = min(max(X − 20, 0), 50)` | 0 when `X = 10`; 50 when `X = 100` |
| Simulation | 20,000 years, seed 20261029, one fixed state |

[`setup.json`](setup.json) authors the loss as a native severity mixture of two degenerate five-quantile laws (every knot 10; every knot 100) with weights 0.8 and 0.2, which is exactly the two-point law. [`plan.json`](plan.json) requests four views: the annual aggregate (`gross`), the annual largest occurrence (`oep_gross`), the annual payout (`financial`) and the annual largest occurrence payout (`oep_financial`). Each incident has exactly one loss, so an occurrence is an incident.

## Run it

From the repository root, build the setup, run it and replay the capture with the two tail levels derived here:

```sh
uv run --frozen catmodel setup build \
  --spec config/setups/golden-poisson/setup.json \
  --plan config/setups/golden-poisson/plan.json \
  --setup-id golden-poisson --output /tmp/golden-setup
uv run --frozen catmodel run --setup /tmp/golden-setup --output /tmp/golden-run
uv run --frozen catmodel results replay --setup /tmp/golden-setup \
  --capture /tmp/golden-run/capture --level 0.95 --level 0.98 \
  --output /tmp/golden-replay
```

## Hand derivation

Write `e^−0.5 = 0.6065307` and `e^−0.1 = 0.9048374`.

### Expected annual loss, variance and loss probability

`E[X] = 0.8 × 10 + 0.2 × 100 = 28`, so `EAL = λ E[X] = 0.5 × 28 = 14`.

For a compound Poisson total `L`, `Var(L) = λ E[X²] = 0.5 × (0.8 × 100 + 0.2 × 10,000) = 1,040`, so the empirical mean of 20,000 simulated years has standard error `√(1,040 / 20,000) = 0.228`.

`P(L > 0) = P(N ≥ 1) = 1 − e^−0.5 = 0.393469`.

### Annual aggregate distribution (AEP)

Poisson thinning splits the incidents into two independent streams: `A ~ Poisson(0.4)` incidents of 10 and `B ~ Poisson(0.1)` incidents of 100. So `L = 10A + 100B` and

`P(L = 10a + 100b) = e^−0.5 × 0.4^a / a! × 0.1^b / b!`.

The lattice probabilities `g(k)` of a loss of `10k` are:

| `k` | `g(k)` | `F(10k) = P(L ≤ 10k)` |
| ---: | ---: | ---: |
| 0 | 0.6065307 | 0.6065307 |
| 1 | 0.2426123 | 0.8491429 |
| 2 | 0.0485225 | 0.8976654 |
| 3 | 0.0064697 | 0.9041350 |
| 9 | ≈ 0 | 0.9048374 |
| 10 | 0.0606531 | 0.9654905 |
| 11 | 0.0242612 | 0.9897517 |
| 12 | 0.0048522 | 0.9946040 |

The export's points for `L` fall on this lattice; among them, `P(L > 10) = 1 − 1.4 e^−0.5 = 0.150857`, `P(L > 20) = 0.102335` and `P(L > 100) = 0.034510`.

### VaR and TVaR

VaR is the smallest `x` with `F(x) ≥ α`, and TVaR the mean of the worst `1 − α` probability, including the share of the VaR atom needed to fill it:

`TVaR_α = (E[L; L > VaR] + (F(VaR) − α) × VaR) / (1 − α)`.

`E[L; L ≤ 100] = 10 E[A] P(B = 0) + 100 P(B = 1, A = 0) = 4 e^−0.1 + 10 e^−0.5 = 9.684656`, so `E[L; L > 100] = 14 − 9.684656 = 4.315344`. Then:

| Level | VaR | Why | TVaR |
| ---: | ---: | --- | ---: |
| 0.95 | 100 | `F(90) < 0.95 ≤ F(100)` | `(4.315344 + 0.01549048 × 100) / 0.05 = 117.2878` |
| 0.98 | 110 | `F(100) < 0.98 ≤ F(110)` | `(4.315344 − 110 × 0.02426123 + 0.00975171 × 110) / 0.02 = 135.9648` |

Because the gross EAL is exact, the engine reports a control-variate TVaR (`tvar_estimator`): the sample tail mean plus `b (14 − m) / (1 − α)`, where `m` is the mean of the simulated years and `b = Cov((L − VaR)₊, L) / Var(L)` is estimated from them. The exact coefficients are 0.130 at 0.95 and 0.093 at 0.98. The payout and occurrence-maximum views have no exact mean, so their TVaR is the sample tail mean (coefficient 0).

### Annual largest occurrence (OEP)

The annual maximum `M` is at most `x` exactly when no incident exceeds `x`: `P(M ≤ x) = exp(−λ P(X > x))`. So `P(M = 0) = e^−0.5 = 0.606531`, `P(M = 10) = e^−0.1 − e^−0.5 = 0.298307` and `P(M = 100) = 1 − e^−0.1 = 0.095163`.

`E[M] = 10 × 0.298307 + 100 × 0.095163 = 12.4993`. The mean of an annual maximum is not the EAL. `P(M > 10) = P(M > 50) = 0.095163` and `P(M > 100) = 0`. VaR and TVaR are 100 at both levels.

### Payout after the retention and limit

Annual payout is `P = 50B` with `B ~ Poisson(0.1)`:

- payout EAL `= 50 × 0.1 = 5`, with variance `2,500 × 0.1 = 250`
- `P(P > 0) = 1 − e^−0.1 = 0.095163`, `P(P > 50) = 1 − 1.1 e^−0.1 = 0.004679`, `P(P > 100) = 0.000155`
- `F(0) = e^−0.1 = 0.904837` and `F(50) = 1.1 e^−0.1 = 0.995321`, so VaR is 50 at both 0.95 and 0.98
- `E[P; P > 50] = 5 − 50 × 0.1 e^−0.1 = 0.475813`, so `TVaR_0.95 = (0.475813 + 0.04532116 × 50) / 0.05 = 54.8374` and `TVaR_0.98 = (0.475813 + 0.01532116 × 50) / 0.02 = 62.0935`

The largest occurrence payout is 50 with probability `1 − e^−0.1` and 0 otherwise, so its mean is `4.7581`.

## Tolerances and statuses

The analytic gross EAL must equal 14 to floating-point precision, an exceedance probability of exactly 0 must be reproduced exactly, and empirical VaR must equal the hand atom. Each other empirical quantity must lie within five standard errors of its hand value at 20,000 years. For TVaR at a known VaR the standard error is `sd((L − VaR)₊ − b (L − E[L])) / ((1 − α) √n)`, with `b` the exact coefficient (zero without an exact mean). For the gross coefficient it is `√(E[e² (L − E[L])²] / n) / Var(L)`, with `e = (L − VaR)₊ − E[(L − VaR)₊] − b (L − E[L])` the regression residual: about 0.008 at both levels.

The tail convergence gate (`NUM-TAIL-GATE`) works on 16 batches of 1,250 years. A TVaR's expected half-width is `t₁₅,₀.₉₇₅` times its standard error: ±1.7% of the gross TVaR at 0.95 and ±3.4% at 0.98, and ±2.0% and ±4.4% of the payout TVaR, within the gate's 10%. At both levels every statistic is `available`.

A gross VaR or TVaR also needs the simulated years to reconcile with the analytic EAL (`eal_reconciliation`): `z`, their mean less 14 over its batch-means standard error, must lie within ±5. Under the exact law `z` is about t-distributed with 15 degrees of freedom, so it lies beyond ±5 with probability 1.6 × 10⁻⁴.
