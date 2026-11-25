# Capability forecasting and context-bound risk inputs

The [methodology](../methodology.md#observable-capability-conditioning) defines the public-frontier forecast and its [trajectory quadrature](../methodology.md#trajectory-quadrature). This page covers what an operator needs: how the forecast reads its inputs, its fields and defaults, and how to read its outputs.

## Joint public-frontier forecast

`capability-forecast public-study` fits the forecast and writes paired public and not-yet-public risk instruments for every trajectory node, so each contextual collection is conditioned on one whole trajectory. A `PublicFrontierForecast` (contract `src/catmodel/contracts/public_frontier.py`, template `data/capability/public-frontier-forecast-template.json`) lists its labs as `coordinates` on one general-ECI scale. `public_frontier` is reserved for their derived maximum, so the declared roster determines what that maximum covers.

**Scale reference.** Every node's condition carries the forecast's `scale_anchors`, each lab's as-of score and the labs' highest score one year before as-of. Where the forecast records a `scale_correspondence`, what Epoch reports a gain in points to have corresponded to in another measure, each question quotes it with the doublings that its own path's rise would be.

**Releases.** A forecast that states no releases takes them from Epoch AI's ECI model-score table, captured in the study's source bundle as `evidence/epoch_eci_scores.csv`:

- It selects the roster labs' public models (API, hosted or open-weights access) released by as-of, mapped by Epoch's organization label. Models with limited, unreleased or unknown access are excluded.
- Each release is dated at Epoch's release date (`availability_basis: release_date_proxy`), which does not show when a model became accessible. Checked against the developers' own release pages on 2026-11-25, 38 of the 44 trend-window records match to the day, five differ by 1 to 21 days and one has no dated developer page.
- Its benchmark error (`benchmark_error_sd`) is half its 90% bootstrap interval width over 1.645. Epoch's two scale anchors, Claude 3.5 Sonnet at 130 and GPT-5 at 150, have no interval and get zero benchmark error; any other row without an interval is refused.
- Scores are on Epoch's current refit, not the values published at release ([Epoch](https://epoch.ai/data-insights/gpt-4-longest-eci-lead)).

**Records.** A lab's records are its frontier-raising releases, those that strictly exceed every earlier release of the same lab. Equal or lower scores, including re-listed scores and smaller models, are not frontier observations. The last record is the lab's observed as-of frontier. Records available on or after `trend_window_start` enter one pooled likelihood, once each; earlier records set only the record threshold and the anchor. A lab with one window record carries no rate information, and carrying a frontier through a plateau adds no observation.

**Rate.** `trend-posterior.json` gives the rate's Gaussian kernel location and variance before the rate is conditioned positive, not the moments of the truncated distribution. The default rate prior is diffuse, `N(0, 100²)` before conditioning, so the records set the rate. When `trend_discrepancy_sd` (`τ`) is omitted, it is the pooled restricted-maximum-likelihood estimate from the window records, reported with its 90% profile-likelihood interval; an authored value overrides it as a sensitivity. A lower interval end of zero means that benchmark error alone fits the records. The leave-one-out record residuals in `forecast-comparison.json` have a root mean square near one when `τ` fits the records, and well above one when it is too small.

**Limits of the fit.** The record likelihood ignores the waiting time since a lab's last record and any serial dependence between record jumps. Records are upward-selected releases, which biases the pooled rate upward ([limitation 6](limitations.md#6-the-capability-forecast-rests-on-few-releases)). One shared trend rate, a simplicity choice recorded in `pooling_rationale`, fits labs with different rates to their average; differences between labs' futures come from their independent paces. Anchoring assumes that no progress has accumulated since a lab's last record.

**Paths.** Each lab follows `F_l(t) = F0_l + B · M_l · (t − as_of)` from its observed as-of frontier `F0_l`, with the shared rate `B` and its own pace multiplier `M_l`, drawn from a log-logistic law with median one and shape `pace_shape`, truncated at `pace_cap`. The multiplier applies from as-of through the period end. The public frontier is the pointwise maximum of the lab paths within each draw, evaluated at the policy period's start, each UTC month start inside it and its end, with crossing points kept as knots.

**Defaults.**

- `pace_shape` is 5, a round judgment informed by `scripts/backtest_public_frontier.py`, a rolling-origin pinball backtest on the captured Epoch releases, which favored shape 8 overall and 5 at the 3-month horizon (`tests/test_public_frontier_epoch.py`). The shape was chosen on the cases that score it, so its coverage is effectively in-sample.
- `pace_cap` is 4, a judgment that no lab progresses faster than four times the pooled trend rate through the forecast horizon. On the captured table, the fastest twelve months of any roster lab's frontier gained 29.6 ECI points, 1.8 times the pooled 16.2 a year, and its fastest six months, annualized, 3.1 times that rate, so the cap is about twice the fastest lab-year on record (`tests/test_public_frontier_epoch.py`). At shape 5 and cap 4, `P(M > 2) ≈ 0.03` for each lab, and the cap removes 0.1% of the log-logistic law.

**Target.** The risk lab follows a target path: `public_frontier` for the hypothetical frontier lab, or a named lab's. Every question shows all the joint lab paths. Neither the target path nor the joint forecast is a forecast of internal capability.

**Domain benchmarks.** Domain benchmark results, such as cyber or biology evaluations, are context and not forecast axes. Questions about cyber or biological risk condition on general ECI and the lab description, and respondents carry any uncertainty about domain capability in their judgments. Answers conditioned on another coordinate cannot be reused by relabeling it; collect fresh general-ECI instruments.

### Trajectory quadrature

`public-study` integrates the forecast over its frontier progress by a fixed rule on a few collected trajectories ([methodology](../methodology.md#trajectory-quadrature)). The base law and the forecast's fixed alternatives (pace shapes 3 and 8, a lower pace cap of 2, the trend discrepancy at its 90% interval ends, and no progress) are weighted on the same nodes. A pace cap above the forecast's cannot be represented, so author another cap, above 1, before planning; `public-frontier.json` records a higher one as unavailable.

`setup import --study` recomputes the rule frozen in `public-frontier.json` and rejects a study whose rule differs beyond floating-point roundoff. A NumPy or SciPy algorithm change can therefore break replay, so keep the generating software environment with the captured evidence. The reported loss moments are those of the node mixture: they do not establish that expected loss or its variance is finite under the continuous forecast.

Source status is fixed when the incident originates, so a public model used internally stays public. The annual laws are already conditional on the public trajectory and get no second capability adjustment, and the setup mixes complete loss distributions, never an expected capability or averaged conditional quantiles. Shared public paths are the only dependence this conditioning carries ([comparison arms](setup-run-results.md#context-bound-comparisons)).

## Answer format

Every numerical answer carries a short reasoning record (`Reasoning` in `src/catmodel/contracts/elicitation.py`). Answers are parsed as strict JSON; a failed check returns a stable error code and field location for the retry and never repairs a number.
