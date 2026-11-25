# Reading and presenting research results

A figure, table or paper reports one recorded result of one setup, with the sensitivities its plan requires, and states its study year, currency, monetary basis, scenarios, reference case and loss view. The [glossary](glossary.md) defines the terms used here.

## What the headline measures

The headline reports societal harm (`welfare`) and each named lab's economic loss (`lab_loss::<lab_id>`). [Incident accounting](incident-accounting.md) defines these and the other views, and the [coverage assessment](taxonomy-coverage.md) lists the harms that stay unquantified.

- `welfare` counts losses to theft, fraud and ransom in full, without offsetting the perpetrators' gains. Report `net_resource_cost`, which cancels every transfer on the same incidents, beside it.
- `gross` is not what any payer pays.
- Lab and provider views do not add up to the welfare total, and scenario quantiles do not add up to a portfolio quantile.
- EAL is the expected annual loss under the model's own judgments, with no margin added for risk. A scenario's EAL uses the mean loss per incident of the whole severity law, not its median.

## Reference, sensitivities and controls

Report the reference from the [standard elicitation and pooling procedure](../methodology.md#pooling-and-parameter-uncertainty), without selecting answers or framings for their dollar amounts.

- **EAL parameter band** (`eal_parameter_band`). Label its p5 to p95 range as modeled uncertainty about the EAL; it is not a range of annual losses. A study in which a model does not answer a scenario reports no band for a view that weights the model's samples, and states the reason (`POOL-UNANSWERED-SCENARIOS`). Medians do not add across scenarios, so breakdowns use expected contributions.
- **Sensitivities.** Show the comparisons the frozen plan requires beside the headline, with unavailable arms visible, and absolute effects where the baseline is zero or tiny. Tornado bars do not add and are not confidence intervals.
- **Plausibility checks** ([`NUM-PLAUSIBILITY`](../assumptions.md#num-plausibility)). They give each welfare EAL as a share of world output and, for a contextual setup, the expected annual number of incidents of at least USD 1 billion, which a reader can set against the incidents of that size the public record attributes to any one developer's models. Show a flagged figure with the result.
- **Reporting domains by severity tier.** For a contextual setup, [`summary.md`](setup-run-results.md#readable-summary) has a table with one row for each reporting domain, such as CBRN weapons or cyber and fraud. A row adds up its scenarios' EALs and the expected annual numbers of incidents of at least USD 1 billion and of at least USD 100 billion that the run's laws imply. The count at USD 100 billion rests on the governed tail shape: no law reads the respondents' stated count at that level.

Fixed-ECI, timing, provider and rank comparisons are statistical diagnostics: they do not establish the causal effect of a deployment, safeguard or mitigation. An interactive control selects an executed case or runs the native calculation; it does not rescale a stored loss curve.

## Curves and uncertainty

VaR, TVaR and the occurrence (OEP) statistics are illustrative. They depend on the governed tail shapes, the caps and the conditional independence of scenarios far more than EAL does ([limitation 3](limitations.md#3-tail-shapes-caps-and-spread-factors-are-judgments), [limitation 5](limitations.md#5-scenarios-are-conditionally-independent)). `summary.md` labels their tables illustrative; keep the label wherever they are shown.

- A 1-in-N loss is the annual quantile at `1 − 1/N` of the predictive annual law, which mixes the ECI forecast, respondents and parameter ranks.
- An occurrence curve (OEP) describes the year's largest single loss, which is zero in a year without an incident. State its occurrence scope with it.
- Exports use strict exceedance, `P(loss > threshold)`, which differs from inclusive exceedance at atoms such as caps.
- Carry each statistic's status through plotting and rounding: a `withheld` or `unavailable` value stays so.

## Capability-conditioned results

A public-study result reports `capability_scenarios` at four paths ([`NUM-CAPABILITY-SCENARIOS`](../assumptions.md#num-capability-scenarios)): `current_frontier`, which is the no-progress counterfactual at `node-00`, and the 10th, 50th and 90th percentiles of the base forecast's [frontier progress](../methodology.md#trajectory-quadrature) (`p10`, `p50`, `p90`). Each is one conditioning path with no probability mass of its own.

A percentile path interpolates between its two bracketing nodes with weight `t`, so it rests on the answers at about two trajectory nodes. Its EAL is interpolated linearly, with standard error `sqrt((1−t)² SE_a² + t² SE_b²)`, and is not the mean of the interpolated loss curve. A percentile outside the collected interior nodes is unavailable. In `safeguard_grid`, which gives the same four paths for each level of a study with safeguard-level variants, a percentile outside a level's node range instead takes the nearest node's law and is marked `capability_clamped`.

Show the exhibit's `disclosures` with any chart of it. `results export` writes the exhibit as `capability_scenarios.json` and, without the grid, as `capability_scenarios.csv` and `capability_nodes.csv`; `contracts/results/results.diagnostics.schema.json` gives its fields (`CapabilityScenarios`).

Results are research estimates ([limitation 10](limitations.md#10-results-are-research-estimates-not-actuarial-work-product)); the [intended-purpose statement](../model-intended-purpose.md) says what they are for.
