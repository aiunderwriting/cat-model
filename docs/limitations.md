# Limitations

The most material limitations come first. IDs such as `POOL-SPREAD` are rows of the [assumption register](../assumptions.md), and the [glossary](glossary.md) defines the terms.

## 1. The risk judgments come from LLMs and cannot be backtested in general

Respondent LLMs supply every frequency, severity and lab-loss judgment of a study. The model's predictions cannot in general be backtested: most scenarios have no matching incident data, and the study describes a hypothetical lab. Where a narrower or representative prediction can be checked against outcomes, it is. The [respondent calibration](../methodology.md#respondent-spread-calibration) on known-answer count questions is the first such check. It measures how wide and how biased the respondents' answers are. Whether that carries over to the risk questions is untested.

- Results read every answer at face value, with no correction for overconfidence or for a bias the respondents share (`POOL-SPREAD`, `POOL-COMMON-BIAS`). A shared bias multiplies every count, and so the EAL, by the same factor. Correcting overconfidence by a spread factor of 1.5 to 2 would raise a count law's mean 1.7 to 3.4 times, for p95/p5 ratios of 20 to 400 ([methodology](../methodology.md#pooling-and-parameter-uncertainty)).
- Respondents answer without search or tools, from the question text and the evidence excerpts it quotes. No excerpt counts incidents of at least USD 100,000 involving one lab's models, so scaling a reference class to the lab is the respondent's judgment. Most questions also quote a [threat model](threat-models.md), whose evidence on developers' safeguards comes mostly from one developer's publications. One analysis shown to every respondent can also anchor the panel on its framing.
- Repeated samples of one LLM are not independent experts, and a study of a named lab can ask an LLM about its own developer's systems. A model that declines a scenario, such as a CBRN one, is left out of it. The scenario then rests on the models that answered, possibly one (`POOL-UNANSWERED-SCENARIOS`).
- The frequency cross-checks compare two judgments, not a judgment with data: direct against chain counts, stated against implied EAL, and a category against its parts (`FREQ-CHAIN-POOL`, `FREQ-EAL-CROSSCHECK`, `HIER-CATEGORY-POOL`). Both come from the same respondent, so agreement can reflect a shared error. The pools that combine them are fixed rules, not calibrated ones.

## 2. The study describes a hypothetical lab

The hypothetical-lab study describes a stipulated lab, not an actual company (`POP-HYPOTHETICAL-LAB`). Its deployments, their scales, its safeguards and its place at the public frontier are all stipulations (`POP-SAFEGUARD-LEVEL`). Its deployment scales grow with a trajectory's capability progress, up to three times their growth on the median path; that link is a judgment. The example setups, such as [golden Poisson](../config/setups/golden-poisson/README.md), use stipulated numbers and test the software, not risk.

## 3. Tail shapes, caps and spread factors are judgments

Five stated quantiles say little about a law beyond its p95, so judgment sets the tails. Tail statistics (VaR, TVaR and the occurrence views) are therefore reported as illustrative. The EAL depends on the same judgments: on the count laws' unstated tails beyond p5 and p95, and on the governed severity tail shape. The EAL parameter band shows only the count laws' and respondents' uncertainty. It is not a range of annual losses or a confidence interval (`NUM-EAL-PARAMETER-BAND`).

- The severity tail is a GPD of shape 0.5, 0.7 or 0.8 by tail class, stressed by ±0.2 (`SEV-GPD-SHAPE`).
- A respondent's expected annual count of incidents of at least USD 1 billion sets how many reach that level; its odds are stressed by ×⅓ and ×3 (`SEV-LARGE-LOSS-ANCHOR`). Beyond that level the governed shape decides alone. With a stated p99 far below it, about 1 in 10,000, 720 or 316 such incidents reaches USD 100 billion at shapes 0.5, 0.7 and 0.8.
- One catastrophe-class incident is capped at USD 100 trillion, stressed at USD 10 and 500 trillion and uncapped (`SEV-CAP`). A frontier-lab value of USD 1 trillion caps each incident's lab loss and each scenario's annual business loss (`SEV-LAB-CAP`). It is stressed at USD 300 billion, USD 3 trillion and removal, and a year's lab loss can exceed it.
- Count-law tails beyond the stated p5 and p95 are extrapolated, with width factors of 0.75 and 1.5 as stresses (`POOL-PCHIP`). The spread alternatives are `k` of 0.75 and 1.5, and the bias alternatives ×0.5 and ×2 (item 1). Counts are Poisson, with a negative-binomial alternative at a variance-to-mean ratio of 2 (`FREQ-POISSON`).

## 4. The main pipeline challenges only some of its assumptions

By default a study's setup compares its reference with the forecast's alternatives, each respondent model left out in turn, the tail-shape step and the uncapped severity stress. Results report `status: partial` and name every comparison that did not run. A fixed-anchor setup runs only its reference arm.

- No arm challenges the capability response itself (`OBS-CAUSAL`), the independence of the labs' paces, severity location or the dependence assumptions of item 5.
- The count laws' spread, tail-width and bias alternatives, the lab-value and cap-endpoint stresses and the negative-binomial count are opt-in.
- The exposure-scale and safeguard-level comparisons need separately collected variant studies, so each effect includes the difference between two collections. A safeguard effect is the respondents' judgment of stated controls, which no incident data validates. Each level changes scope, permissions and the lab's own monitoring together. No level changes who reviews an output or action, because the taxonomy assigns an incident to a scenario by that review and a level that changed it would move incidents between scenarios: safeguards that work by requiring human approval are in no level. The stipulated safeguards describe what leading labs publish about their practice and cite no source. The stronger level is the strictest safeguards that the roster labs' published safety frameworks describe, most of which those documents attach to capability thresholds that have not been reached: it is a posture that no lab is known to run. Neither level changes who uses the lab's models or how much, so holding a model back, the strongest measure those frameworks name, is in no level (`POP-SAFEGUARD-LEVEL`).
- The EAL and every forecast comparison are a quadrature over the forecast on a study's trajectory nodes, seven by default. Losses that change sharply between nodes are integrated less accurately ([methodology](../methodology.md#trajectory-quadrature)).

## 5. Scenarios are conditionally independent

Scenarios depend on each other only through the capability trajectory and the respondent they share (`DEP-CONDITIONAL-INDEPENDENCE`). Given those, counts and severities in different scenarios are independent: no shock is common to several. Tail statistics can therefore understate correlated catastrophes. The EAL does not depend on this assumption.

- A lab's annual business loss is drawn independently of the same year's societal losses, and does not grow with the year's count beyond one incident (`ACC-LAB-BEARER`, `ACC-LAB-ANNUAL-BUSINESS-LOSS`).
- The lab's compensation share is the same for every incident of a scenario.
- The engine's dependence stresses, a shared gamma intensity and a Gaussian count copula, run only in authored setups, on counts and with placeholder parameters (`DEP-RESIDUAL-FACTOR`).

## 6. The capability forecast rests on few releases

The forecaster fits one trend rate, pooled across five labs, to their frontier-raising releases in Epoch's ECI table (`OBS-PUBLIC-JOINT`, `OBS-PUBLIC-TIMING`; [capability forecasting](capability-frontiers.md)). The committed table holds 44 such records in its 24-month window. Epoch refits the index, so each score is today's value, not the one published at release.

- The pace law, a log-logistic shape of 5 truncated at 4 times the trend rate, is a judgment. The shape was chosen on the backtest cases that score it. The cap is about twice the fastest lab-year on record, and progress beyond it enters no result.
- The three fastest of the hypothetical-lab study's 15 trajectory nodes put the frontier 65 to 78 ECI points above its as-of level in 14 months. The year before brought 18. They carry under 1% of the forecast's weight, yet a loss growing like the fourth power of progress takes 20% of its EAL from them.
- How capability changes risk is the respondents' conditional judgment, not a causal estimate (`OBS-CAUSAL`): capability, adoption and controls can move together. Each question says what an ECI point means by quoting one correspondence [Epoch](https://epoch.ai/eci) reported, about 5 points per doubling of METR's time horizon (`OBS-ECI-SUFFICIENCY`).
- Between the slowest and the fastest trajectory a direct question differs in about 0.3% of its characters ([example](example-questions/fastest-trajectory.diff)). What in a trajectory moves the answers is measured only by the optional [conditioning pilot](../runbook.md#conditioning-pilot).

## 7. The model quantifies selected populations only

A result covers only the scenarios and deployments its setup selects. Of the taxonomy leaves without a scenario, <!-- scenario-count:leaves.unestimated -->4<!-- /scenario-count --> are *unestimated* and <!-- scenario-count:leaves.unsupported -->1<!-- /scenario-count --> is *unsupported* ([taxonomy](taxonomy.md#which-categories-have-estimates)). Unquantified populations stay out of the totals, with no allowance added for them, and are reported as unquantified, not as zero.

- Every leaf admits only incidents whose gross loss across all parties is at least USD 100,000 (`POP-ADMISSION`). Smaller incidents are an unestimated leftover.
- Each incident is counted whole, in the leaf of the process that caused most of its loss (`POP-WHOLE-INCIDENT`). No leaf subtracts the loss that would have happened without the model, so a result is not an estimate of marginal risk. An attack that used a model without material assistance is not counted.
- Non-monetary harm, diffuse macroeconomic effects and AI's benefits are outside the loss measure. No value per fatality is stipulated: respondents value injury and death inside each loss judgment, as compensation, not a value of a statistical life.
- In <!-- scenario-count:normalisation_bases -->16<!-- /scenario-count --> scenarios an exposure base only normalises the lab's size; the [gap register](../registries/coverage_gaps.yaml) lists these and other coverage gaps. A category's remainder, the part its scenarios miss, is elicited and reported but never simulated or added to a total (`HIER-REMAINDER`).

## 8. Event definitions are authored judgments

Each scenario's definition, counted unit, year rule, event chain and exposure bases are authored structural judgments without matched incident data (`POP-TERMINAL-INCIDENT`, `FREQ-EVENT-CHAIN`, `FREQ-CHAIN-EXPOSURE`). Whether a model's help to an attacker was material is a judgment of what the attacker would have had without any AI model, and no incident record settles it.

## 9. Numerical precision at default budgets is modest

Import starts every joint state at 1,024 simulated years. The EAL of most views is analytic, with no sampling error; payout, occurrence-maximum and risk-unit views report the simulated mean. VaR, TVaR and P(L > 0) come from the simulated years.

- The tail convergence gate doubles an arm's years, up to a cap, until each statistic's 95% half-width is within 10% of its value (`NUM-TAIL-GATE`). A statistic still unconverged is withheld with its half-width.
- The half-widths cover Monte Carlo error only. They exclude parameter, elicitation, forecast and model uncertainty, and integration error over the trajectory nodes ([methodology](../methodology.md#numerical-execution-and-outputs)).
- A 1-in-N loss is exceeded with probability 1/N in one predictive year, not once in N years under a persistent but unknown rate.
- Small incidents are summed per year by a normal law with their exact mean and variance; each result bounds the resulting TVaR difference (`NUM-SMALL-EVENTS`).

## 10. Results are research estimates, not actuarial work product

The model is experimental research software, released under the [Apache License 2.0](../LICENSE). Its results are research estimates whose frequency basis is LLM judgment, not claims experience, so they do not support insurance pricing, reserving or capital decisions.

## 11. Captured evidence can carry third-party rights

Source bundles, collections and imported setups retain selected evidence, captured prompts and LLM responses without redaction. Nothing in the workflow screens responses for hazardous detail. A results export leaves the captured prompts and responses out unless `--include-answers` asks for them ([portable results](setup-run-results.md#portable-results-and-conditional-distributions)), and publishing them needs rights clearance and a check for hazardous detail. An export does not grant permission to redistribute the evidence it holds, and provider terms apply to published LLM responses.
