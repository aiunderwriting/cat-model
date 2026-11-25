# Glossary

Code identifiers appear in `code` where a term has a direct counterpart.

## Words with more than one meaning

| Word | Sense | Meaning |
| --- | --- | --- |
| Model | the model | This repository's probabilistic loss model. |
| | AI model, assessed model | An AI system whose risk is assessed; an `Entity` of kind `model`. |
| | respondent LLM, respondent | A large language model that answers the questions. |
| | law, forecaster | A statistical model inside the calculation: a probability distribution, or the capability forecaster. |
| Node | trajectory node | A quadrature node of the public-frontier forecast. Each interior node carries one representative path of every lab's ECI; the no-progress node holds every lab at its observed frontier. |
| | taxonomy node | A split or a leaf of the [loss taxonomy](taxonomy.md). |
| | scenario node (`ContextualRiskNode`) | The population, counted unit and event chain that an instrument asks about. |
| Origin | causal origin | The taxonomy's first split, on what started an incident: deliberate misuse, AI failure, provider conduct or other origin. |
| | physical origin (`origin_id`) | One incident law inside a setup, which owns the losses of its incidents. |
| | evidence origin (`Evidence.origin`) | How an input was obtained: `observation`, `fit`, `elicitation` or `stipulation`. Every origin enters the same calculation. |
| Kernel, a conditional probability law | mark kernel (`MarkKernel`) | The law of an incident's mark, given that the incident occurs. |
| | conditional kernel (`ConditionalKernel`) | A study's annual loss law at one capability value, such as one integer ECI score. |
| | payment kernel (`NativePaymentKernel`) | A captured law for the share of a loss that a developer pays. |
| Capture | run capture (`RunCapture`) | The immutable record that `catmodel run` writes: aligned annual samples of every requested view, with their state weights. |
| | captured evidence | Frozen inputs from outside the repository: the files of a source bundle, or the prompts, raw responses and parsed answers of an LLM collection. |

## Terms

**Admission.** The check that an input or law fits its declared population, units and conditioning before it enters a calculation. It checks structure and identity, not truth. Distinct from *admission threshold*.

**Admission threshold.** The one loss test every leaf of the taxonomy applies. An incident is admitted when its gross loss is at least USD 100,000. Gross loss is summed over all affected parties and loss kinds, before any recovery, in policy-start USD ([methodology](../methodology.md#loss-taxonomy)). Smaller incidents are an unestimated leftover. The [severity tiers](taxonomy.md#severity-tiers) list it beside the two large-loss levels.

**AEP (aggregate exceedance probability).** See *exceedance probability*.

**Anchor.** A fixed capability score at which questions are asked. In conditional what-if analysis a respondent answers the same question at several ECI anchors, and laws between anchors interpolate the answers.

**Answer table.** A directory of provider responses, each keyed by its exact prompt, the model and its settings, and the sample index. A collection given one (`--answers`) reads it before each request and adds every response to it, so an unchanged question is not asked twice ([runbook](../runbook.md#answer-table)).

**Arm.** One executed version of a setup, with its own joint states, simulation count and numerical controls. The *reference arm* (`reference`) is the base case. An *alternative arm* changes one declared input, as a `scientific_alternative` (a sensitivity) or a `numerical_refinement` (a larger budget for the same laws).

**Campaign, member and cluster.** One initiating incident (a *campaign* or *episode*) can harm several *members*, and the *cluster* law gives their number. Every scenario counts whole incidents of one member each; authored engine setups can declare clusters.

**Category pool.** The fixed rule that pools a taxonomy parent's own answer for its whole category with what is beneath it, its modeled children plus its remainder (`HIER-CATEGORY-POOL`). The pooled value is the weighted geometric mean of the two, and its factor scales the expected count of every leaf beneath the parent ([methodology](../methodology.md#event-populations-and-frequency)).

**Chain pool.** A leaf's count law (`FREQ-CHAIN-POOL`). At each of p5, p25, p50, p75 and p95 it is the equal-weight average, in log space, of the cross-checked direct count and the *event chain*'s count (*Vincentization*). A wholly zero direct answer uses the chain alone; an unavailable or wholly zero chain uses the direct count alone.

**Comonotone.** Moving together perfectly: comonotone quantities are all increasing functions of one shared rank. It is the strongest positive dependence, and the model uses it as the most dependent case of the *EAL parameter band* and of the results' dependence bracket ([methodology](../methodology.md#pooling-and-parameter-uncertainty)).

**Comparison family.** A named group of alternative arms that challenge one assumption, such as `tail_shape`, `public_forecast` or `model_pooling_leave_one_out`. The families are listed in [`sensitivity_catalog.py`](../src/catmodel/setup/sensitivity_catalog.py) and tied to rows of the [assumption register](../assumptions.md).

**Component.** One distinct loss item within an incident, such as a victim's resource loss or a regulator's fine. Each component has a bearer and an economic nature: `resource_cost`, `injury_valuation`, `transfer` or `appropriated_transfer`.

**Context-bound path.** Questions about one explicitly named lab, AI model and deployment. Asked along trajectories from the public-frontier forecast, they form the main pipeline, which gives an absolute EAL. Asked at fixed ECI anchors, they give conditional what-if analysis ([workflows](workflows.md)).

**Coverage and result limitations.** A result's `coverage` table lists the views, comparison families and setup populations it includes or leaves out, each exclusion with a reason. Its `limitations` list gives the setup's limitations, the run's own gaps and the model-wide items of the [limitations list](limitations.md).

**EAL (expected annual loss).** The mean of a view's annual loss distribution. Results report the analytic EAL of the native laws where one exists, beside the empirical mean of the simulated years.

**EAL cross-check (leaf).** The check of a leaf's *direct EAL* against its *implied EAL* (`FREQ-EAL-CROSSCHECK`). The direct EAL is the respondent's p5–p95 of the node's expected annual societal harm. The implied EAL is the direct count law's mean times the compiled law's welfare mean per incident.

**EAL parameter band.** The `eal_parameter_band` exhibit (`NUM-EAL-PARAMETER-BAND`): p5, p50 and p95 of a view's EAL with each count law held at one rank for every year instead of integrated over it. Its mean is the EAL. [Limitation 3](limitations.md#3-tail-shapes-caps-and-spread-factors-are-judgments) says what it does not show.

**ECI (Epoch Capabilities Index).** [Epoch AI's](https://epoch.ai/eci) composite index of AI capability, built from many benchmark results on one scale. The model uses general ECI as its observable capability coordinate.

**Elicitation.** Asking respondent LLMs structured questions and recording their answers. An *instrument* is the frozen specification its questions are rendered from, with the respondent roster. A *collection* is one run of an instrument, with every prompt, raw attempt and parsed answer retained.

**Entity.** A lab, AI model, operator, deployment or population named in a setup's scope, marked `actual` or `hypothetical`. Both kinds use the same mathematics.

**Event chain.** A scenario's authored bottom-up route to its annual count (`event_chain`). It runs from an annual opportunity, counted against an *exposure base* for each deployment, through one to four conditional stages to the scenario's qualifying incident. Each opportunity yields at most one incident. The expected count is therefore the base count times the elicited *opportunity rate* (opportunities per base unit, `FREQ-CHAIN-EXPOSURE`) times the product of the stage probabilities ([methodology](../methodology.md#event-populations-and-frequency)).

**Exceedance probability (EP), AEP and OEP.** An EP point is `P(annual loss > threshold)`. The aggregate curve (AEP) describes a year's total loss. The occurrence curve (OEP, the `oep_*` views) describes the largest single occurrence in a year ([incident accounting](incident-accounting.md#loss-and-view-definitions)).

**Exhibit.** A report-only analysis of a run, held in the diagnostics file beside `results.json`: the sensitivity summary and its parts, such as `large_loss_anchor` and `eal_parameter_band`, and `capability_scenarios`, `scenario_groups` and `tail_gate`. An exhibit changes no headline statistic ([setup, run and results](setup-run-results.md)).

**Exposure base.** The natural unit a chain's opportunities are counted against, for one deployment, with a stipulated count shown to the respondent. The count is either a stipulated deployment scale times an authored factor, which moves with the trajectory, or a number the scenario fixes. Its `basis` records whether the count rests on judgment, data or only a normalisation of the lab's size (`FREQ-CHAIN-EXPOSURE`).

**Governed tail.** The tail shape (*GPD* shape ξ) and cap of an elicited severity law. They come from the scenario's tail class in [`severity_governance_policies.yaml`](../scenarios/severity_governance_policies.yaml), not from respondents. Respondents give quantiles, and the governed shape extends the law beyond p95 through their p99 and, where the *large-loss anchor* applies, beyond the anchor.

**GPD (generalized Pareto distribution).** The family the model uses for a severity law's tail above p95, with shape ξ and scale σ. A larger shape is a heavier tail: uncapped, the variance is infinite at ξ of 0.5 or more and the mean at 1 or more ([methodology](../methodology.md#conditional-severity)).

**Joint state.** One weighted combination of a capability path and one respondent's judgments (`JointState`). The annual loss distribution is the weighted mixture of the distributions in every joint state.

**Lab economic loss.** The `lab_loss::<lab_id>` view: a named lab's incident-linked costs and liabilities before recoveries. In a contextual setup an incident's lab loss is its *provider-borne components* (those whose authored bearer is `provider`) plus the *lab compensation share* of its other components (`lab_compensation`, `ACC-LAB-BEARER`). The sum is capped at the *frontier-lab value*, a stipulated USD 1 trillion (`SEV-LAB-CAP`). Each scenario adds its *lab business loss* (`lab_annual_business_loss`, `ACC-LAB-ANNUAL-BUSINESS-LOSS`), capped separately, in each year with at least one incident ([incident accounting](incident-accounting.md#lab-economic-loss)).

**Large-loss anchor.** The expected annual number of a leaf's admitted incidents with gross loss of at least USD 1 billion (`SEV-LARGE-LOSS-ANCHOR`). Its ratio to the count, the *large-loss share*, is the probability that an admitted incident reaches USD 1 billion. Where that level lies above the law's p99, the tail above p99 reproduces the share. The same count at USD 100 billion enters only the *two-anchor tail*, a comparison law ([methodology](../methodology.md#conditional-severity)).

**Mark.** Everything attached to a simulated incident beyond the fact that it occurred: its cluster size, member outcomes, loss components and their ownership. The model is a *marked* Poisson process.

**Material assistance.** The tie between a provider's model and an attack, in the scenarios where a model helps an attacker (convention 7 of the [loss taxonomy](../methodology.md#loss-taxonomy)). Help is material when it meets two conditions. It gave the attacker capability, scale or speed that the attacker would not have had without any AI model. And the loss, or more than half of it, would not have occurred without that gain. The comparison is with what the same attacker would have had with no generative AI model of any developer, including what it would have bought or hired. Help that another AI model could have given is therefore still material. All eleven such scenarios state the test, and each opens its event chain with an `uplift` stage. A campaign or program that only uses the model's help is an *opportunity* there, not an incident.

**Matched control.** The arm a sensitivity is compared with instead of the reference (`matched_control_id`): identical to the sensitivity in every law except the one declared change. A *variant study* is separately collected, so its control is the captures of the study it varies, on the same trajectory nodes and weights.

**Native law.** A law kept in its own exact form (atoms, qualifying floors, caps and tails), sampled and integrated directly and never discretized onto a grid. *Native severity* is the five-quantile body joined to an optional GPD tail from p95; *native analytic EAL* is the exact mean of the native laws.

**Origin-year ultimate.** The model's only temporal basis. The study year's losses are the eventual losses of the incidents that originate in it, however late they are claimed, imposed or paid. Each scenario's year rule dates its incidents.

**PCHIP.** A piecewise cubic Hermite interpolating polynomial: a smooth curve through given points that stays monotone between them (Fritsch and Carlson, 1980). The model uses it to draw a quantile function through a respondent's stated quantiles.

**PML and return period.** A return period of T years names an annual exceedance probability of 1/T. The probable maximum loss (PML) at return period T is the annual-loss quantile at 1 − 1/T: a quantile, not a worst possible loss. Results write these levels as 1-in-T.

**Population.** A scenario's declared loss population in one study (`Population`): its definition, causal qualification, initiating, terminal and severity units, threshold, monetary basis, currency and accounting standing. *Qualification* is the causal criterion and threshold an incident must meet to count. *Standing* is whose loss is measured and on what accounting basis.

**Profile (model profile).** A named respondent roster, each model with its provider, reasoning effort and output ceiling, and the settings of its collections. Collection commands read it from a profiles file: the repository's example, [`example_model_profiles.yaml`](../data/example_model_profiles.yaml), or a study's own ([runbook](../runbook.md#model-profiles)).

**Quadrature.** Numerical integration as a weighted sum over fixed nodes. *Trajectory quadrature* integrates a study over the forecast's frontier progress on a fixed set of trajectory nodes, each with a weight, instead of sampling paths ([methodology](../methodology.md#trajectory-quadrature)).

**Rank.** A uniform number in (0, 1) that selects a value from a respondent's stated uncertainty: a parameter is read off that respondent's quantile function at the rank. Each simulated year of a joint state reads one rank per scenario and source status, the same in every arm of the state and independent of every other state's.

**Reporting domain.** A label on whole leaves by which results add scenario EALs up (`scenario_groups`); it does not affect classification. Under deliberate misuse the codes are `cyber` (cyber and fraud), `cbrn` (CBRN weapons) and `societal` (influence operations). Under AI failure they are `agent` (agent and physical-control failures), `eo` (erroneous outputs and harm to users), `svc` (service failures) and `ai_rd` (loss of developer control). Provider conduct is `regulatory` (regulatory and legal action). The [taxonomy guide](taxonomy.md#reading-the-taxonomy-by-risk-domain) lists each domain's leaves and scenarios.

**Respondent.** One respondent LLM and one sample index answering an instrument. A *profile* fixes the roster and settings.

**Scenario (cell).** One forecast population in the [scenario library](../scenarios/): authored definitions that fix what counts as one incident, its event chain and the severity population. *Scenario* and *cell* are used interchangeably. A scenario forecasts one leaf or a disjoint union of leaves of the taxonomy; today each owns one.

**Scope.** The part of a setup that declares what is modeled: the period and information dates, entities, deployments, capability scales and populations (`scope_id`). Populations outside the scope are unquantified.

**Sensitivity.** An executed alternative arm that changes one assumption, reported as its effect against the matched control or the reference. An alternative reuses the study's captures, recompiled or reweighted under its one change; an input that needs new answers is a *variant study* instead.

**Setup.** The frozen input to a run (`RunSetup`): scope, evidence, assumptions, joint states and laws, alternatives and the execution plan. `catmodel setup build` freezes an authored setup; `catmodel setup import` builds one from captured elicitation.

**Societal harm.** The `welfare` view: resource costs, monetary injury valuations and appropriated assets across everyone affected, counted once, before recoveries. Ordinary transfers such as fines cancel, and perpetrator proceeds earn no credit. `net_resource_cost` also cancels appropriated assets.

**Source bundle.** A checksum-verified, frozen directory of upstream evidence, such as source cards, threat models and ECI captures, built by `catmodel package build-source-bundle`. Its identity is the SHA-256 of its manifest, which instruments bind; the builder's `provenance.yaml` records the build and is outside that identity. No elicitation output ever enters a source bundle.

**Spread factor k.** The multiplier on a respondent count law's normal score: the model reads `Q(Phi(k z))`, so a value above one moves ranks toward the stated tails and keeps the median. The *common frequency bias* b is a pool-wide factor on every count quantile. Every setup reads `k = 1` and `b = 1` (`POOL-SPREAD`, `POOL-COMMON-BIAS`). Opt-in alternatives read `k` of 1.5 and 0.75 and b of ×0.5 and ×2. The respondent spread calibration fits both factors on known-answer count questions, and no setup reads its fit ([limitation 1](limitations.md#1-the-risk-judgments-come-from-llms-and-cannot-be-backtested-in-general)).

**Statistic status.** Each headline statistic of a result view is `available`, `withheld` or `unavailable`. A withheld value exists but the run cannot support it, such as a VaR that the *tail convergence gate* has not converged. An unavailable one has no finite value under the law. Coverage rows also use `not_requested`, and every status other than `available` carries a reason.

**Stipulation.** An input asserted openly, for illustration or as an explicit judgment, without evidence. The golden and stipulated examples use stipulations only.

**Structural zero, unavailable and unquantified.** A *structural zero* is an explicitly authored zero. An *unavailable* result is missing an identified input and carries a reason. An *unquantified* population lies outside the study's scope. The model never reports a missing input as zero.

**Tail convergence gate.** The rule that decides whether a reported VaR or TVaR is precise enough to quote (`NUM-TAIL-GATE`). Its 95% half-width, from 16 batches of simulated years, must be at most 10% of the value. Every batch must hold at least 10 years at or above the VaR. A view with an analytic EAL must also have simulated years that agree with it within five standard errors. An unconverged statistic is `withheld` with its half-width ([methodology](../methodology.md#numerical-execution-and-outputs)).

**Threat model.** The research a scenario's questions can quote: who causes its incidents, the necessary steps to a counted incident, what limits each step and what defenders do. It cites a source for everything it reports and gives no probability or loss estimate ([threat models](threat-models.md)).

**Unanswered scenario.** A scenario that a respondent model does not answer in a study: more than 10% of the model's samples lack its direct answer. Import sets the model's answers to that scenario aside. Each of its samples takes the answers of a *donor*, a sample of an answering model at the same trajectory node. The scenario is then pooled over the models that answered (`POOL-UNANSWERED-SCENARIOS`). Results report neither a respondent-sampling standard error nor an EAL parameter band for an EAL that weights a borrowing sample. A sample that lacks any other scenario is an *excluded sample* (`POOL-EXCLUDED-SAMPLES`; [methodology](../methodology.md#pooling-and-parameter-uncertainty)). A sample whose answers declare the lab's own loss unavailable is not excluded: it is *left out* of the views that count that loss, the lab-loss views and the gross views, where its model's other samples at the node share its weight.

**VaR and TVaR.** The value at risk (VaR) at level α is the α-quantile of annual loss: the smallest loss x with P(L ≤ x) ≥ α. The tail value at risk (TVaR) is the mean annual loss over the worst 1 − α of probability. It is estimated from the worst simulated years, with a control variate where a view has an analytic EAL ([methodology](../methodology.md#numerical-execution-and-outputs)).

**Variant study.** A separately collected study that differs from another study in one stipulated input and is measured against its *matched control*. The *exposure variant studies* (`exposure_scale`) halve or double every path-linked deployment scale. The *safeguard variant studies* (`safeguard_level`) state weaker or stronger safeguards. They are distinct from the forecast's `variants`, which are alternative forecast laws.

**View.** An observable computed from every simulated year, such as `welfare`, `gross`, `net_resource_cost`, `lab_loss::<lab_id>`, `financial` and the `oep_*` occurrence maxima. [Incident accounting](incident-accounting.md#loss-and-view-definitions) defines each.

**Vincentization.** Pooling distributions by averaging their quantiles at each probability level, instead of their probabilities. The *chain pool* averages log quantiles this way.
