# Assumption register

This register states each assumption the model makes, the reasoning or evidence behind it, its evidence status and the sensitivity that challenges it. [Methodology](methodology.md) explains the calculations, and [limitations](docs/limitations.md) says how the assumptions qualify a result.

`OK` means adequately supported for the stated use, including verified structural rules that make no calibration claim. `INITIAL_ESTIMATE` means partial evidence or reasoned judgment, `PLACEHOLDER` a temporary value awaiting grounding, and `GAP` missing evidence, rationale, sensitivity or implementation. The labels describe evidence, not implementation or intended use.

## Declared and executed sensitivity scope

The Challenge mechanisms field restates the catalog in [`sensitivity_catalog.py`](src/catmodel/setup/sensitivity_catalog.py): the sensitivity families that challenge the entry (`SENSITIVITY_FAMILIES`), the reason a required challenge is unavailable, or the check that stands in for a study alternative. `tests/test_assumption_register_sensitivity.py` keeps the two in step.

A run executes only the arms its frozen plan declares. Public-study import (`catmodel setup import --study`) freezes every catalog family as a comparison requirement. By default it selects `public_forecast`, `model_pooling_leave_one_out`, `tail_shape`, `uncapped_severity`, `exposure_scale` and `safeguard_level` (`DEFAULT_CONTEXT_BOUND_FAMILIES`). The last two stay unavailable until their separately collected variant studies are imported. Select the lab-loss families, `lab_loss_cap` and `lab_loss_provider_only`, whenever lab loss is reported. Fixed-anchor contextual import (`catmodel setup import --run-directory`) executes only its reference arm.

Every context-bound plan also freezes the challenges that no such setup executes (`CONTEXT_BOUND_UNAVAILABLE_CHALLENGES`) as unavailable requirements keyed by register ID, so comparison coverage cannot read complete ([methodology](methodology.md#sensitivity-validation-and-communication)).

## Population and accounting

| ID | Assumption | Status |
| --- | --- | --- |
| [`POP-UNIVERSE`](#pop-universe) | Five-lab frontier universe | `INITIAL_ESTIMATE` |
| [`POP-HYPOTHETICAL-LAB`](#pop-hypothetical-lab) | Stipulated hypothetical lab | `INITIAL_ESTIMATE` |
| [`POP-SAFEGUARD-LEVEL`](#pop-safeguard-level) | Stipulated safeguards | `INITIAL_ESTIMATE` |
| [`POP-HISTORY-UNITS`](#pop-history-units) | One history behind every unit | `INITIAL_ESTIMATE`, `OK` |
| [`POP-ADMISSION`](#pop-admission) | USD 100,000 admission threshold | `INITIAL_ESTIMATE` |
| [`POP-WHOLE-INCIDENT`](#pop-whole-incident) | Whole incident in one leaf | `INITIAL_ESTIMATE` |
| [`POP-PRIVATE-CLAIMS`](#pop-private-claims) | Private civil matters | `GAP` |
| [`POP-TERMINAL-INCIDENT`](#pop-terminal-incident) | Direct terminal incident counts | `INITIAL_ESTIMATE` |
| [`ACC-ATTRIBUTION`](#acc-attribution) | Ecosystem attribution | `INITIAL_ESTIMATE` |
| [`ACC-WELFARE`](#acc-welfare) | Unique welfare accounting | `INITIAL_ESTIMATE` |
| [`ACC-LAB-BEARER`](#acc-lab-bearer) | Lab-borne components and compensation | `INITIAL_ESTIMATE`, `GAP` |
| [`ACC-LAB-ANNUAL-BUSINESS-LOSS`](#acc-lab-annual-business-loss) | Annual lab business loss | `INITIAL_ESTIMATE` |
| [`ACC-ORIGIN`](#acc-origin) | Shared origin and component identity | `INITIAL_ESTIMATE` |
| [`ACC-COMPOSITION`](#acc-composition) | Composition shares | `INITIAL_ESTIMATE` |
| [`ACC-SCOPE`](#acc-scope) | Excluded value and pathways | `GAP` |

### POP-UNIVERSE

**Assumption.** Five labs define the public-frontier roster: OpenAI, Anthropic, Google, Meta and xAI. Each study names its risk lab and deployment.

**Rationale.** A fixed roster keeps the frontier's definition from changing silently. It is not evidence that other providers contribute no harm.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** A provider add/drop comparison.

**Challenge mechanisms.** Unavailable: A provider add/drop comparison requires a separately captured compatible universe and study.

### POP-HYPOTHETICAL-LAB

**Assumption.** The hypothetical-lab study's risk lab is stipulated in [`context.json`](config/studies/hypothetical-frontier-lab/context.json): a hypothetical frontier lab with a consumer assistant, an enterprise API and internal use of not-yet-public models, whose public capability follows the public frontier. Each deployment scale has an as-of level (October 2026) and a 2027 level on the forecast's median path: weekly consumer users 500 million and 600 million, paying business customers 1 million and 1.6 million, API tokens a day 40 trillion and 100 trillion, staff 6,000 and 8,000. A trajectory node renders each scale, to two significant figures, as as-of + (median-path level − as-of) × g(x). Here x is the node's frontier progress over the base forecast's median progress, both at the middle of the policy period. g(x) = x for x ≤ 1 and 1 + (M − 1) tanh((x − 1)/(M − 1)) above, with M = 3 (`max_growth_multiple`).

**Rationale.** A hypothetical lab lets the study report numbers without attributing an estimate to a company. The scales are stipulations sized like a large frontier lab. Reports give their order of magnitude: 1.2 billion weekly users ([OpenAI DevDay recap, 29 September 2026](https://openai.com/index/devday-2026-recap/)), more than 2 million businesses ([Bloomberg, 31 July 2026](https://news.bloomberglaw.com/artificial-intelligence/openai-has-more-than-1-billion-users-2-million-businesses)), nearly 32 billion API tokens a minute, about 46 trillion a day ([Alphabet, Q3 2026 earnings remarks](https://blog.google/company-news/inside-google/message-ceo/alphabet-earnings-q3-2026/)) and about 4,500 OpenAI staff with 8,000 planned by the end of 2026 (Financial Times, via [Bloomberg, 21 March 2026](https://www.bloomberg.com/news/articles/2026-03-21/openai-plans-to-almost-double-its-headcount-this-year-ft-says)). The link's form and its ceiling of three are judgments: it saturates because a study year's scale is bounded by capacity committed before it.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Every scale halved and doubled (`EXPOSURE_SCALE_VARIANTS`, `INITIAL_ESTIMATE` factors), each variant a separate study at three samples per model. Fixed scales at the as-of or median-path levels, or a growth ceiling of 2 or 5 (`INITIAL_ESTIMATE` ends).

**Challenge mechanisms.** `exposure_scale`. Unavailable: Fixed scales, or a growth ceiling of 2 or 5, need their own collected public study; none is generated.

### POP-SAFEGUARD-LEVEL

**Assumption.** The hypothetical-lab study's safeguards are stipulated text in [`context.json`](config/studies/hypothetical-frontier-lab/context.json) (`SAFEGUARD_LEVEL_FIELDS`), held fixed for the study year. Two alternative postures, `weaker` and `stronger` (`SAFEGUARD_LEVELS`), are authored beside them in `safeguard_levels`. Each changes scope, permissions and the lab's own monitoring together, and leaves who reviews an output or action as stipulated.

**Rationale.** Each level is grounded in published documents, which the config's `rationale` cites. Weaker sits around RAND SL2 weight security. Stronger is the strictest safeguards that the five roster labs' published safety frameworks and safety posts describe for a deployed model and for internal use, applied now: most are attached there to capability thresholds that have not been reached, and several are recommendations for the industry, not commitments, so the level is a posture that no lab is known to run. It sits at RAND SL4, with vetted access to advanced cyber and biological capabilities, monitoring that stops unauthorized model actions, and incident reporting under California SB 53 and the EU General-Purpose AI Code of Practice. It leaves out the strongest measure those documents name, holding a model back or pausing its development, because a level changes neither adoption nor scale. No level changes who reviews an output or action either: user confirmation, customers' oversight and staff review of agent work are the stipulated sentences in every level, and a level says nothing else about review, approval or confirmation (`tests/test_safeguard_level_variants.py`). The taxonomy assigns an incident to a scenario by whether a person considered the output or action, so a level that required or removed that review would move incidents between the agent, automated-decision and reliance scenarios without removing them, and scenarios answered in separate calls do not keep such a movement in balance. The human approval of agent actions that the frameworks also describe is therefore in no level. The stipulated safeguards themselves, "controls typical of a leading frontier lab", cite no source: they describe what leading labs publish about their practice, which incident reports show is not always what is applied. Each text stays within 10% of the stipulated text's word count (`tests/test_safeguard_level_variants.py`).

**Status.** `INITIAL_ESTIMATE`: no incident data validates the respondents' response to the stipulated levels

**Required sensitivity.** Both levels, each a separate study at three samples per model.

**Challenge mechanisms.** `safeguard_level`. Unavailable: Levels that vary scope, permissions or oversight one at a time, or that pair a safeguard level with an exposure variant, need their own collected public study. A level that changes who reviews an output or action needs questions that ask the scenarios it moves incidents between together.

### POP-HISTORY-UNITS

**Assumption.** A scenario's exposure, attribution, terminal count, severity and occurrence units describe the same history, which the hierarchy contract and the occurrence contracts define.

**Rationale.** Counting prompts, payments, reports or affected parties as interchangeable would change the population.

**Status.** `INITIAL_ESTIMATE` for population choices; structural validation is `OK`

**Required sensitivity.** The definition checks of `POP-TERMINAL-INCIDENT`.

**Challenge mechanisms.** Unavailable: Population, unit and threshold alternatives require complete new annual responses.

### POP-ADMISSION

**Assumption.** Every leaf admits an incident when its gross loss, summed over all affected parties and loss kinds before any recovery, is at least USD 100,000 in policy-start USD (`FREQ-MONETARY-BASIS`). The hierarchy contract states the threshold once (`admission`), and it is the floor of every severity law. Incidents below it are a disclosed, unestimated leftover.

**Rationale.** One test on the whole incident lets counts as well as losses add within every parent. USD 100,000 is a judgment, not fit to data.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Admission at USD 10,000 and USD 1,000,000. [`coverage_gaps.yaml`](registries/coverage_gaps.yaml) names the leaves where the leftover is likely material.

**Challenge mechanisms.** Unavailable: Each admission alternative defines a different population and requires complete new frequency and severity answers.

### POP-WHOLE-INCIDENT

**Assumption.** One incident is one root cause, campaign, defect or matter. The whole incident, with all its consequences, belongs to the leaf of the process that caused the largest share of its loss ([methodology](methodology.md#loss-taxonomy)).

**Rationale.** Counting each incident once makes counts and losses add within every parent. The cost is that a minority process's loss is estimated in the majority leaf: an AI-assisted intrusion that supports a bio attack is wholly bio.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Linked parts across leaves; `tests/fixtures/taxonomy_boundary_cases.yaml` holds the boundary cases.

**Challenge mechanisms.** Unavailable: Linked parts across leaves with shared-origin dependence are a different taxonomy and require complete new answers.

### POP-PRIVATE-CLAIMS

**Assumption.** One private civil matter over the provider's own conduct is one incident: a class or representative action, a consolidated or coordinated proceeding, or a coordinated mass arbitration. Matters are split by defining allegation (copyright and IP, privacy and data protection, product and consumer harm, other), dated by first loss and carry their whole-life loss. The provider's payments and its legal and remedy costs are its direct economic loss; its payments to claimants have zero direct welfare weight.

**Rationale.** Procedural consolidation is observable in court records, so frequency and severity share one countable unit. The unit and its boundary with commercial disputes are judgments.

**Status.** `GAP`: no matter census or whole-life loss data supports this unit

**Required sensitivity.** The consolidation, separate-action and contract-dispute cases in the taxonomy boundary fixtures.

**Challenge mechanisms.** Unavailable: A course-of-conduct matter unit or a different boundary with commercial contract disputes defines a different population and requires complete new annual responses.

### POP-TERMINAL-INCIDENT

**Assumption.** Every scenario counts its terminal incidents directly: its counting identity, terminal unit and year rule define one incident, which is one episode with one draw of its loss components. A unit counts when it meets these conditions in fact, whether or not it is discovered, reported or traced to the provider's model.

**Rationale.** Counting what occurs, not what is detected, keeps the estimand the loss itself. No data measures the undetected share, which respondents must judge.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Check each definition against an ignored warning, a costly recovery, a loss at or just below the threshold and repeated incidents in one year.

**Challenge mechanisms.** Unavailable: An alternative incident definition defines a different population and requires complete new annual responses.

### ACC-ATTRIBUTION

**Assumption.** Ecosystem attribution follows the scenario's attribution nexus and a complete allocation of counts to providers. A provider's contribution can include an incident's entire loss, for more than one contributor.

**Rationale.** Conserving allocated counts is useful accounting, but it does not identify unique shared damage or legal recovery.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Complete allocation and ownership alternatives that preserve the population.

**Challenge mechanisms.** Unavailable: Provider allocation and ownership alternatives require separately captured compatible inputs.

### ACC-WELFARE

**Assumption.** The `welfare` view counts resources, injury valuations and appropriated assets once; neutral transfers, fines included, have weight zero. A node's annual lab business loss counts in `gross` and the lab view, not in `welfare` or `net_resource_cost` (`ACC-LAB-ANNUAL-BUSINESS-LOSS`).

**Rationale.** A standing convention: perpetrator proceeds earn no welfare credit, and lost revenue and franchise value are largely transfers to competitors and customers.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** The `net_resource_cost` basis, which also sets appropriated transfers to zero.

**Challenge mechanisms.** Not a study alternative: Where component composition identifies both views, report net_resource_cost beside welfare; otherwise disclose the unavailable view. The two bases are accounting conventions, not uncertainty bounds.

### ACC-LAB-BEARER

**Assumption.** A component whose authored bearer is `provider` (the `subject` of the scenario's composition cell) falls on the named lab. Respondents add a lab compensation share: one point share in [0, 1] of the node's other components that the lab pays to others, applied to the same draw. Provider-borne components plus compensation are capped once per episode (`SEV-LAB-CAP`). The lab's annual business loss is drawn independently of the societal marks (`ACC-LAB-ANNUAL-BUSINESS-LOSS`). An answer may declare the compensation share or the business loss unavailable; its leaf then has no lab-loss law, and the samples that read it are left out of the lab-loss views and the gross views (`POOL-EXCLUDED-SAMPLES`).

**Rationale.** A share of the same draw cannot exceed the harm it compensates. The point share gives the compensation's mean exactly but omits its variation between episodes, and no data identify it.

**Status.** `INITIAL_ESTIMATE` for the bearer classification and the point share; `GAP` for the annual business loss's independence of societal severity

**Required sensitivity.** The provider-components-only lower bound, a full-compensation upper bound and a business loss comonotone with the societal severity rank.

**Challenge mechanisms.** `lab_loss_provider_only`. Unavailable: No arm generates a business loss comonotone with the societal severity rank or a full-compensation bound. Only public-study setups that select it generate the provider-only bound. Another bearer classification needs new captures.

### ACC-LAB-ANNUAL-BUSINESS-LOSS

**Assumption.** Each contextual node elicits the lab's business loss as one annual figure (`lab_annual_business_loss`): its lost revenue, churn and asset impairment from all of the node's incidents in a year with at least one. The answer is a positive probability and p5 through p999 of the positive total. A simulated year with at least one incident draws it once, with the governed tail shape above p95 and the cap of `SEV-LAB-CAP`.

**Rationale.** A lab's reputational and revenue effects saturate: many small incidents in a year move its customers and value broadly once, so a per-incident loss summed over the year would overstate it. The answer does not grow with the year's count beyond one.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** `INITIAL_ESTIMATE` alternatives: the business loss and compensation at zero (`lab_loss_provider_only`), and the cap at USD 300 billion, USD 3 trillion and none (`lab_loss_cap`).

**Challenge mechanisms.** `lab_loss_cap`, `lab_loss_provider_only`. Unavailable: A business loss that rises with the year's incident count, or a per-incident business loss, needs a new question and new captures. Only public-study setups that select them generate the provider-only bound and the lab-cap alternatives.

### ACC-ORIGIN

**Assumption.** Shared origin and component identity come from the complete authored physical joint law.

**Rationale.** Separate provider simulations cannot identify the same physical loss. Authored identities and ownership prevent duplication only within their stated scope.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Authored common-occurrence and nested-ownership alternatives.

**Challenge mechanisms.** Unavailable: Common-occurrence and nested-ownership alternatives are not implemented.

### ACC-COMPOSITION

**Assumption.** Composition shares are expected capped component dollars divided by expected capped severity. Every composition is a roster of at least two cells, named from [`composition_rosters.yaml`](scenarios/composition_rosters.yaml) or authored by the scenario, whose shares the contextual instrument elicits.

**Rationale.** No matched study identifies joint shares, so they are elicited judgments. One aggregate vector does not identify size-dependent composition.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Alternative or size-dependent composition.

**Challenge mechanisms.** Unavailable: Contextual nodes elicit composition as partition shares; alternative or size-dependent composition requires new captures.

### ACC-SCOPE

**Assumption.** Existential or civilizational value without a quantifiable economic measure is excluded where the scenario scope says so; quantifiable loss counts however large. Omitted pathways receive no numerical loading.

**Rationale.** The selected populations and monetary mechanisms do not quantify those harms. Their absence is a scope boundary, not zero risk.

**Status.** `GAP` for wider risk completeness

**Required sensitivity.** A low, base and high loading for unmodeled populations and excluded harms.

**Challenge mechanisms.** Unavailable: No loading for unmodeled populations or excluded harms is implemented; omitted scope is disclosed, not quantified.

## Observable paths and forecast probabilities

| ID | Assumption | Status |
| --- | --- | --- |
| [`OBS-PUBLIC-JOINT`](#obs-public-joint) | Pooled-rate public forecast | `INITIAL_ESTIMATE` |
| [`OBS-PUBLIC-TIMING`](#obs-public-timing) | Anchored lab paths | `INITIAL_ESTIMATE` |
| [`DEP-PUBLIC-CONTEXT`](#dep-public-context) | Whole-path risk conditioning | `INITIAL_ESTIMATE` |
| [`OBS-ECI-SUFFICIENCY`](#obs-eci-sufficiency) | General ECI as the only coordinate | `INITIAL_ESTIMATE` |
| [`OBS-RESPONSE-FORM`](#obs-response-form) | Authored response curves | `INITIAL_ESTIMATE` |
| [`OBS-CONTEXT-CONTINUATION`](#obs-context-continuation) | Fixed-anchor interpolation | `INITIAL_ESTIMATE` |
| [`OBS-CAUSAL`](#obs-causal) | Conditional, not causal, effects | `GAP` |

### OBS-PUBLIC-JOINT

**Assumption.** One positive general-ECI trend rate is pooled across labs and fitted to their frontier-raising releases in Epoch's ECI table, with a pooled restricted-maximum-likelihood trend discrepancy. Each lab draws a pace multiplier independently from a median-one log-logistic law of shape 5, truncated at a pace cap of 4 ([methodology](methodology.md#observable-capability-conditioning)).

**Rationale.** One rate is the simplest model the records support. On the table captured on 2026-10-29 (44 records in a 24-month window) the discrepancy is 1.64 points and the rate 16.2 ± 0.6 a year. A rolling-origin backtest scored shape 8 lowest and shape 5 best at 3 months, and the cap is about twice the fastest lab-year on record (`tests/test_public_frontier_epoch.py`).

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Pace shape 3 and 8, a pace cap of 2, the discrepancy's 90% interval ends (1.12 and 2.27) and no progress, all executed by default. A higher pace cap, or none; correlated or shared lab paces.

**Challenge mechanisms.** `public_forecast`. Unavailable: A pace cap above the forecast's, or none, has paths beyond every node, so it needs a study planned and collected under that cap. Correlated or shared lab paces need new paired captures, and window and prior alternatives need authored inputs.

### OBS-PUBLIC-TIMING

**Assumption.** Each lab path starts at the lab's observed as-of frontier. One rate draw and each lab's own pace draw persist through the policy period, and the public frontier is the pointwise maximum of the lab paths.

**Rationale.** Anchoring at the last published record keeps every draw at or above the frontier already public, but assumes no progress since that record.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** No progress, executed by default; the embedded lower-order trajectory rule; more paths per node.

**Challenge mechanisms.** `public_forecast`. Unavailable: A finer trajectory rule needs its additional nodes collected, so the embedded lower-order rule is the integration check. More representative paths per node need new paired captures.

### DEP-PUBLIC-CONTEXT

**Assumption.** Whole annual risk laws are conditioned on the general-ECI path of a selected lab or of the public frontier. Internal capabilities are implicit, and domain benchmarks, such as cyber or biology evaluations, are context only.

**Rationale.** The shared public path links the public and not-yet-public populations but does not identify their residual dependence.

**Status.** `INITIAL_ESTIMATE`; the validity of the responses is untested

**Required sensitivity.** Stated rationales for residual independence and parameter coupling; domain-specific contexts, which need new collection.

**Challenge mechanisms.** Unavailable: Public-path independence comparisons are offline; alternative residual couplings require separately authored conditional loss laws.

### OBS-ECI-SUFFICIENCY

**Assumption.** General ECI is the only capability coordinate: a fixed-anchor context conditions on one constant score and a public-study context on one whole public trajectory.

**Rationale.** A published external score with captured history avoids latent factors. What a point means in practice is the respondent's judgment. Where the index provenance records it, a question quotes the correspondence [Epoch](https://epoch.ai/eci) reported: about 5 points to a doubling of METR's time horizon.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** A second capability coordinate.

**Challenge mechanisms.** Unavailable: A second capability coordinate requires an observable series, a forecast and new paired captures.

### OBS-RESPONSE-FORM

**Assumption.** Authored `LocalResponse` curves interpolate in value or log between their points. `last_slope` continues the top segment for one top-anchor spacing above the highest point (`LAST_SLOPE_CAP_SPACINGS`) and is flat beyond; every curve is flat below its lowest point.

**Rationale.** The form is authored and has no empirical validation. Uncapped, an exponential continuation under a heavy-tailed path has no finite mean. Neither context-bound route uses the continuation.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Zero to one spacing, a one-sided truncation policy and the other interpolation.

**Challenge mechanisms.** Unavailable: No route generates response-form alternatives; an authored setup declares its own as alternative arms.

### OBS-CONTEXT-CONTINUATION

**Assumption.** Fixed-anchor responses are interpolated linearly between elicited anchors and never extrapolated: the anchor hull must cover the modeled integer range and the reference score.

**Rationale.** Interpolated rate quantiles, branch probabilities and component laws have no supported continuation outside the anchors.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** A wider anchor hull.

**Challenge mechanisms.** Unavailable: A wider anchor hull or a continuation beyond it changes the frozen question and requires a separately collected instrument.

### OBS-CAUSAL

**Assumption.** Capability effects are conditional judgments, not causal ECI coefficients.

**Rationale.** Capability, adoption, controls and consequences can move together, and presenting capability levels together may inflate the contrast between them.

**Status.** `GAP` for causal and predictive validity

**Required sensitivity.** A reference-held comparison at fixed capability; respondent repeatability, framing and held-out diagnostics.

**Challenge mechanisms.** Unavailable: A reference-held response needs new paired captures on a held trajectory; respondent repeatability, framing and held-out diagnostics have no collection protocol.

## Annual frequency and source pooling

| ID | Assumption | Status |
| --- | --- | --- |
| [`POOL-WEIGHTS`](#pool-weights) | Equal respondent weights | `INITIAL_ESTIMATE` |
| [`POOL-EXCLUDED-SAMPLES`](#pool-excluded-samples) | Exhausted calls and excluded samples | `INITIAL_ESTIMATE` |
| [`POOL-UNANSWERED-SCENARIOS`](#pool-unanswered-scenarios) | A model left out of a scenario it does not answer | `INITIAL_ESTIMATE` |
| [`POOL-SHARED-MODEL`](#pool-shared-model) | Shared respondent identity and ranks | `INITIAL_ESTIMATE` |
| [`POOL-PCHIP`](#pool-pchip) | Five-knot PCHIP count law | `INITIAL_ESTIMATE` |
| [`POOL-COMMON-BIAS`](#pool-common-bias) | Common level bias | `INITIAL_ESTIMATE` |
| [`POOL-SPREAD`](#pool-spread) | Respondent spread and its calibration | `INITIAL_ESTIMATE` |
| [`FREQ-PARAMETER-PROCESS`](#freq-parameter-process) | Parameter and process uncertainty | `OK` |
| [`FREQ-TERMINAL`](#freq-terminal) | Authoritative terminal count | `OK` |
| [`FREQ-EVENT-CHAIN`](#freq-event-chain) | Event-chain count route | `INITIAL_ESTIMATE` |
| [`FREQ-CHAIN-EXPOSURE`](#freq-chain-exposure) | Chain exposure bases | `INITIAL_ESTIMATE` |
| [`FREQ-CHAIN-INDEPENDENCE`](#freq-chain-independence) | Independent chain judgments | `INITIAL_ESTIMATE`, `GAP` |
| [`FREQ-CHAIN-POOL`](#freq-chain-pool) | Direct and chain pool | `INITIAL_ESTIMATE`, `GAP` |
| [`FREQ-CHAIN-GUARDRAIL`](#freq-chain-guardrail) | Chain guardrail retries | `INITIAL_ESTIMATE` |
| [`FREQ-EAL-CROSSCHECK`](#freq-eal-crosscheck) | Leaf EAL cross-check | `INITIAL_ESTIMATE`, `GAP` |
| [`HIER-CATEGORY-POOL`](#hier-category-pool) | Taxonomy category pool | `INITIAL_ESTIMATE` |
| [`HIER-REMAINDER`](#hier-remainder) | Unsimulated parent remainders | `GAP` |
| [`HIER-THRESHOLD`](#hier-threshold) | Shared parent threshold | `INITIAL_ESTIMATE` |
| [`FREQ-POISSON`](#freq-poisson) | Poisson counts | `INITIAL_ESTIMATE` |
| [`FREQ-MONETARY-BASIS`](#freq-monetary-basis) | Policy-start monetary basis | `INITIAL_ESTIMATE` |

### POOL-WEIGHTS

**Assumption.** Models have equal weights, and samples have equal weights within each model. Each sample's complete laws are held together through each scenario and year.

**Rationale.** No task-specific performance record supports other weights. Repeated samples of one model are not independent experts.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Single-model results, and results with each model left out in turn, which need at least three models.

**Challenge mechanisms.** `model_pooling`, `model_pooling_leave_one_out`

### POOL-EXCLUDED-SAMPLES

**Assumption.** A call is exhausted when every validation attempt failed, a provider's refusal or output-limit stop included. A sample, one respondent at one trajectory node, lacks a leaf whose direct question is exhausted. Where more than `max_excluded_sample_share` (0.10) of a model's samples lack one leaf, the model does not answer it (`POOL-UNANSWERED-SCENARIOS`). Any other sample that lacks a leaf is excluded at import, and its model's weight stays with its remaining samples. Import refuses a study in which a model keeps fewer than half its samples at a node, or in which an excluded share exceeds the control ([methodology](methodology.md#pooling-and-parameter-uncertainty)). A direct answer that declares the lab's own loss unavailable excludes no sample and changes no weight. A view that counts the lab's own loss is computed on the samples that give it: the lab-loss views, and the gross views, because `gross` adds the lab's annual business loss (`ACC-LAB-ANNUAL-BUSINESS-LOSS`) and a state without that law would omit it. The sample whose states read such an answer, as its own answer or a donor's, is left out of those views: there it weighs zero, and its model's other samples at the node share its weight in every arm. Welfare, net resource cost and their scenario and source-status views keep every sample at its weight. The same limits bound the excluded and left-out samples together; where they would be breached, no sample is left out and the lab-loss views and the gross views are withheld.

**Rationale.** A missing answer cannot be imputed without fabricating a judgment, and a sample compiled without one leaf would understate its loss. Refusals are content-linked, so kept samples can differ from excluded ones. At about 60 direct questions per sample, independent exhaustions reach a 10% excluded share at about 1.75 per 1,000 questions. Whether a respondent judged the lab's loss does not bear on its societal judgment: ten samples, one with a societal estimate of 100 and nine with 1, have a mean of 10.9, and excluding the first for its lab answer would report 1.0. The lab-loss views and the gross views hold the samples that gave the lab's loss, which can differ from the ones that declared it unavailable.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** The control at 0.05 and 0.20, by reimporting with `max_excluded_sample_share`.

**Challenge mechanisms.** Unavailable: No result with the excluded samples exists, because the answers they lack do not, and no lab-loss or gross view with a left-out sample, because its lab-loss answer does not. Two computable alternatives are not generated: a reweighting that excludes each excluded respondent at every node, and one that leaves each left-out sample out of every view.

### POOL-UNANSWERED-SCENARIOS

**Assumption.** A model does not answer a leaf when more than `max_excluded_sample_share` (0.10) of its samples, over every node, lack it. Its answers to the leaf are then set aside. Each of its kept samples takes, for that leaf alone, the whole case of a kept sample of an answering model, chosen by a fixed rotation ([methodology](methodology.md#pooling-and-parameter-uncertainty)). A sample that borrows a case declaring the lab's own loss unavailable is left out of the lab-loss views and the gross views with its donor (`POOL-EXCLUDED-SAMPLES`). A variant study takes the unanswered pairs and donors of the study it varies. Import refuses a study in which no model answers a leaf, or in which a model does not answer more than half of the leaves.

**Rationale.** A refusal is content-linked and concerns one scenario, so excluding the model's samples whole would discard its other answers. Borrowing whole cases keeps every state a complete set of laws. A borrowed case repeats its donor's answers, so the samples are not independent. Three diagnostics count them as independent and are not reported where a borrowing state has weight, each with its reason: the respondent-sampling standard error (`NUM-RESPONDENT-SAMPLING`), the EAL parameter band (`NUM-EAL-PARAMETER-BAND`) and the standard deviation of a trajectory node's state EALs in the trajectory quadrature indicator (`state_eal_sd`). None is computed over the answering samples alone: a borrowing state's category pools combine its own answers with the donor's, so its laws are neither a copy of the donor's nor independent of them.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** The control at 0.05 and 0.20 (`POOL-EXCLUDED-SAMPLES`).

**Challenge mechanisms.** Unavailable: No result uses a model's own answers to a scenario it does not answer: the answers its samples lack do not exist, and the rest are a selected subset. Another donor assignment is computable from the same captures but not generated.

### POOL-SHARED-MODEL

**Assumption.** Each joint state shares one respondent identity across every node and source status. Each simulated year of a state reads one parameter rank per scenario and source status, independent across scenarios.

**Rationale.** Sharing identity keeps a respondent's correlated judgments together. It is a judgment: matching independent stateless samples is not evidence of a joint posterior.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Independent-respondent selection by node, and comonotone node ranks as the upper coupling reference.

**Challenge mechanisms.** Unavailable: Respondents are matched across nodes and source statuses; independent-respondent coupling and comonotone node ranks are not generated.

### POOL-PCHIP

**Assumption.** Each count law is a five-knot PCHIP through p5, p25, p50, p75 and p95, with local normal-score log tails whose width is capped at Φ⁻¹(0.95) ≈ 1.645 ([methodology](methodology.md#pooling-and-parameter-uncertainty)).

**Rationale.** Five judgments do not identify tails, so the tail form and its cap are a regularization convention. The cap is the largest width at which the upper tail's mean integrand peaks at p95.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** `INITIAL_ESTIMATE` tail-width factors 0.75 and 1.5 on the capped widths, keeping the five quantiles.

**Challenge mechanisms.** `epistemic_tail_width`. Unavailable: Fixed-anchor contextual setups execute only their reference arm, so only public-study setups generate the tail-width arms; global lognormal parameter-law fits are not implemented.

### POOL-COMMON-BIAS

**Assumption.** A shared multiplicative level bias can move every respondent model's count laws together, and equal-weight pooling does not remove it. Every setup leaves it uncorrected, at b = 1.

**Rationale.** No evidence measures a level bias on the risk questions. The calibration's b_hat comes from known-count seeds, 37 of whose 43 primary windows lie in 2026 Q3 ([`seeds.yaml`](data/calibration/respondent_spread/seeds.yaml)), and its transfer to risk questions is unvalidated.

**Status.** `INITIAL_ESTIMATE`: b = 1 is an untested judgment

**Required sensitivity.** `INITIAL_ESTIMATE` factors 0.5 and 2 on every count quantile and expected count, with severity and shares fixed.

**Challenge mechanisms.** `common_frequency_bias`. Unavailable: Fixed-anchor contextual setups execute only their reference arm, so they do not generate common frequency-bias alternatives.

### POOL-SPREAD

**Assumption.** Every setup reads the respondents' stated count quantiles at face value, k = 1 and b = 1. The respondent spread calibration, which fits a spread k and a level bias b on known-count questions, is a diagnostic that no setup reads ([methodology](methodology.md#respondent-spread-calibration)).

**Rationale.** The fit measures how wide and how biased one-shot answers to known-count questions were in one period. Its transfer to speculative conditional counts is unvalidated, so no result rests on it.

**Status.** `INITIAL_ESTIMATE`: face value is an untested judgment

**Required sensitivity.** `INITIAL_ESTIMATE` spread alternatives k = 1.5 and 0.75, a normal-score stretch of every count law that preserves medians, capped after the stretch (`POOL-PCHIP`).

**Challenge mechanisms.** `epistemic_spread_calibration`

### FREQ-PARAMETER-PROCESS

**Assumption.** Conditional parameter uncertainty is distinct from process variation.

**Rationale.** A quantile of the expected annual count cannot replace a count-family distribution. Severity quantiles describe occurrence losses, not epistemic frequency or annual return periods.

**Status.** `OK` dimensional rule

**Required sensitivity.** Native analytic expected-count, severity-mean and compound-episode oracles.

**Challenge mechanisms.** Not a study alternative: A dimensional rule with no alternative value, which the analytic oracles check.

### FREQ-TERMINAL

**Assumption.** The expected annual count of qualifying terminal units is authoritative: the respondent's direct count after its leaf EAL cross-check (`FREQ-EAL-CROSSCHECK`), pooled with its chain count (`FREQ-CHAIN-POOL`). No gate, adoption or provider-share factor multiplies it.

**Rationale.** This avoids eliciting values that cancel, or applying terminal qualification twice.

**Status.** `OK` accounting; elicited levels are judgmental

**Required sensitivity.** The `INITIAL_ESTIMATE` negative-binomial alternative at fixed mean (`FREQ-POISSON`).

**Challenge mechanisms.** `count_family`

### FREQ-EVENT-CHAIN

**Assumption.** Each scenario's authored `event_chain` is a second, bottom-up route to the same annual count. It states an annual opportunity, an exposure base (`FREQ-CHAIN-EXPOSURE`) and one to four authored stages before the terminal unit. The chain count is the base count times the elicited opportunity rate times every stage's probability ([methodology](methodology.md#event-populations-and-frequency)). In the eleven scenarios in which a model helps an attacker, the first stage is the gain the attacker would not have had without any AI model.

**Rationale.** Decomposing the count into judgeable transitions gives a second construction of it. Each chain's structure is an authored judgment without matched data.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Check that every stage is necessary for the terminal and that no stage applies the admission threshold.

**Challenge mechanisms.** Unavailable: An alternative chain structure is a different chain question and requires new chain answers.

### FREQ-CHAIN-EXPOSURE

**Assumption.** Each chain names an exposure base for each capability access (`event_chain.exposure`): a natural unit with a stipulated count. A derived count is one of the study's deployment scales times an authored factor, so it moves with the trajectory; a fixed count is one number the scenario authors. A base's `basis` is `judgment`, `data` or `normalisation`. A normalisation base only measures the lab's size, so its cell has no governed exposure denominator.

**Rationale.** A stipulated base states the count's scale once, so the respondent never converts users or tokens into opportunities.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** The `exposure_scale` variants (`INITIAL_ESTIMATE` factors, `POP-HYPOTHETICAL-LAB`) halve and double every derived count.

**Challenge mechanisms.** `exposure_scale`. Unavailable: An alternative exposure base, factor or derivation changes the chain question and requires new chain answers.

### FREQ-CHAIN-INDEPENDENCE

**Assumption.** A respondent's opportunity-rate and stage-probability judgments are combined as independent epistemic quantities in log space, on a fixed scrambled Sobol design of 2^14 points. Each is read only between its own p5 and p95 and held at them beyond.

**Rationale.** The mean of a product of wide judged laws is dominated by tails no respondent stated. Independence is a judgment: shared optimism across stages would make the true law wider.

**Status.** `INITIAL_ESTIMATE`; the dependence between chain judgments is an unmeasured `GAP`

**Required sensitivity.** An alternative coupling of the chain's components.

**Challenge mechanisms.** Not a study alternative: No alternative coupling is reported: it would need a dependence parameter that no question elicits. The frequency_estimate exhibit reports the reference EAL under the chain-only count law.

### FREQ-CHAIN-POOL

**Assumption.** Each leaf's count law is a fixed pool of the cross-checked direct count law and the chain count law. At each of p5, p25, p50, p75 and p95 it is exp(0.5 ln direct + 0.5 ln chain), the equal-weight average of the two quantile functions in log space. A wholly zero or missing law leaves the other alone.

**Rationale.** A mechanical pool fixed before the answers are seen cannot pick a tail. Mechanical aggregation is preferred to behavioral aggregation (Clemen & Winkler 1999, *Risk Analysis*), and equal weights are a robust default without a track record (Clemen 1989, *International Journal of Forecasting*; Armstrong 2001, *Principles of Forecasting*). A decomposed estimate combined with a holistic one improves on either where both are credible (MacGregor & Armstrong 1994, *International Journal of Forecasting*). Averaging quantiles keeps the pooled law's shape (Lichtendahl, Grushka-Cockayne & Winkler 2013, *Management Science*), and stated weights on alternative methods follow logic-tree practice in seismic hazard analysis (US NRC, NUREG-2117).

**Status.** `INITIAL_ESTIMATE` weight; the correlation of the two estimates' errors is an unmeasured `GAP`

**Required sensitivity.** Weight 0 (direct only) and weight 1 (chain only).

**Challenge mechanisms.** Not a study alternative: The frequency_estimate exhibit reports the reference EAL under the direct-only (weight 0) and chain-only (weight 1) counts, which bracket the pool weight 0.5. Tail statistics would need simulated arms, which are not generated.

### FREQ-CHAIN-GUARDRAIL

**Assumption.** A chain answer gets one retry when its opportunity rate has p95/p5 above 1,000 or its terminal stage has p50 below 10^-3. The retry states the values that tripped it, without the thresholds, and asks the respondent to confirm or revise. Its answer drives the chain even where it still violates.

**Rationale.** Both thresholds are judgments that no data identify. The retry refuses nothing, because a wide rate or a rare terminal can be a true judgment.

**Status.** `INITIAL_ESTIMATE` thresholds

**Required sensitivity.** The EAL with every chain answer as first given.

**Challenge mechanisms.** Not a study alternative: The frequency_estimate exhibit reports how the retries ended, and its before_chain_guardrail arm gives the reference EAL with every chain answer as first given. Looser thresholds can be computed from stored calls; stricter ones need new retry calls.

### FREQ-EAL-CROSSCHECK

**Assumption.** Every leaf's direct question also asks for its direct EAL: p5–p95 of the respondent's uncertainty about the node's expected annual societal harm. Before pooling, its mean is compared with the implied EAL, the direct count's mean times the compiled law's analytic welfare mean. Where they differ by more than a factor of three, one more call shows both and asks which judgments, if any, the respondent revises ([methodology](methodology.md#event-populations-and-frequency)).

**Rationale.** Wide count ranges combined with heavy severity tails can imply losses far above what a respondent reasoning from medians intends. The direct EAL is a check that is never simulated, and it can itself be the wrong judgment or anchor a revision.

**Status.** `INITIAL_ESTIMATE` design and factor of three; anchoring and the direct EAL's own error are unmeasured `GAP`s

**Required sensitivity.** EAL without the cross-check, reverting every revision.

**Challenge mechanisms.** Not a study alternative: The frequency_estimate exhibit's before_eal_crosscheck arm reports the reference EAL with the count and law before the cross-check. A higher factor can be computed from stored calls by reverting the cases it would not call; a lower one needs new cross-check calls.

### HIER-CATEGORY-POOL

**Assumption.** Every split of the taxonomy whose subtree holds an estimated scenario is a parent, asked in its own call for its whole-category EAL and for a remainder covering what its modeled children miss. Each parent's answer is pooled with what is beneath it by a fixed rule, bottom-up, whatever the size of the gap. D is the mean of the parent's category law, and M the sum of its children's mean EALs and its remainder's mean. With h the parent's height (1 where its modeled children are all leaves), the factor is g = (D / M)^w with w = 1 / (h + 1). The pooled value M × g is thus the weighted geometric mean of M and D. g scales the count law and both large-loss counts of every leaf beneath the parent and leaves severity as stated. Each leaf's cumulative factor is clamped to 0.0001 to 10,000 ([methodology](methodology.md#event-populations-and-frequency)).

**Rationale.** Judgments of a whole and of its unpacked parts disagree systematically (Tversky & Koehler 1994, *Psychological Review*), and nothing identifies which is right, so a fixed pool gives both weight. With w = 1 / (h + 1), the leaves' sum and every category answer above it weigh equally along the tallest branch. The weight depends on the taxonomy's grouping as well as on the answers: a level added on a parent's tallest branch lowers that parent's weight, even where the added category's answer equals the sum beneath it.

**Status.** `INITIAL_ESTIMATE`: the weight, the allocation to frequency alone and the clamp are judgments

**Required sensitivity.** Weight 0 (the leaves alone) and weight 1 at every parent.

**Challenge mechanisms.** Not a study alternative: The frequency_estimate exhibit reports the reference EAL at the two ends of the category answer's weight: before_hierarchy (weight 0) and hierarchy_parents_only (weight 1 at every parent). Across levels the run's EAL need not lie between the two arms. Tail statistics would need simulated arms, which are not generated.

### HIER-REMAINDER

**Assumption.** Each parent's remainder, the part of its category outside its modeled children, is elicited and reported as an estimated, unsimulated expected annual loss. It is scaled by the pool factors above it (`HIER-CATEGORY-POOL`) and never enters a simulated total or tail.

**Rationale.** An unquantified population has no severity or count law, so adding it to totals would mix an unsimulated judgment into simulated results.

**Status.** `GAP`: no population or law exists for the remainder

**Required sensitivity.** Authored populations for the unestimated leaves.

**Challenge mechanisms.** Unavailable: The remainder has no count or severity law; simulating it needs authored populations for the unestimated leaves.

### HIER-THRESHOLD

**Assumption.** Every parent question states the one admission threshold that every leaf shares (`POP-ADMISSION`), and the remainder covers only categories with no modeled child.

**Rationale.** With one threshold, a parent's admitted incidents are exactly its children's, so no losses fall between thresholds.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Another admission threshold (`POP-ADMISSION`).

**Challenge mechanisms.** Unavailable: Another admission threshold is a different parent question and leaf population that need new answers.

### FREQ-POISSON

**Assumption.** Repeated counts and episode arrivals are Poisson, with variance-to-mean ratio 1, unless a node declares a negative-binomial family.

**Rationale.** Sparse incident data do not identify overdispersion: negative-binomial laws fit daily counts of all reported data breaches, while Poisson fits large-breach frequency (Edwards, Hofmeyr and Forrest 2016, *Journal of Cybersecurity* 2(1)).

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** A negative-binomial variance-to-mean ratio of 2 at fixed mean, an `INITIAL_ESTIMATE` judgment of mild overdispersion that leaves EAL unchanged.

**Challenge mechanisms.** `count_family`

### FREQ-MONETARY-BASIS

**Assumption.** Elicited amounts use the policy-start monetary basis, and full-year answers already include prospective context. Every amount is converted at the exchange rate of the date it occurs or is imposed and deflated to the policy start by US consumer prices.

**Rationale.** This avoids a second growth or inflation adjustment after collection and keeps inflation from moving incidents across the admission threshold.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Another monetary basis or growth adjustment.

**Challenge mechanisms.** Unavailable: Alternative monetary bases or growth adjustments require compatible new answers.

## Severity and tails

Each scenario's governed tail values are listed in [`scenario_catalog.json`](scenarios/scenario_catalog.json) under `scenarios[].severity_governance`. The three tail classes and the catastrophe endpoint convention are authored once in [`severity_governance_policies.yaml`](scenarios/severity_governance_policies.yaml). The shapes and caps are judgments: no governed shape comes from a same-unit occurrence-loss fit, and heavier tails than the class shapes cannot be excluded.

| ID | Assumption | Status |
| --- | --- | --- |
| [`SEV-QUANTILE-LAW`](#sev-quantile-law) | Quantile severity law | `INITIAL_ESTIMATE` |
| [`SEV-GPD-SHAPE`](#sev-gpd-shape) | Tail-class GPD shapes | `INITIAL_ESTIMATE`, `OK` |
| [`SEV-LARGE-LOSS-ANCHOR`](#sev-large-loss-anchor) | Large-loss anchor | `INITIAL_ESTIMATE` |
| [`SEV-ONE-LAW`](#sev-one-law) | One severity law per scenario-year | `INITIAL_ESTIMATE` |
| [`SEV-INFINITE-VARIANCE`](#sev-infinite-variance) | Infinite severity variance | `OK` |
| [`SEV-CAP`](#sev-cap) | Scenario caps and endpoints | `INITIAL_ESTIMATE` |
| [`SEV-LAB-CAP`](#sev-lab-cap) | Lab enterprise-value cap | `INITIAL_ESTIMATE` |
| [`SEV-EMPIRICAL-TRANSFER`](#sev-empirical-transfer) | Empirical record transfer | `GAP` |

### SEV-QUANTILE-LAW

**Assumption.** Elicited p5, p25, p50, p75, p95 and p99 describe one qualifying occurrence. The body is a PCHIP spline of log excess over the floor against the normal score, and a GPD tail starts at p95 with a scale that reproduces p99 ([methodology](methodology.md#conditional-severity)).

**Rationale.** Interpretable quantiles replace direct tail-parameter judgments, but sparse quantiles do not identify the deep tail. The body form is a convention that keeps the [p5, p95] integral within 1.5% of eleven analytic laws (`tests/test_quantile_spline_body.py`).

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** The tail-shape step (`SEV-GPD-SHAPE`, an `INITIAL_ESTIMATE` step). The p99 excess over p95 ×0.7 and ×1.3 (`INITIAL_ESTIMATE`), and the pre-cap mean ×10^-0.25 and ×10^0.25 (`PLACEHOLDER`). Splice-threshold and winsorization alternatives.

**Challenge mechanisms.** `tail_shape`. Unavailable: Severity-location and p99-excess alternatives are not generated; splice-threshold and winsorization alternatives are not implemented.

### SEV-GPD-SHAPE

**Assumption.** The GPD shape comes from the scenario's tail class: `liability_enforcement` 0.5, `cyber_operational` 0.7 or `mass_casualty_catastrophe` 0.8. Respondents never see the shape or the cap, so a governance revision reprices the same answers.

**Rationale.** The shapes are judgments informed by analog evidence, tabulated in [`references/calibration/README.md`](references/calibration/README.md). NPDB malpractice payments give 0.534 for the liability class. Basel operational-loss shapes above 1 (Moscadelli 2004) and infinite-mean cyber losses (Malavasi et al. 2022) lie above the cyber/operational class. Terrorism deaths at 0.6 to 0.8 (Clauset and Wiegel 2010) and disputed pandemic estimates near 1.6 (Cirillo and Taleb 2020) motivate the capped catastrophe class.

**Status.** `INITIAL_ESTIMATE`; finite-mean and finite-variance mathematics `OK`

**Required sensitivity.** Shape ±0.2 (`INITIAL_ESTIMATE` step) on every law with a GPD component, keeping its knots and p99. Heavier published estimates lie outside the step.

**Challenge mechanisms.** `tail_shape`

### SEV-LARGE-LOSS-ANCHOR

**Assumption.** Every leaf's direct question also asks the expected annual numbers of admitted incidents of at least USD 1 billion (the anchor) and USD 100 billion (the second count). The share π = E[large-loss count] / E[count] is held fixed across count ranks. Where USD 1 billion lies above the law's p99 and 0 < π < 0.01, the tail above p99 is reshaped so that S(USD 1 billion) = π: a power law up to USD 1 billion, then the governed tail conditioned on reaching it. A share that contradicts the quantiles gets one neutral retry; an unresolved case compiles the p99-only tail (`severity_unresolved`). Only the first count enters a reference law ([methodology](methodology.md#conditional-severity)).

**Rationale.** A leaf's stated p99 can lie decades below a catastrophic loss, which would leave that tail to the governed shape's extrapolation. The anchor is an LLM judgment of rare-event frequency with no calibration record. The second count is rarer still and has no coherence retry, so it shapes no reference law.

**Status.** `INITIAL_ESTIMATE`; the anchor values are elicited judgments

**Required sensitivity.** The analytic `large_loss_anchor` exhibit gives the EAL effects of the anchor ignored, of each shaping share's conditional odds multiplied by ⅓ and 3 (an `INITIAL_ESTIMATE` factor) and of the two-anchor tail. On a synthetic leaf the odds factors move welfare EAL by about −37% and +64%, and little for a share near 0.01 (`tests/test_large_loss_reporting.py`). On synthetic leaves a second share ten times below or above the shape's own moves it by −7% to −33% or +20% to +209% (`tests/test_two_anchor_comparison.py`).

**Challenge mechanisms.** `large_loss_anchor`, `large_loss_anchor_ignored`, `very_large_loss_anchor`. Unavailable: Only public-study setups that select the families simulate the anchor alternatives and the two-anchor tail; fixed-anchor setups execute only their reference arm. Every contextual result carries the analytic large_loss_anchor exhibit.

### SEV-ONE-LAW

**Assumption.** One complete respondent severity law is held for each scenario and year.

**Rationale.** This avoids treating model disagreement as event-level heterogeneity. Within-law event randomness remains.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** Fixed-model portfolios, each conditioned on one model's complete laws.

**Challenge mechanisms.** `model_pooling`

### SEV-INFINITE-VARIANCE

**Assumption.** An uncapped shape of at least 0.5 gives infinite severity variance, although a shape below one gives a finite mean.

**Rationale.** A finite sample variance cannot estimate an infinite population variance.

**Status.** `OK` mathematical property

**Required sensitivity.** Results mark precision unavailable for each affected view.

**Challenge mechanisms.** Not a study alternative: A mathematical property of the GPD; results suppress unsupported dollar variance and precision instead of varying it.

### SEV-CAP

**Assumption.** A finite cap applies to one episode's whole loss, after the uncapped law is built, and every cap or endpoint stress records its basis. Every catastrophe-class cell shares one endpoint: a USD 100 trillion cap on one episode's origin-year ultimate loss. The other two classes are uncapped.

**Rationale.** USD 100 trillion is somewhat below one year of world output (IMF World Economic Outlook database, read on 2026-11-11: USD 111.9 trillion in 2024). It is a valuation convention, not a physical bound. Caps bound one episode only, so annual losses summed over cells and repeated events can exceed world output.

**Status.** `INITIAL_ESTIMATE`, judgment-dominant

**Required sensitivity.** Catastrophe class: USD 10 trillion (below the IMF's January 2022 projection of USD 13.8 trillion cumulative output loss from COVID-19), USD 500 trillion (about global personal wealth; UBS Global Wealth Report 2025) and uncapped. A scenario-authored finite cap: its authored endpoints, or 75% and 125% of the cap. The enforcement leaf: a USD 50 billion endpoint stress (about the EU AI Act's Article 99(3) maximum).

**Challenge mechanisms.** `uncapped_severity`, `severity_cap`, `severity_endpoint_stress`. Unavailable: A cap or endpoint challenge requires an authored outer cap or finite endpoint and a transform consistent with the retained qualification population; missing lower-threshold count/mark inputs are not inferred.

### SEV-LAB-CAP

**Assumption.** Each episode's lab loss from provider-borne components plus compensation is capped at a stipulated frontier-lab enterprise value, and so, separately, is each node's annual business loss. The value is USD 1 trillion for every lab (`FRONTIER_LAB_VALUE_USD` in [`sensitivity_catalog.py`](src/catmodel/setup/sensitivity_catalog.py)), or the tail-class endpoint where that is lower.

**Rationale.** Limited liability: a lab cannot bear a loss larger than its enterprise value. The round value is a stipulation with weak grounding, and it bounds each episode and each annual business loss separately, so a year's lab loss can exceed it.

**Status.** `INITIAL_ESTIMATE`

**Required sensitivity.** USD 300 billion and USD 3 trillion (`INITIAL_ESTIMATE` factor-three stresses), and no lab cap.

**Challenge mechanisms.** `lab_loss_cap`. Unavailable: Fixed-anchor contextual setups execute only their reference arm, so only public-study setups that select the family generate the lab-cap alternatives.

### SEV-EMPIRICAL-TRANSFER

**Assumption.** Empirical loss records do not automatically match runtime occurrence distributions.

**Rationale.** Reporting, selection, censoring, attribution and monetary scope differ across complaints, fines, cyber losses and malpractice reports.

**Status.** `GAP` for many transfers

**Required sensitivity.** Captured source and population fits.

**Challenge mechanisms.** Unavailable: Source-to-target transfer alternatives require captured source and population fits.

## Dependence and financial assumptions

| ID | Assumption | Status |
| --- | --- | --- |
| [`DEP-CONDITIONAL-INDEPENDENCE`](#dep-conditional-independence) | Conditionally independent nodes | `INITIAL_ESTIMATE` |
| [`DEP-RESIDUAL-FACTOR`](#dep-residual-factor) | Gamma residual factor | `PLACEHOLDER` |
| [`DEP-MIXTURE`](#dep-mixture) | Whole-distribution mixtures | `OK` |
| [`FIN-POLICY-TERMS`](#fin-policy-terms) | Authored policy terms | `INITIAL_ESTIMATE`, `PLACEHOLDER` |
| [`FIN-TRIGGER`](#fin-trigger) | No contract trigger | `INITIAL_ESTIMATE` |
| [`FIN-NOMINAL-TERMS`](#fin-nominal-terms) | Fixed nominal terms | `INITIAL_ESTIMATE` |
| [`FIN-LIABILITY`](#fin-liability) | Optional liability allocation | `GAP`, `PLACEHOLDER` |

### DEP-CONDITIONAL-INDEPENDENCE

**Assumption.** Given its capability anchor or public trajectory and the shared respondent identity, a context-bound setup treats distinct nodes and source-status populations as conditionally independent, with explicit physical origin sharing.

**Rationale.** No empirical evidence establishes residual conditional independence. Each incident belongs wholly to one leaf (`POP-WHOLE-INCIDENT`), so no incident starts events in two nodes.

**Status.** `INITIAL_ESTIMATE`, material structural uncertainty

**Required sensitivity.** Independent-respondent pairing and comonotone node ranks. Gamma intensity stresses (`DEP-RESIDUAL-FACTOR`) and Gaussian count-copula stresses at latent correlations 0.25, 0.5 and 0.75 (`PLACEHOLDER` values), where an authored setup declares them. Every run reports a within-state comonotone reference, a TVaR upper bound for its conditional marginals.

**Challenge mechanisms.** Unavailable: Independent-respondent coupling, comonotone node ranks and count-dependence stresses are not generated; additional dependence or common-occurrence alternatives require captured complete joint laws.

### DEP-RESIDUAL-FACTOR

**Assumption.** A mean-one gamma factor multiplies conditional initiating rates once per simulated year: portfolio CV 0.5, 1 and 2, and risk-domain CV 1 where relevant. Each shared-factor alternative has an independent-origin control at the same CV.

**Rationale.** The factor is an unfitted sector-factor stress that keeps conditional mean counts: for Poisson origins it adds λ·CV² to the variance-to-mean ratio. No matched AI incident series grounds the CV or the grouping.

**Status.** `PLACEHOLDER`

**Required sensitivity.** The CV grid with matched controls.

**Challenge mechanisms.** Unavailable: Gamma-intensity stresses are not generated; an authored setup declares them with matched controls.

### DEP-MIXTURE

**Assumption.** Mixture probabilities weight whole conditional distributions.

**Rationale.** Total probability and expectation preserve the forecast law. Averaging conditional quantiles or TVaR would not.

**Status.** `OK` mathematical identity

**Required sensitivity.** None: a total-probability identity.

**Challenge mechanisms.** Not a study alternative: A total-probability identity with no alternative value; alternative forecasts reweight unchanged stored states, and draw-budget changes never alter weights.

### FIN-POLICY-TERMS

**Assumption.** Occurrence identity and policy and annual aggregate terms are explicitly authored.

**Rationale.** Related claims and shared contract limits can materially change payouts. A ground-up schedule is a declared analytical choice, not an actual insured policy.

**Status.** `INITIAL_ESTIMATE` or `PLACEHOLDER`

**Required sensitivity.** Other occurrence mappings, retentions and limits.

**Challenge mechanisms.** Unavailable: Occurrence-mapping, retention and limit alternatives require separately captured compatible packages.

### FIN-TRIGGER

**Assumption.** A contract responds to the losses that the origin-year ultimate basis assigns to the study year. It has no trigger, retroactive date or reporting window of its own.

**Rationale.** The contract pays the eventual loss of incidents originating in the study year however late they are claimed, like an occurrence trigger with no reporting limit.

**Status.** `INITIAL_ESTIMATE` structural convention

**Required sensitivity.** A population on another temporal basis, such as claims-made.

**Challenge mechanisms.** Unavailable: Claims-made reporting windows, retroactive dates and extended reporting periods are not implemented; another temporal basis would need a new setup field and a separately authored compatible population.

### FIN-NOMINAL-TERMS

**Assumption.** Retentions and limits are fixed amounts applied directly to losses stated in the population's currency and monetary basis.

**Rationale.** An actual policy's terms are fixed nominal amounts. Claim inflation before settlement lowers the real retention and limit, which usually raises payments to a layer with a retention; that effect is not modeled.

**Status.** `INITIAL_ESTIMATE`; the omitted effect is unquantified

**Required sensitivity.** A separate setup with every retention and limit divided by 1.1 and by 1.25 (`PLACEHOLDER`: about 3% a year for three years and 4.5% a year for five).

**Challenge mechanisms.** Unavailable: Claim-inflation and settlement-delay alternatives are not implemented; deflated terms require a separately authored compatible setup.

### FIN-LIABILITY

**Assumption.** Liability allocation is optional. It enters only through authored payment-kernel parameters (`NativeLiabilityParameters` in [`components.py`](src/catmodel/contracts/components.py)) in a hand-authored setup, sampled as a recovery probability times a conditional Beta share ([`liability.py`](src/catmodel/probability/liability.py)).

**Rationale.** `setup import` produces none of these parameters and none has a default, so a setup without them allocates no liability. Analog legal judgments do not establish AI case outcomes.

**Status.** `GAP` for empirical AI allocation; authored values and the challenge steps are `PLACEHOLDER` judgment

**Required sensitivity.** In a setup that authors them: recovery ±0.15, unconditional mean ±0.05, concentration ×0.5 and ×1.5, an authored severity slope ±0.005, an authored regulatory recovery uplift against zero, and another parameter set where the plaintiff relationship or jurisdiction is uncertain.

**Challenge mechanisms.** Unavailable: No study alternative varies liability parameters; each alternative requires a separately authored compatible setup, and collected evidence has no finance input.

## Numerical controls and validation

| ID | Control |
| --- | --- |
| [`NUM-BUDGETS`](#num-budgets) | Starting years and work allowances |
| [`NUM-PRECISION`](#num-precision) | Control-variate TVaR |
| [`NUM-TAIL-GATE`](#num-tail-gate) | Tail convergence gate |
| [`NUM-STREAMS`](#num-streams) | Keyed random streams |
| [`NUM-SMALL-EVENTS`](#num-small-events) | Summed small events |
| [`NUM-SEVERITY-DOMAIN`](#num-severity-domain) | Severity numerical domain |
| [`NUM-EAL-PARAMETER-BAND`](#num-eal-parameter-band) | EAL parameter band |
| [`NUM-RESPONDENT-SAMPLING`](#num-respondent-sampling) | Respondent-sampling error |
| [`NUM-CAPABILITY-SCENARIOS`](#num-capability-scenarios) | Capability scenario paths |
| [`NUM-PLAUSIBILITY`](#num-plausibility) | Plausibility checks |
| [`NUM-CALL-LIMITS`](#num-call-limits) | Output and deadline limits |
| [`NUM-CONNECTION-LOST`](#num-connection-lost) | Lost-answer re-sends |
| [`NUM-PROVIDER-FAILURES`](#num-provider-failures) | Provider-failure re-sends |

### NUM-BUDGETS

**Control.** `setup import` starts every state at 1,024 simulated years, which the tail gate doubles up to each arm's cap (`NUM-TAIL-GATE`). It freezes work allowances per simulated year of a state: 50,000 individually drawn events and 50,000 realized campaign members, with 2,000,000 members per campaign. Over-limit work is refused, not clipped.

**Rationale.** These are execution defaults, not precision targets: 1,024 years is 64 in each of the gate's 16 batches, and the allowances are compute guards. All are `INITIAL_ESTIMATE` numerical judgments.

**Required numerical comparison.** Allowances halved and doubled; a run gives the same results under every allowance it fits.

**Challenge mechanisms.** Unavailable: Other process-year budgets and work allowances are not executed as study alternatives; each is a separate import through the import functions' `years_per_state`, `max_reference_state_years`, `max_comparison_state_years` and `work_allowances`.

### NUM-PRECISION

**Control.** TVaR is a control-variate estimate: the sample tail mean plus β times the analytic EAL less the simulated years' mean, over 1 − α, floored at VaR. β is 0 without an analytic mean, 1 for a law without a finite second moment, and otherwise the regression coefficient Cov((L − VaR)₊, L)/Var(L), clipped to [0, 1]. An integrated mean is certified to 0.1% (`CERTIFIED_INTEGRATION_RELATIVE_ERROR`; [methodology](methodology.md#numerical-execution-and-outputs)).

**Rationale.** Without the control, a TVaR is a sample tail mean: biased low for heavy tails, with no half-width where the law has no finite second moment. A coefficient estimated from the same years keeps part of that bias.

**Required numerical comparison.** Quadrature tolerance 1e-4, 1e-3 and 1e-2; `tvar_estimator` reports each coefficient.

**Challenge mechanisms.** Not a study alternative: Seeded tests of estimator bias and coverage and of quadrature certificates assess the numerical claim. None establishes model or forecast accuracy.

### NUM-TAIL-GATE

**Control.** Every reported VaR, TVaR and exceedance probability, and every arm effect on them, is gated by batch means over K = 16 batches of years. A statistic converges when its 95% half-width is at most 10% of its value and every batch has at least 10 years in the tail (`MIN_TAIL_EFFECTIVE_UNITS`); otherwise it is `withheld`. So are a view's VaR and TVaR while its simulated years differ from its analytic EAL by more than five standard errors (`EAL_RECONCILIATION_LIMIT`). Each arm doubles its years per state until its driver views converge or it reaches its cap: 2,000,000 state-years for the reference arm and 500,000 for each comparison arm ([methodology](methodology.md#numerical-execution-and-outputs)).

**Rationale.** 16 batches give 15 degrees of freedom, within the 10 to 30 that Schmeiser (1982, *Operations Research* 30(3)) recommends for batch-means intervals on a mean. K, the floor and the limit are `INITIAL_ESTIMATE` numerical judgments, and the 10% tolerance and the caps are judgments too. At the caps, a 15-node study with two variant arms simulates about 4.3 million state-years and holds about 2.2 GB of views.

**Required numerical comparison.** Tolerance 5% and 20%; caps halved and doubled; batches 8 and 32; floor 5 and 20; limit 3 and 8 standard errors.

**Challenge mechanisms.** Not a study alternative: Every gated statistic reports its relative half-width, so availability at another tolerance can be read from the output; other caps need a rerun with the import options. Seeded tests in `tests/test_tail_gate.py` check the half-widths, withholding, doubling, replay and interval coverage.

### NUM-STREAMS

**Control.** Keyed random streams draw initiating counts and every event's marks. Each year's count is drawn by inversion from one uniform addressed by the state, origin and year. Each year's parameter ranks come from a stream keyed by the plan seed, the state's rank stream, the rank group and the year. A variant-study state reads the rank stream of the matching state of the study it varies (`JointState.rank_stream_id`).

**Rationale.** Stable addresses give paired comparisons between arms. A seed gives no cross-platform bitwise guarantee.

**Required numerical comparison.** Tests of origin and roster-order invariance, of inverted counts against 60-digit CDFs, and of rank uniformity and independence across states.

**Challenge mechanisms.** Not a study alternative: Chunk-boundary invariance and pairing with the base are software properties that regression tests check.

### NUM-SMALL-EVENTS

**Control.** An event-addressed origin whose gross loss is one severity draw split by fixed shares sums its small incidents per year. Events below split points are summed by two normal laws with their exact conditional mean and variance when at least `max(n_min, (skewness/γ_max)²)` of them occur. Defaults: κ = 16 individually drawn events targeted per origin-year, γ_max = 0.1, ρ = 2, z_max = 5, n_min = 100 and at most 64 rank buckets ([methodology](methodology.md#numerical-execution-and-outputs)).

**Rationale.** Each state's annual mean and variance are exact, so the analytic EAL is unchanged. The defaults are `INITIAL_ESTIMATE` numerical judgments.

**Required numerical comparison.** γ_max 0.05 and 0.2 (`max_body_skewness`), κ 8 and 32 (`target_large_events_per_year`), ρ 1.5 and 4 (`bucket_rate_ratio`), z_max 4 and 6 (`top_rank_score`) and n_min 50 and 200 (`min_summed_count`).

**Challenge mechanisms.** Not a study alternative: The audit control `small_events.body = individual` draws every small event, and regression tests in `tests/test_small_events.py` compare its annual totals and tails with the summed run. The reported envelope bounds the TVaR difference without simulation.

### NUM-SEVERITY-DOMAIN

**Control.** Evidence is admitted only with monetary amounts zero or between 10⁻⁹ and 10²⁰ currency units. Every severity law and amount component stays in a wider numerical domain: amounts zero or between 10⁻¹⁰⁰ and 10¹⁰⁰, scale factors zero or between 10⁻¹² and 10¹², and GPD shape in [−1, 5].

**Rationale.** Inside the numerical domain every product of an amount, two factors and a tail quantile stays in binary64's normal range.

**Required numerical comparison.** Values outside either domain are refused, never clamped.

**Challenge mechanisms.** Not a study alternative: A numerical admission rule; laws and components outside the domain are refused at their contracts, which regression tests check.

### NUM-EAL-PARAMETER-BAND

**Control.** Beside every welfare, gross and lab-loss EAL that is analytic in every state, results report its parameter band: p5, p50 and p95 of the EAL with each rank group's count law held at one rank for every year. Two cases are reported: independent ranks across rank groups, and one rank for every group. Comonotone quantiles use 131,073 ranks (`RANK_GRID_POINTS`) and independent ones 131,072 draws (`INDEPENDENT_SAMPLES`; [methodology](methodology.md#pooling-and-parameter-uncertainty)). No band is reported for a view in which a state with weight reads a donor's answers (`POOL-UNANSWERED-SCENARIOS`; `JointState.borrowed_leaves`); the view is `unavailable` with its reason. The run captures the band of every view either way, and results apply the rule when they read the capture.

**Rationale.** No other output shows how uncertain the EAL is because the elicited count laws are uncertain. The band is neither a range of annual losses nor a confidence interval. It mixes the respondents as separate sources of the count laws. A respondent that borrows repeats a donor's laws of the borrowed leaf, so the mixture would count one source's laws twice. The spread that separate answers would give is not in the captured answers.

**Required numerical comparison.** Grid and draws halved and doubled (`INITIAL_ESTIMATE` budgets).

**Challenge mechanisms.** Not a study alternative: A report-only exhibit. Regression tests in `tests/test_eal_parameter_band.py` check that a fixed rate collapses it to a point and that replay accepts a recomputation only within its bounds and tolerance.

### NUM-RESPONDENT-SAMPLING

**Control.** Results report a respondent-sampling standard error beside every analytic EAL and every arm's change ([`respondent_sampling.py`](src/catmodel/results/respondent_sampling.py)). Each state belongs to one sampling group, its trajectory node and respondent model, whose kept samples are treated as independent draws. A level's SE² is Σ_g W_g² s_g²/n_g and a paired change's is Σ_g n_g s_g²(y). An absolute z below 2 is marked within noise ([methodology](methodology.md#pooling-and-parameter-uncertainty)). No standard error or z is reported for an EAL or a change in which a state with weight reads a donor's answers (`POOL-UNANSWERED-SCENARIOS`; `JointState.borrowed_leaves`). The groups of a lab-loss view or a gross view hold the samples that are not left out of those views, at their weights in that view (`POOL-EXCLUDED-SAMPLES`; `JointState.lab_loss_left_out`).

**Rationale.** It covers sampling within each model only, not a bias the samples share or the choice of models. With five samples per group the threshold of 2 is a reading aid, not a test. A borrowed answer is in its donor's group and its borrower's, so the groups are not independent and their variances do not add.

**Required numerical comparison.** None: every z is reported, so another threshold can be applied.

**Challenge mechanisms.** Not a study alternative: A report-only exhibit. Regression tests in `tests/test_respondent_sampling.py` check it against a hand calculation and a stratified delete-one jackknife.

### NUM-CAPABILITY-SCENARIOS

**Control.** The public capability results show the no-progress counterfactual and the 10th, 50th and 90th percentiles of the base forecast law of frontier progress. EAL is interpolated linearly in log progress between bracketing nodes, and the annual loss curve is Vincentized in log1p(loss). A percentile outside the collected nodes' range is clamped to the nearest node and flagged.

**Rationale.** A percentile path is a representative conditioning point, not a probability-weighted scenario.

**Required numerical comparison.** Other percentiles; the three shown are an `INITIAL_ESTIMATE` presentation choice.

**Challenge mechanisms.** Not a study alternative: A deterministic report-only exhibit, tested in `tests/test_capability_scenarios.py` against direct sampling of independent lab paths. Other percentiles recompute from the retained base law without new collection or simulation.

### NUM-PLAUSIBILITY

**Control.** Results report each scenario's welfare EAL and the total as a share of world output for the study year: USD 135.093 trillion in 2027, the IMF's projection ([`world_output.yaml`](data/reference/world_output.yaml)). A leaf above 0.1% or a total above 1% is flagged. A contextual summary also gives each leaf's expected annual number of incidents of at least USD 1 billion, and flags a leaf count of at least 0.1 a year or a total of at least 1 (`LARGE_INCIDENT_FLAGS` in [`plausibility.py`](src/catmodel/results/plausibility.py)).

**Rationale.** World output is an order-of-magnitude anchor that no respondent judgment controls. A share is a ratio, not a bound, and the counts are checked against no incident data. The thresholds are coarse, uncalibrated `INITIAL_ESTIMATE` reading aids.

**Required numerical comparison.** The thresholds at half and twice their values.

**Challenge mechanisms.** Not a study alternative: A report-only reading aid. Every share and count is reported, so any other threshold can be applied to the same tables.

### NUM-CALL-LIMITS

**Control.** Every elicitation call asks for at most 100,000 output tokens (`ELICITATION_MAX_TOKENS` in [`config.py`](src/catmodel/config.py)), lowered to the model's recorded ceiling (`max_output_tokens` in its [model profile](data/example_model_profiles.yaml), whose comments give each ceiling's basis). It runs within the profile's request deadline (`request_timeout_seconds`): 2,400 s in the example and test profiles, and 300 s in dev.

**Rationale.** A binding budget selects which answers are accepted: a stop at the output limit is a failed validation attempt, so accepted answers lean toward shorter deliberation. The deadline lets a 100,000-token answer finish at 46 tokens a second.

**Required numerical comparison.** Any output-limit stop, or a generation time near the deadline, calls for a higher limit in a new run directory. `scripts/probe_output_ceilings.py` can check each ceiling before a run.

**Challenge mechanisms.** Not a study alternative: An execution control. `capability-risk validate` reports each model's output-limit stops and generation times; another budget, ceiling or deadline needs a new run directory and fresh answers.

### NUM-CONNECTION-LOST

**Control.** A request whose stream is cut, or whose background run fails with a transient code (`RESENT_FAILURE_CODES`) or is cancelled at its queue limit or deadline, produced no answer. It is sent again, at most twice (`RESENDS` in [`litellm_provider.py`](src/catmodel/llm/litellm_provider.py)), and each re-send is recorded as `connection_lost` ([runbook](runbook.md#3-collection-settings-and-validation)).

**Rationale.** A cut is likelier the longer an answer runs, so answers accepted after a re-send lean toward shorter deliberation: under a constant cut hazard λ per minute, an answer of t minutes survives with probability e^(−λt).

**Required numerical comparison.** With r re-sends over a model's total streamed time T, λ ≈ r/T; state the tilt e^(−λt) beside that model's results. A model whose re-sends exceed 1% of its responses (a judgment) is examined first.

**Challenge mechanisms.** Not a study alternative: An execution control. `capability-risk validate` reports each model's `connection_lost` re-sends; another re-send bound needs a new run directory and fresh answers.

### NUM-PROVIDER-FAILURES

**Control.** A call whose request fails at the provider with an error is frozen as an attempt with no response. A transient failure (`is_transient` in [`transport.py`](src/catmodel/llm/transport.py)) fails only its call's (coordinate, respondent) group; any other failure stops the study. A content-policy rejection of the prompt counts as a refusal (`POLICY_REJECTION_CODES` and `POLICY_REJECTION_PHRASES` in [`litellm_provider.py`](src/catmodel/llm/litellm_provider.py)). Rerunning the command sends each failed call again, up to three provider failures per call (`PROVIDER_FAILURES_PER_CALL` in [`frozen_calls.py`](src/catmodel/frozen_calls.py); [runbook](runbook.md#3-collection-settings-and-validation)).

**Rationale.** The bound of three is an `INITIAL_ESTIMATE` judgment: three transient failures on separate runs point to a request-specific fault.

**Required numerical comparison.** The bound at 1 to 5.

**Challenge mechanisms.** Not a study alternative: An execution control. A re-sent provider failure selects nothing the respondent said, and every failed attempt stays in the capture. Another bound needs only a changed constant.
