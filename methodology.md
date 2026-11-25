# Methodology

This page defines the model's calculation. Terms follow the [glossary](docs/glossary.md), and [limitations](docs/limitations.md) says how far the results can be relied on. IDs such as [`POOL-SPREAD`](assumptions.md#pool-spread) name entries in the [assumption register](assumptions.md), which gives each assumption's basis, evidence status and required sensitivity.

## Model on one page

The main pipeline asks questions about one named lab, model and deployment along the trajectory nodes of a forecast of public-frontier capability on Epoch AI's Epoch Capabilities Index (ECI). It therefore reports an absolute expected annual loss (EAL). The same questions asked at fixed ECI anchors give conditional what-if results.

### What the model computes

For one study year and the scenarios a setup selects, the model computes the distribution of annual loss L in each requested view and its mean, the EAL. The main views are societal harm (`welfare`) and the named lab's economic loss (`lab_loss::<lab_id>`). A loss belongs to the year in which its incident originates, at its eventual amount (origin-year ultimate): a fine matter originates in the year its conduct begins and carries every fine it eventually draws. The [README](README.md#what-a-result-measures) tabulates what a result counts and what it is not.

### The equation

A *joint state* z fixes a capability condition c and one respondent r (one LLM and one sample index). The condition is a trajectory node of the forecast or, in what-if analysis, a fixed ECI score. Each simulated year y of a state reads one uniform rank u_{s,y} per scenario s and source status, with normal score a_{s,y} = Φ⁻¹(u_{s,y}). Ranks are independent across states, scenarios and years and the same in every arm of a state. Given the state and the year's scores:

```text
L        = Σ_s Σ_{i=1..N_s} X_{s,i}           annual loss
N_s      ~ Poisson(Λ_s)                       qualifying incidents of scenario s
Λ_s      = Q_{r,s}(Φ(k_s·a_{s,y}); c)         expected annual count at the year's score
X_{s,i}  = Σ_j X_{s,i,j} ~ F_{r,s}(·; c)      one severity draw, split into components j by elicited shares
lab loss = Σ_s [ Σ_i min(provider-borne_i + η_{r,s}(c)·other_i, lab_cap) + 1{N_s ≥ 1}·min(H_s, lab_cap) ]
H_s      = 0 with probability 1 − p_{r,s}(c), else a draw from a positive law
gross    = Σ_s [ Σ_i X_{s,i} + 1{N_s ≥ 1}·min(H_s, lab_cap) ]
```

- **Counts.** Q_{r,s} is the quantile function of respondent r's final count law for scenario s, a fixed pool of its direct and event-chain answers ([event populations and frequency](#event-populations-and-frequency)). The rank spread k_s is 1 except in the opt-in spread alternatives ([pooling](#pooling-and-parameter-uncertainty)).
- **Capability.** Every answer is conditional on one whole trajectory, so each node has its own laws. Between fixed ECI anchors every elicited quantity is interpolated linearly; scores outside them are unavailable.
- **Severity.** F_{r,s} is a monotone spline through the stated p5–p95 of loss per incident, with a generalized Pareto (GPD) tail above p95. A stated count of incidents of at least USD 1 billion shapes the tail beyond p99, and a scenario cap applies last ([conditional severity](#conditional-severity)).
- **Lab loss.** Components whose authored bearer is the provider fall on the named lab, which also pays a compensation share η of the other components ([`ACC-LAB-BEARER`](assumptions.md#acc-lab-bearer)). H_s is the lab's own business loss, such as lost revenue: one amount per year for all of a scenario's incidents ([`ACC-LAB-ANNUAL-BUSINESS-LOSS`](assumptions.md#acc-lab-annual-business-loss)). `lab_cap` is a stipulated frontier-lab value of USD 1 trillion, or the catastrophe class endpoint where lower ([`SEV-LAB-CAP`](assumptions.md#sev-lab-cap)).
- **Views.** `welfare` gives ordinary transfers, such as fines, weight zero and omits H_s; `gross` counts both ([incident accounting](docs/incident-accounting.md)).
- **Independence.** Given z and the year's ranks, counts and losses are independent across scenarios and incidents. H_s is independent of the year's incident losses and of its count beyond N_s ≥ 1.

The reported distribution mixes the joint states:

```text
Pr(L ≤ x) = Σ_z w_z·Pr(L ≤ x | z),    w_z = ω_d·v_r
```

ω_d is the quadrature weight of the state's trajectory node d ([trajectory quadrature](#trajectory-quadrature)), or 1 at a fixed ECI score, and v_r is respondent r's pooling weight. Because the rank moves only the expected count, the EAL integrates each count over its rank, with Z standard normal:

```text
EAL = Σ_z w_z Σ_s [ Λ̄_s·E[X_s | z] + E_Z[Pr(N_s ≥ 1)]·E[min(H_s, lab_cap)] ],    Λ̄_s = E_Z[Q_{r,s}(Φ(k_s·Z); c)]
```

X_s is the incident's loss in the view, and the H_s term enters the gross and lab views only. The EAL is computed from the laws, in closed form or within a certified integration bound, so it does not depend on the seed. Quantiles and the tail value at risk (TVaR) come from simulated years ([numerical execution](#numerical-execution-and-outputs)), as does the EAL of views without an analytic mean, such as payouts.

### Worked example

**Arithmetic.** The [golden Poisson example](config/setups/golden-poisson/README.md) has one joint state, a constant rate Λ = 0.5 and a loss with E[X] = 28, so EAL = Λ·E[X] = 14, which the engine reports exactly. Its page derives the loss probability, the value at risk (VaR) and the TVaR by hand.

**Capability response.** The numbers below are invented. One respondent states the expected annual number of qualifying incidents for one scenario at two fixed ECI anchors; in the main pipeline the same contrast appears between forecast trajectories.

| ECI | p5 | p25 | p50 | p75 | p95 | Mean over ranks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 140 (reference) | 0.5 | 1 | 2 | 4 | 10 | 3.35 |
| 150 (interpolated) | 0.75 | 1.5 | 3 | 6 | 15 | 5.02 |
| 160 | 1 | 2 | 4 | 8 | 20 | 6.70 |

At ECI 150 each quantile is the average of its two anchor values. A year whose rank is 0.5 reads Λ = 3.0, and one whose rank is 0.9 reads Λ = 12.10 from the monotone cubic (PCHIP) curve between p75 and p95.

The respondent states loss per incident, in USD million, as p5–p95 = 0.1, 0.5, 1, 3, 10 and p99 = 30 at both anchors. The tail class sets the GPD shape ξ = 0.5, with no qualifying floor and no cap. The GPD scale is then σ = 0.5 × 20/(5^0.5 − 1) = 8.09, and the law's mean is E[X] = 3.02. Integrated over ranks, the EAL is 3.35 × 3.02 = 10.1 at ECI 140, 15.2 at ECI 150 and 20.2 at ECI 160 ([`test_methodology_worked_example.py`](tests/test_methodology_worked_example.py)). The doubling from 140 to 160 is this respondent's capability response; no question asks for it as a separate multiplier.

## Purpose and scope

One engine accepts admitted observations, fitted laws, elicited judgments and stipulations (`Evidence.origin`). The hypothetical-lab study's lab is stipulated, and its deployment scales grow with each trajectory's capability progress up to a ceiling ([`POP-HYPOTHETICAL-LAB`](assumptions.md#pop-hypothetical-lab)). Populations a study does not declare are unquantified ([intended purpose](model-intended-purpose.md)).

## Loss taxonomy

The tree splits losses by how they arise: what started it, what the AI did, and what process caused the loss. A leaf is one population of attempts taking one path to loss, so a node is split only where its incidents cannot share an exposure unit, an event chain or severity evidence. A split asks about a fact fixed when the incident begins: never one settled as it unfolds, such as its size, its reach or who bore it, never a condition of the whole study, such as capability or safeguards, and never a named scenario's definition. Each incident is counted once, with all its losses, under the process that caused most of them.

Only a leaf carries an estimate, so a threat that needs its own number needs a leaf. The [hierarchy contract](contracts/scenarios/hierarchy-contract.yaml) quotes the rule in its `rule` field, and its `conventions` apply it:

1. **Incident identity.** One incident is one root cause, campaign, defect or matter: a campaign across many victims, a defect across many deployments or users, or a matter with all its fines and costs. A design or behavior is an identified defect only when a specific, correctable model, version, feature or configuration property is shown or acknowledged to cause the harms; a pleaded theory alone does not merge losses into one incident.
2. **Whole-incident ownership.** The whole incident, with its claims, fines, legal, recovery and remediation costs, goes to the leaf of the process that caused most of its loss. "Most" means the largest share: no majority is needed, and a tie goes to the process whose loss came first. Every split applies this test, with no linked parts across leaves and no allocation weights, so counts and losses add within every parent ([`POP-WHOLE-INCIDENT`](assumptions.md#pop-whole-incident)).
   - An intrusion, fraud or influence operation that is part of a CBRN program belongs to that program.
   - A claim whose defining allegation is loss from a discrete incident, such as a breach, outage or harmful output, follows that incident; only claims over the provider's own conduct are provider-conduct private claims.
   - A claim or enforcement action alleging bodily or psychological injury or death of identified persons from the model's outputs follows the incident that injured them, whatever theory is pleaded. Product and consumer claims, and enforcement, cover matters without such injury incidents.
   - An enforcement action over a practice that cites such injuries from several counted incidents is an enforcement matter, and its fines and costs are part of none of those incidents.
3. **Admission.** An incident is admitted when its gross loss, summed over all affected parties and loss kinds before any recovery, is at least USD 100,000 in policy-start USD. Each amount is converted at the exchange rate of the date it occurs or is imposed and deflated to the policy start by US consumer prices. Incidents below the threshold are a disclosed, unestimated leftover, neither zero nor another leaf's incident. Every leaf uses this one threshold ([`POP-ADMISSION`](assumptions.md#pop-admission)).
4. **Accumulation.** How far one incident spreads is part of its severity, never a split. Whether an attack's code spreads on its own is fixed when the code is released, so it is a split: a self-propagating compromise is its own leaf whatever it reaches. Outcome tests, such as success or reaching a second organization, are never splits; they appear as event-chain stages only where the terminal needs them.
5. **Other and unknown.** An "other" child appears only under an open-ended question. An unknown fact is uncertainty, recorded as a mixture over the possible answers, not "other".
6. **Several AIs.** The AI nearest the largest share of the loss decides "what the AI did".
7. **Help to an attacker.** "Helped the attacker" covers any copy of the provider's model however obtained: served, stolen, leaked or open weights. The theft leaf counts only the theft's own losses; harm from later use of a stolen copy belongs to the leaf where it helped an attacker.
   - The model helped only when its assistance was *material*: it gave the attacker capability, scale or speed that the attacker would not have had without any AI model, and the loss, or more than half of it, would not have occurred without that gain.
   - The comparison is with what the same attacker would in fact have had if no generative AI model of any developer, voice and image generators included, had been available to it: its own skills and staff, software in ordinary use before such models, and whatever it would have bought or hired instead. Help that another AI model could have given is therefore still material.
   - An attack that used the model without material assistance is no incident of this branch and, unless it meets another leaf's own conditions, no leaf's incident, other origin included. A claim, enforcement action or designation over it is then a provider-conduct matter, even where it alleges injury or death of identified persons.
8. **Other origin.** A complete history with no named origin is other origin; its losses are unsupported, never assigned to a named leaf.

### Adding or changing a split

A split is an internal node whose children answer one classification question on one declared axis of the contract. To add or change one, author its question, axis and `split_rationale` there, and answer four review checks:

1. Is the answer fixed when the incident begins? If not, it belongs to severity, an event-chain stage or a study variant.
2. What can the sides not share: an exposure unit, an event chain or severity evidence? The answer is the `split_rationale`.
3. Would the distinction exist if no scenario had ever been written?
4. Would estimating the sides separately move a result a reader uses? If not, the difference stays an attribute of the history.

Validation requires a `split_rationale` on every split and refuses a split on a loss threshold or an undeclared axis. Review the change against concrete histories, add resolved cases to `tests/fixtures/taxonomy_boundary_cases.yaml` and bump `hierarchy_version`. A changed split changes leaf populations, whose answers must be collected again ([adding a scenario](docs/adding-a-scenario.md)).

### Tree

The contract has <!-- scenario-count:splits -->14<!-- /scenario-count --> splits and <!-- scenario-count:leaves -->37<!-- /scenario-count --> leaves: <!-- scenario-count:leaves.estimated -->32<!-- /scenario-count --> estimated, each by one scenario, <!-- scenario-count:leaves.unestimated -->4<!-- /scenario-count --> unestimated "other" leaves under open-ended questions, and other origin, which is unsupported. Split questions are in bold, and the [taxonomy reference](docs/taxonomy.md) names each leaf's scenario.

- **What started it?** Deliberate misuse, AI failure, provider conduct, or other origin (unsupported).
  - Deliberate misuse: **What did the AI do?** Helped the attacker; made to act for the attacker; stolen or attacked as an asset.
    - Helped the attacker: **What process caused the loss?** System compromise; fraud; influence; CBRN weapon; other process (unestimated).
      - System compromise: **Whose systems did the campaign attack?** Critical-infrastructure operators'; other organizations'; individuals'; any it could reach, by spreading on its own.
      - Fraud: **Whose value did the fraud go after?** Individuals'; organizations'.
      - CBRN weapon: **Which hazard?** Biological; chemical; radiological or nuclear.
        - Biological: **Is it an expert program?** Expert, including state; novice.
    - Made to act for the attacker: **How was it made to act?** Hijacked through its inputs; tampered model, artifact or configuration.
    - Stolen or attacked as an asset: **Were its artifacts taken or its service attacked?** Artifacts stolen or leaked; service attacked.
  - AI failure: **What did the AI do?** Acted on its own for a deployer; ran a bounded decision task; controlled physical equipment; output relied on by a person; harmed a person who engaged with it; service failed; acted against its developer's control; other AI failure (unestimated).
    - Output relied on by a person: **Which field?** Health; legal; financial; software; other field.
    - Service failed: **Which failure?** Outage or degradation; other service failure (unestimated).
  - Provider conduct: **Which process?** Government designation or restriction; enforcement; private claims; other process (unestimated).
    - Private claims: **What is alleged?** Copyright and IP; privacy and data protection; product and consumer harm; other allegations.

### Counted units

Each scenario names its counted unit and year rule, and every leaf counts incidents directly (`qualified_terminal`). The unit is one misuse campaign or weapons program across everything it reaches; one root cause of an AI failure across every deployment or user it reaches; one loss-of-control history of one model, with every copy it makes; or one designation, enforcement or private matter. A private matter is a class or representative action, a consolidated or coordinated proceeding, or a coordinated mass arbitration ([`POP-PRIVATE-CLAIMS`](assumptions.md#pop-private-claims)).

A unit counts when it meets its scenario's conditions, whether or not anyone discovers it, reports it or traces it to the provider's model, so incident databases and court records are lower bounds. An incident involving several providers' models counts as one whole incident for each provider whose model meets the conditions ([`ACC-ATTRIBUTION`](assumptions.md#acc-attribution)).

All eleven scenarios in which a model helps an attacker, seven outside CBRN and the four CBRN scenarios, state the material-assistance test of convention 7. In their event chains a campaign or program that uses the model's help is an opportunity, and the first stage (`uplift`) is the gain. That the loss, or more than half of it, depended on the gain is part of the last stage. A counted incident's loss is never apportioned to an AI-caused share.

An incident is dated by its first loss, and an enforcement matter by the first occurrence of the conduct it concerns; a practice extended to new models keeps its original start year.

### Scenario contributions

`scenario_contributions` reports each scenario's EAL in one additive measure, chosen from the per-scenario views with societal harm first, and its share of the pooled EAL. An `unallocated` row gives the pooled EAL less the scenario sum. `scenario_groups` adds scenario EALs up by reporting domain and by each facet on which every scenario has one value. A group holds whole scenarios, so it cannot show a threat that is part of a scenario or spread across several, such as ransomware.

## Evidence, setup and execution

A setup (`RunSetup`) freezes the scope, the evidence, the assumptions, the resolved conditional laws and an execution plan. A run simulates it into an immutable `RunCapture`, which results requests reduce ([setup, run and results](docs/setup-run-results.md)). `catmodel setup import` replays the original prompts and every raw attempt under the captured model, settings, parser and retry protocol.

Every question is rendered from its frozen instrument ([`question_text.py`](src/catmodel/elicitation/question_text.py); [examples](docs/example-questions/README.md)). In order, it states how to answer; the lab, model and deployment; the capability condition; the scenario or category; its evidence; the questions; and the answer schema, which lists reasoning first and quantiles median first. It does not say how code processes the answers, such as tail shapes, caps and pooling.

- **Capability condition.** A trajectory node is a dated table of each lab's general ECI. It names the models whose fixed scores define the scale, opens with each lab's observed score on the forecast's as-of date and gives the labs' highest score one year earlier. Where the forecast records what the index's publisher reports a gain in points to have corresponded to, the question quotes it and asks the respondent to include the uncertainty about whether it holds.
- **Evidence.** A scenario's evidence is the dated source cards its entry in the source bundle lists ([runbook](runbook.md#1-evidence-and-scope)). Each card states only what its cited sources report and holds no authored prior, default or answering instruction. In the hypothetical-lab study's [cards](data/elicitation_source_cards.yaml), most scenarios have a frequency in a reference class: a count, share or rate that a source states for a population or exposure. The step from that class to the lab's count is the respondent's judgment ([limitation 1](docs/limitations.md#1-the-risk-judgments-come-from-llms-and-cannot-be-backtested-in-general)).
- **Threat model.** Where the bundle's register holds one, a question also quotes the scenario's [threat model](docs/threat-models.md), which states no probability or loss estimate.

## Observable capability conditioning

A lab's *records* are its releases that strictly exceed every earlier release of that lab on one frozen general-ECI scale; its as-of frontier `F0_l` is its last record. Records in the authored trend window enter one pooled Gaussian likelihood once each:

```text
score_i = A_l + B·(t_i − as_of) + ε_i,    ε_i ~ N(0, τ² + σ_i²), independent
```

`A_l` is lab l's level (flat prior, integrated out), `B` one trend rate shared by the labs (diffuse prior, conditioned positive), `τ` the trend discrepancy and `σ_i` release i's benchmark error. Unless authored, `τ` is its pooled restricted-maximum-likelihood estimate. On Epoch AI's ECI table captured 2026-10-29, with a 24-month window (44 records), that estimate is 1.64 ECI points (90% profile interval 1.12 to 2.27) and the rate 16.2 ± 0.6 points a year ([`test_public_frontier_epoch.py`](tests/test_public_frontier_epoch.py)).

Each lab's forecast path is linear from its observed frontier:

```text
F_l(t) = F0_l + B·M_l·(t − as_of)
```

`B` is drawn once from the positive rate posterior, and each lab's pace `M_l` independently from a median-one log-logistic law truncated at the pace cap:

```text
G(m) = [m^pace_shape/(1 + m^pace_shape)] / [pace_cap^pace_shape/(1 + pace_cap^pace_shape)],    0 < m ≤ pace_cap
```

The public frontier is the pointwise maximum of the lab paths, so its leader can change ([capability forecasting](docs/capability-frontiers.md); [`OBS-PUBLIC-JOINT`](assumptions.md#obs-public-joint)). Both pace defaults are judgments: `pace_shape` is 5, and `pace_cap` is 4, about twice the fastest lab-year in the table. An untruncated pace has a power-law tail; under a cap every moment of the pace is finite. The forecast comparison repeats the baseline at pace shapes 3 and 8, at a pace cap of 2 where the forecast's own cap is higher, and at the ends of `τ`'s 90% interval. A cap above the forecast's needs a study planned and collected under it.

### Trajectory quadrature

The study integrates the forecast over the frontier progress `P`, the public frontier's gain over its as-of level at the middle of the policy period, by a fixed rule, not by sampled paths ([`public_frontier_quadrature.py`](src/catmodel/forecast/public_frontier_quadrature.py)). With `Δ` the years from as-of to the period's middle and `F0_max` the highest as-of frontier, `P ≤ p` exactly when every lab's pace is at most its threshold `m_l`:

```text
m_l = (p + F0_max − F0_l)/(B·Δ),    Pr(P ≤ p) = E_B[ Π_l G(m_l) ]
```

The covering law, with density ψ and distribution function Ψ, is an equal mixture of the base law and the widest pace alternative (shape 3), both under the forecast's cap. With `x_d` and `ν_d` the nodes and weights of Fejér's second rule on (0, 1), the `n` interior nodes (odd, default 7) and a forecast law's normalized node weights are

```text
p_d = Ψ⁻¹(Φ(1.5·Φ⁻¹(x_d))),    ω_d ∝ ν_d·J_d·f(p_d)/ψ(p_d)
```

where `f` is the law's density of `P` and `J_d` the Jacobian of the stretched scale. Each interior node carries one representative joint path drawn from the base law given its progress: the rate, the leading lab at its threshold pace, and every other lab's pace from its own law below its threshold. A no-progress node holds every lab at its as-of frontier and carries the no-progress law's whole weight.

The base law and the forecast alternatives are integrated on the same collected nodes, so every forecast comparison is exactly paired. On the committed forecast, seven nodes recover the shape-3 alternative's effect on the mean of `P` and of `P²`, +23% and +61%, and give the lower cap's as −6% and −16% against exact −8% and −19% ([`test_public_frontier_epoch.py`](tests/test_public_frontier_epoch.py)). Any other feature of a path, such as which lab leads, enters through one draw per node ([limitation 4](docs/limitations.md#4-the-main-pipeline-challenges-only-some-of-its-assumptions)).

Fejér's rule with `n` nodes contains the rule with `(n − 1)/2` nodes, and the sensitivity summary reports each integrated arm's EAL under both. The difference indicates the coarser rule's error and bounds neither rule's. Its indicator is `uninformative` when one node carries more than 95% of the embedded rule's weight, and `exact` for the no-progress point mass.

No capability response is extrapolated without a bound. A fixed-anchor instrument's anchors must cover its integer range, and authored `LocalResponse` curves with `extrapolation: last_slope` continue the top slope for one top-anchor spacing and are flat beyond (`LAST_SLOPE_CAP_SPACINGS`).

## Event populations and frequency

In authored setups (`catmodel setup build`) the count is conditional Poisson with cluster, member and component marks:

```text
Λ = λ·Σ_j e_j·q_j·(1/|T_j|) ∫_{T_j} response(c(t)) dt
```

with reference rate λ, exposure segments `T_j` with shares `e_j` and initiation gates `q_j`, and a capability response curve. In context-bound setups all but λ are one: respondents state already-qualifying counts.

**Direct count.** Every scenario's frequency is the expected annual count of its terminal incidents ([loss taxonomy](#loss-taxonomy)). The direct question asks for p5–p95 of the respondent's uncertainty about it. A missing law is never read as zero.

**Event chain.** Every scenario also authors an `event_chain`, a bottom-up route to the same count, asked in a separate call that does not see the direct answer ([`FREQ-EVENT-CHAIN`](assumptions.md#freq-event-chain)). It has three parts:

- an *opportunity*, with the rule that counts opportunities and the year's risk set;
- an *exposure base*: a stipulated count of a natural unit, fixed or tied to one of the study's deployment scales;
- one to four authored *stages*, each conditional on every earlier one, followed by the scenario's qualified terminal unit as the last stage.

Each opportunity yields at most one terminal unit, so by the chain rule, with no independence between stages assumed:

```text
expected count = base count × opportunity rate × Π stage probabilities
```

The call asks for p5–p95 of the opportunity rate per base unit and of each stage's conditional probability. An answer whose rate has p95/p5 above 1,000, or whose terminal stage has p50 below 10⁻³, gets one retry that names no threshold or direction ([`FREQ-CHAIN-GUARDRAIL`](assumptions.md#freq-chain-guardrail)). The chain count law combines the rate and stage laws as independent quantities in log space, each read only between its own p5 and p95 ([`FREQ-CHAIN-INDEPENDENCE`](assumptions.md#freq-chain-independence)).

**Fixed pool.** The leaf's count law pools the two estimates by a rule set before any answer is seen ([`FREQ-CHAIN-POOL`](assumptions.md#freq-chain-pool)). At each of p5, p25, p50, p75 and p95:

```text
pooled = exp(0.5·ln direct + 0.5·ln chain)
```

A quantile at which either is zero pools to zero. A wholly zero direct answer uses the chain alone, and an unavailable or wholly zero chain the direct count alone. Both large-loss counts are scaled by the ratio of the pooled mean to the direct mean, which keeps the direct answer's large-loss share.

**EAL cross-check.** The direct question also asks for a *direct EAL*: p5–p95 of the respondent's uncertainty about the node's expected annual societal harm ([`FREQ-EAL-CROSSCHECK`](assumptions.md#freq-eal-crosscheck)). Before pooling, its mean is compared with the *implied EAL*: the direct count's mean times the law's analytic welfare mean per incident. When they differ by more than a factor of three, one more call shows the respondent both, and the respondent may revise the count, the severity, the large-loss count or the direct EAL. Revised answers drive the run even when the two still differ. The direct EAL is never simulated.

**Taxonomy parents.** A complete-catalog instrument also asks every *parent*, a split whose subtree holds an estimated scenario, for its whole category's expected annual societal harm, in a call that never sees the children's answers ([`HIER-CATEGORY-POOL`](assumptions.md#hier-category-pool)). A parent with modeled children is also asked for its *remainder*: its unestimated child leaves (for the root, other origin), child splits with no estimated scenario and child nodes that cannot be collected.

The answer and what is beneath it are two estimates of the category's EAL, compared as means. `D` is the mean of the parent's category law. `M` is the sum of its modeled children's mean EALs plus its remainder's mean. A leaf's mean EAL is its pooled count's mean times its law's analytic welfare mean per incident; a parent child's is its own pooled value. The height `h` is 1 where the modeled children are all leaves, otherwise one more than the tallest parent among them. Parents are pooled bottom-up:

```text
w_h = 1/(h + 1),    g = (D/M)^w_h,    pooled value = M·g = M^(1 − w_h)·D^w_h
```

Every parent whose two sides are identified and positive is `pooled`, whatever its gap. Otherwise its factor is one: `not_pooled` (a side is zero), `parent_exhausted` or `comparison_unavailable` (a side is not identified). With this weight, the leaves' sum and every category answer above it weigh equally along the tallest branch. The weight therefore depends on how the taxonomy groups its leaves. Leaves that sum to 100 under a category answer of 900 pool to 300. With one more level between them, whose own answer is 100, the category has height 2 and the pool is 208.

`g` multiplies the count law and both large-loss counts of every leaf beneath the parent, so severity is unchanged and the leaf's welfare EAL moves in proportion; allocating the whole gap to the count is a convention. Factors compound down the tree, and a leaf's cumulative factor is clamped to the range 0.0001 to 10,000. Remainders are scaled too. They are reported as unsimulated lines and never enter a simulated total ([`HIER-REMAINDER`](assumptions.md#hier-remainder)).

**Frequency-estimate exhibit.** Every context-bound result reports the EAL under alternative counts (`frequency_estimate`): the direct count or the chain alone (`direct`, `chain`), the pool with one follow-up call undone (`before_severity_retry`, `before_chain_guardrail`, `before_eal_crosscheck`), and the two ends of the category answer's weight (`before_hierarchy`, `hierarchy_parents_only`).

## Pooling and parameter uncertainty

**Weights.** Each respondent model has equal mass, split equally across its samples ([`POOL-WEIGHTS`](assumptions.md#pool-weights)).

**Exhausted calls.** A call is exhausted after three failed validation attempts; a refusal or an output-limit stop counts as one. An exhausted direct question leaves its leaf unanswered. Any other exhausted call leaves the answers it would have revised, or its parent unpooled.

**Unanswered leaves.** Import decides over the whole study what an unanswered leaf leaves out ([`excluded_samples.py`](src/catmodel/setup/excluded_samples.py), tested in [`test_excluded_samples.py`](tests/test_excluded_samples.py) and [`test_unanswered_scenarios.py`](tests/test_unanswered_scenarios.py)). A *sample* is one respondent at one trajectory node.

- Where more than 10% of a model's samples lack one leaf (`max_excluded_sample_share`), the model does not answer that leaf, and all its answers to it are set aside ([`POOL-UNANSWERED-SCENARIOS`](assumptions.md#pool-unanswered-scenarios)). Each of its samples takes, for that leaf alone, the whole case of a sample of an answering model at the same node. Donors follow a fixed rotation with no randomness, under which answering samples lend about equally often. The leaf is then pooled over the models that answer it.
- A sample that lacks any other leaf is excluded, and its model's mass is split among its remaining samples at that node ([`POOL-EXCLUDED-SAMPLES`](assumptions.md#pool-excluded-samples)). No answer is imputed.
- A direct answer may declare the lab's own loss unavailable: its compensation share or its annual business loss. Such an answer excludes no sample and changes no weight, unanswered leaf or donor.

A view that counts the lab's own loss is computed on the samples that give it. Those views are the lab-loss views (`lab_loss::<lab>` and each source status's) and the gross views (`gross` and `scenario::<id>::gross`). Gross is among them because it adds each node's annual lab business loss ([`ACC-LAB-ANNUAL-BUSINESS-LOSS`](assumptions.md#acc-lab-annual-business-loss)): a state whose answer declares the lab's loss unavailable has no law for that term, so its gross loss would omit it. A sample whose states read such an answer, as its own or a donor's, is left out of both families. There it weighs zero, and in every arm its sampling group, its model at its node, keeps its weight, which the group's other samples share in proportion to their weights.

```text
w_lab(s) = w(s) · W_g / (W_g − L_g)    for a sample s of group g that is not left out
w_lab(s) = 0                           for a sample that is left out
```

W_g is the arm's weight on group g and L_g its weight on the group's left-out samples. Both families use these weights, so the source statuses' lab-loss EALs add up to the lab's and the scenarios' gross EALs to gross. An arm in which some group has weight only on left-out samples reports neither family. Welfare, net resource cost and their scenario and source-status views count no part of the lab's loss: they keep every sample at its weight and are the same whether or not the lab's loss was given.

Import refuses a study in which no model answers a leaf, a model does not answer more than half the leaves, a model keeps fewer than half its samples at some node, or a model's or the study's excluded share exceeds 10%. A lab loss declared unavailable refuses nothing. The last three limits bound the excluded and left-out samples together: where leaving samples out would breach one, none is left out, their states keep their weight without a lab-loss law, and the run reports neither the lab-loss views nor the gross views.

**Count law.** A respondent's count law (`pchip_local_tails`, [`POOL-PCHIP`](assumptions.md#pool-pchip)) is a five-knot PCHIP on the original scale through p5, p25, p50, p75 and p95 of the expected annual count. Beyond p95 the quantile is

```text
Q(u) = q95·exp(width·[Φ⁻¹(u) − Φ⁻¹(0.95)]),    width = min(secant, Φ⁻¹(0.95))
```

where the secant is the log difference of p75 and p95 over their normal-score distance. Below p5 the same form uses p5 and p25. A tied edge pair or a zero anchor gives a flat tail. The cap Φ⁻¹(0.95) ≈ 1.645 is the largest width at which the upper tail's contribution to the mean peaks at p95 rather than at an unelicited rank. The tails are a judged extrapolation: five quantiles do not identify behavior beyond p5 and p95.

**Face value and alternatives.** Every setup reads count laws at face value: rank spread `k = 1` and level bias `b = 1`. Three opt-in alternatives are fixed, not set from the [calibration fit](#respondent-spread-calibration):

- rank spread `k = 1.5` and `0.75` ([`POOL-SPREAD`](assumptions.md#pool-spread)): each law is read at `Q(Φ(k·a))`, which keeps its median, with the tail width capped at `width·k ≤ Φ⁻¹(0.95)` after the stretch (`calibrated_p95_peak`);
- tail-width factors 0.75 and 1.5 on the capped widths (`epistemic_tail_scale`);
- frequency-bias factors ×0.5 and ×2 on every count quantile ([`POOL-COMMON-BIAS`](assumptions.md#pool-common-bias)).

For a log-symmetric count law, `k = 1.5` multiplies the count mean by 1.7 at p95/p5 = 20 and by 2.2 at 160 to 400; `k = 2` multiplies it by 2.8 and by 3.3 to 3.4 ([`test_rank_spread.py`](tests/test_rank_spread.py)).

**What the tail statistics estimate.** VaR, TVaR and 1-in-N losses are statistics of the one-year predictive law: the weighted mixture, over states and every count's rank, of the conditional annual-loss law. A 1-in-N loss is exceeded with probability 1/N in a single year; it is not the loss exceeded once in N years under a persistent but unknown rate. Their half-widths measure Monte Carlo error, not parameter uncertainty.

**EAL parameter band.** `eal_parameter_band` shows how uncertain the EAL is because the count laws, and the respondents who stated them, are uncertain. It holds each scenario's normal score fixed for every year. With Λ_s(a) = Q_{r,s}(Φ(k_s·a); c):

```text
EAL_{z,s}(a) = Λ_s(a)·E[X_s | z] + Pr(N_s ≥ 1 | Λ_s(a))·E[min(H_s, lab_cap)]
EAL_{r,s}(a) = Σ_{z∈r} (w_z/W_r)·EAL_{z,s}(a),    W_r = Σ_{z∈r} w_z
band         = distribution of Σ_s EAL_{R,s}(Z_s)
```

R is a respondent drawn with probability W_r, and the Z_s are standard normal. Two cases are reported: independent Z_s, the model's own assumption, and comonotone Z_s, which gives the largest sum in convex order (Dhaene et al., 2002, *Insurance: Mathematics and Economics* 31). The band is a distribution of the EAL, not of annual losses ([limitation 3](docs/limitations.md#3-tail-shapes-caps-and-spread-factors-are-judgments)).

The draw of R treats the respondents as separate sources of the count laws. A state whose model does not answer a leaf reads a donor's answers to it ([`POOL-UNANSWERED-SCENARIOS`](assumptions.md#pool-unanswered-scenarios)), so its respondent repeats another respondent's laws of that leaf. Two respondents at weight 0.5 whose own answers give EALs of 1 and 3 mix to a distribution on two points, 1 and 3; if the second reads the first's answers, both hold 1 and the mixture is a single point. The band counts the repeated laws as a second source's, so it does not show the spread that separate answers would give. It is not recomputed over the answering respondents alone: where the study asks category questions, a borrowing state's laws depend on its own category answers and on the donor's counts, so no regrouping of the states gives the spread of separate sources. No band is reported for a view in which a borrowing state has weight; the result states the reason. The run still captures the band of every view, so a capture replays whether or not its results report it.

**Respondent-sampling standard error.** `respondent_sampling` reports how far an EAL could move if the respondents were sampled again. A sampling group is one model's kept samples at one trajectory node, treated as n independent, equally weighted draws. With W a group's weight, ē the mean of its state EALs and var their unbiased sample variance:

```text
EAL = Σ_groups W·ē,    SE² = Σ_groups W²·var/n
```

A change between an arm and its control is paired on the same samples: with δ a sample's weighted EAL in the arm less that in the control, SE² = Σ_groups n·var(δ). Results mark a change within two standard errors as within respondent-sampling noise. The standard error covers sampling within each model only.

A state whose model does not answer a leaf reads a donor's answers to it, so those answers are in two groups and the groups are not independent. The sum over groups then understates the variance: two donor states with EALs 1 and 3 and two borrowing states that copy them, each at weight 0.25, hold two independent answers, whose EAL has standard error 1.0, and the sum gives 0.71. No standard error or z is reported for an EAL or a change in which a borrowing state has weight, in the arm or in its control; the result states the reason. The [trajectory quadrature](#trajectory-quadrature) indicator's `state_eal_sd`, the unbiased standard deviation of one node's state EALs, counts the same samples as independent. It is not reported for a node where one of them borrows, and `state_eal_sd_reason` says why.

The groups of a view that counts the lab's loss, a lab-loss view or a gross view, hold the samples that are not left out of those views, at their weights in that view, so its standard error is that of the samples it holds. The exhibit's counts of groups and samples are the reference arm's.

### Respondent spread calibration

The calibration is a diagnostic of how wide and how biased the respondents' one-shot answers to known-count questions were in one period. No setup reads its fit ([limitation 1](docs/limitations.md#1-the-risk-judgments-come-from-llms-and-cannot-be-backtested-in-general)).

Seeds ask the direct expected-count question about counts with known outcomes. Primary seeds have windows that start after the latest training cutoff among the roster's models. For primary seed i with resolved count x_i, each kept answer gives a five-quantile law Q_{r,i}, weighted v_{r,i} with equal mass per model. The realized count is negative binomial (NB2) around its expected count μ, with variance `μ + phi_i·μ²` and `phi_i` estimated from the seed's pre-window history. The fit maximizes the pooled likelihood over a spread k and a common level bias b ([`respondent_spread.py`](src/catmodel/calibration/respondent_spread.py)):

```text
ℓ_i(k, b) = Σ_r v_{r,i}·E_Z[ NB2(x_i | b·Q_{r,i}(Φ(k·Z)), phi_i) ]
(k̂, b̂)   = argmax Σ_i log ℓ_i(k, b),    0.5 ≤ k ≤ 4,    0.1 ≤ b ≤ 10
```

k̂ is the spread around the bias-corrected level, so it is read with b̂. The fit is accepted only if the optimum is interior on both axes, the leave-one-block-out gain of the fitted k over k = 1 is positive, at least two blocks of related seeds are available and no primary seed has zero likelihood at every coarse grid cell. The [seed package](data/calibration/respondent_spread/README.md) describes the seeds and their weaknesses, and the [runbook](runbook.md#respondent-spread-calibration) gives the commands.

## Conditional severity

**Body.** A five-quantile law's body is a monotone cubic (PCHIP) spline through p5, p25, p50, p75 and p95 of the log excess over the qualifying floor against the normal score Φ⁻¹(u) ([`test_quantile_spline_body.py`](tests/test_quantile_spline_body.py)). Equal knots form atoms.

**Tail.** A GPD tail of shape ξ starts at p95, with the scale that reproduces the stated p99:

```text
σ = ξ·(p99 − p95)/(5^ξ − 1)
```

The scale also sets the body's slope at p95, so a larger tail scale lowers the quantiles between p75 and p95. An uncapped shape of at least 1 has no finite mean, and one of at least 0.5 has infinite variance.

**Tail classes.** ξ is never a respondent answer. Each scenario adopts one of three tail classes ([`severity_governance_policies.yaml`](scenarios/severity_governance_policies.yaml); [`SEV-GPD-SHAPE`](assumptions.md#sev-gpd-shape)):

- `liability_enforcement`, ξ = 0.5: enforcement and private civil matters and business loss from a designation, bounded in practice by the claimants' economic interest, a statutory maximum or one firm's balance sheet;
- `cyber_operational`, ξ = 0.7: compromise, fraud, influence, hijack, tampering and asset-attack campaigns and operational failures, whose loss can propagate across many organizations, systems, users or markets;
- `mass_casualty_catastrophe`, ξ = 0.8: CBRN, loss of developer control and critical-infrastructure compromise, whose loss is limited only by the size of the affected population and economy.

The output-reliance leaves and bounded decision errors take 0.7 because their largest incidents are defects relied on at many deployments or by many users. The tail-shape stress moves every shape 0.2 lower and higher, keeping the knots and p99. Stated p999 values are diagnostics.

**Caps.** Every catastrophe-class cell caps one episode at USD 100 trillion, about one year of world output, with USD 10 trillion, USD 500 trillion and uncapped stresses ([`SEV-CAP`](assumptions.md#sev-cap)). The other classes are uncapped apart from authored endpoint stresses. A cap is a valuation convention, not a physical bound, and it bounds one episode only, so annual losses remain unbounded.

**Large-loss anchor.** The direct question also asks two expected annual counts, each as a five-quantile law ([`SEV-LARGE-LOSS-ANCHOR`](assumptions.md#sev-large-loss-anchor)): admitted incidents of at least A = USD 1 billion (`large_loss_count`, the anchor) and of at least A₂ = USD 100 billion (`very_large_loss_count`, which enters no reference law). The anchor's share is the probability that an admitted incident reaches A, held fixed across count ranks:

```text
π = E[large_loss_count]/E[count]
```

Let x₀ be the admission floor and S₁ the survival function of the law without the anchor, with 0.99 quantile x₉₉. Where A > x₉₉ and 0 < π < 0.01, the anchor replaces the tail above x₉₉:

```text
S(x) = 0.01·((x − x₀)/(x₉₉ − x₀))^(−ζ),    ζ = ln(0.01/π)/ln((A − x₀)/(x₉₉ − x₀))     for x₉₉ ≤ x ≤ A
S(x) = π·S₁(x)/S₁(A)                                                                    for x > A
```

So S(A) = π exactly, and the law at or below p99 is unchanged. Above A the law is a GPD of the governed shape with scale σ_A = σ + ξ(A − p95): the anchor changes how many incidents reach A, not how losses beyond A are distributed ([`test_large_loss_anchor_law.py`](tests/test_large_loss_anchor_law.py)). Where A ≤ x₉₉ the stated quantiles govern and the anchor is a diagnostic.

A share contradicts the quantiles when it implies a different tail band from the one A occupies: A ≤ p95 needs π ≥ 0.05, p95 < A ≤ p99 needs 0.01 ≤ π < 0.05, and A > p99 needs π < 0.01. A contradicting answer gets one retry that is neutral about which judgment should change. If that does not resolve it, the original answers stand and the law uses the p99-only tail (`severity_unresolved`).

**Two-anchor tail.** The reference law takes no answer beyond A. The two-anchor tail, a comparison and never the reference, takes π₂ = E[`very_large_loss_count`]/E[count] as a second point. Where 0 < π₂ < π, a second power law through (A, π) and (A₂, π₂) replaces the reference law between the levels, and the governed GPD continues above A₂ with mass π₂ ([`test_two_anchor_tail.py`](tests/test_two_anchor_tail.py)).

## Loss ownership and financial views

[Incident accounting](docs/incident-accounting.md) defines the loss views, the lab-loss rule and financial routing. Financial totals sum contract exposures without coordination, so they can exceed unique gross. A contract responds to the losses the origin-year ultimate basis assigns to the study year, and its retentions and limits apply without claim inflation.

## Dependence and capability mixture

Authored setups have three count-dependence experiments: a negative binomial count with a fixed variance-to-mean ratio, a shared gamma intensity factor and a Gaussian fixed-marginal count copula, the last two with matched controls. Joint states may have sparse support, so trajectories, respondent judgments and physical factors can be dependent; product weights are an assumption.

## Numerical execution and outputs

VaR, TVaR and occurrence (OEP) statistics are reported as illustrative. They depend more than the EAL does on the [tail conventions](docs/limitations.md#3-tail-shapes-caps-and-spread-factors-are-judgments) and on [conditional independence between scenarios](docs/limitations.md#5-scenarios-are-conditionally-independent).

**Streams and reproducibility.** Every random stream is keyed by the plan seed, the state and the year, so a state's draws do not depend on which process simulates it or on how many years are simulated ([`keyed_streams.py`](src/catmodel/probability/keyed_streams.py)). Counts are drawn by inversion from the year's rank, so an arm's count is comonotone with its control's. On one platform, results are identical whatever the number of workers. Across CPUs and BLAS builds a result reproduces when its numbers agree to a relative 10⁻⁹.

**Summed small events.** Where an origin's loss is one severity draw split by fixed shares, its small incidents are summed per year by normal laws with their exact conditional mean and variance. Larger incidents are drawn individually ([`NUM-SMALL-EVENTS`](assumptions.md#num-small-events); [`small_events.py`](src/catmodel/simulation/small_events.py)). The EAL is unaffected, and results report a bound on the effect on each TVaR.

**Analytic EAL.** Linear views integrate in closed form. A lab-loss view with a binding cap and a view with an annual business loss use certified numerical integrals, whose bounds differ by at most 0.1% of the mean ([`incident_moments.py`](src/catmodel/simulation/incident_moments.py); [`test_aggregate_lab_loss.py`](tests/test_aggregate_lab_loss.py)).

**Tail convergence gate.** The gate measures precision by batch means (Asmussen and Glynn, 2007, *Stochastic Simulation*, III.5; [`NUM-TAIL-GATE`](assumptions.md#num-tail-gate); [`tail_gate.py`](src/catmodel/simulation/tail_gate.py)). Each state's years are split into 16 batches, and a statistic's 95% half-width is t₁₅,₀.₉₇₅·sd/√16 over its batch values. A VaR or TVaR converges when its half-width is at most 10% of its value and every batch has at least 10 years at or above the VaR; exceedance probabilities are gated likewise. An unconverged statistic is `withheld`: its point value is kept for reference and must not be quoted as an estimate.

Execution doubles an arm's years while its welfare, gross or lab-loss views fail, up to 2,000,000 state-years (required states times years per state) for the reference arm and 500,000 for each comparison arm. An arm's effect on a gated statistic converges when the half-width of its batch-by-batch differences, over the two arms' common years, is at most 10% of the reference value. Half-widths treat the states and their weights as exact.

**VaR and TVaR.** VaR at level α is the least loss whose cumulative weight among the simulated years of the mixed law reaches α. TVaR is v + E[(L − v)₊]/(1 − α) at v = VaR (Acerbi and Tasche, 2002, *Journal of Banking and Finance* 26). Its sample estimate is usually low for heavy tails, so where every state of a view has an analytic mean the simulated mean serves as a control variate:

```text
TVaR = sample tail mean + β·(EAL − simulated mean)/(1 − α),    floored at VaR
```

β is 1 when a state's law has no finite second moment, which covers every uncapped tail class, and otherwise the regression coefficient Cov((L − v)₊, L)/Var(L), clipped to [0, 1]. Views without an analytic mean use β = 0 ([`capability_mixture.py`](src/catmodel/simulation/capability_mixture.py); [`test_tail_precision.py`](tests/test_tail_precision.py)).

**EAL reconciliation.** The simulated years and the analytic EAL come from separate implementations. When a view's simulated mean differs from its EAL by more than five batch-means standard errors, its VaR and TVaR are withheld (`eal_reconciliation` in [`tail_gate.py`](src/catmodel/simulation/tail_gate.py); [`test_tail_gate.py`](tests/test_tail_gate.py)).

## Sensitivity, validation and communication

Import freezes every family of the catalog ([`sensitivity_catalog.py`](src/catmodel/setup/sensitivity_catalog.py)) as a comparison requirement that is `required`, `unavailable` or `inapplicable`, with the register challenges no context-bound setup executes. Results therefore disclose what was not challenged. Fixed-anchor import executes only its reference arm.

Public-study import selects six families by default:

- `public_forecast`: the [forecast alternatives](#observable-capability-conditioning) and no progress;
- `model_pooling_leave_one_out`: each respondent model left out in turn;
- `tail_shape`: every GPD shape 0.2 lower and higher;
- `uncapped_severity`: authored caps removed;
- `exposure_scale` and `safeguard_level`: separately collected variant studies with the stipulated deployment scales halved or doubled, or with weaker or stronger safeguards ([`POP-SAFEGUARD-LEVEL`](assumptions.md#pop-safeguard-level)).

The opt-in families are the three count-law alternatives of [pooling](#pooling-and-parameter-uncertainty) (`epistemic_spread_calibration`, `epistemic_tail_width`, `common_frequency_bias`); simulated anchor alternatives (`large_loss_anchor`, `very_large_loss_anchor`, `large_loss_anchor_ignored`); authored cap endpoints (`severity_cap`, `severity_endpoint_stress`); lab-loss alternatives (`lab_loss_cap`, `lab_loss_provider_only`); negative-binomial counts (`count_family`); and each model alone (`model_pooling`).

*Reweighting arms* change only joint-state weights: each forecast alternative is integrated on the study's own nodes by its exact quadrature weights, and leave-one-model-out arms renormalize the remaining respondents' weights. *Variant-study arms* reweight to a variant study's states and are compared with a matched control that reweights the main study's states to the same nodes. *Simulated arms* recompile the captured answers under one change and reuse the reference draws of every state whose laws are unchanged.

Every context-bound result also carries two analytic exhibits with EAL effects only: `frequency_estimate` ([event populations](#event-populations-and-frequency)) and `large_loss_anchor`. The second compiles each law with the anchor ignored, with the [two-anchor tail](#conditional-severity), and with every shaping share's odds of reaching USD 1 billion, given a loss above p99, multiplied by ⅓ and by 3.

A report-only `plausibility` table gives each scenario's welfare EAL, and the total, as a share of world output for the study year: USD 135.093 trillion in 2027, the IMF World Economic Outlook database's projection, read on 2026-11-11 ([`world_output.yaml`](data/reference/world_output.yaml)). It flags a scenario above 0.1% and a total above 1%. It also flags a scenario with at least 0.1 expected incidents a year of at least USD 1 billion, and a total of at least 1. The thresholds are judgments, and a flag changes no result.

The results summary's dependence bracket aligns annual-loss ranks across scenarios within each state. For the stored marginals the comonotone sum is the convex-order upper bound, so no coupling has a larger within-state TVaR; it is not a VaR bound ([`dependence_bracket.py`](src/catmodel/results/dependence_bracket.py)).

[TESTING.md](TESTING.md) sets out the required checks, the [intended-purpose statement](model-intended-purpose.md) governs communication of results, and [model governance](model-governance.md) sets out ownership and change control.
