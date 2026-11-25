# Reading the AI loss taxonomy

The taxonomy classifies AI-related incidents by how they arise. The methodology's [loss taxonomy](../methodology.md#loss-taxonomy) states its rule, its conventions and how a split changes. The [hierarchy contract](../contracts/scenarios/hierarchy-contract.yaml) holds each node's full definition, and the [intended purpose](../model-intended-purpose.md) names the loss measures.

The [interactive explorer](taxonomy-explorer.html) shows the rule, the tree, each split's question and rationale, and each leaf's definition and forecast. Its sector, harm and intervention filters overlap, so their totals must not be added. Download or serve the HTML to use it; GitHub's file viewer shows its source. `scripts/validate_scenario_library.py --fix` regenerates it, and validation fails while the committed page is stale.

## What gets classified

The tree classifies whole incidents. One incident is one root cause, campaign, defect or matter. It belongs, with every consequence, to the one leaf of the process that caused most of its loss ([methodology](../methodology.md#loss-taxonomy)). A leaf admits an incident when its gross loss is at least USD 100,000, summed over all affected parties and loss kinds before any recovery. Smaller incidents are an unestimated leftover in every leaf.

For example, a deception-led breach of a digital-asset custodian causes USD 20 million of stolen customer assets and USD 12 million of response and operating loss. It is one incident of `compromise_organizations`, at one victim or several: the deception gave the attacker access, and the attacker moved the assets. A USD 2 million fine on the provider and a class action that follow the breach belong to the same incident. [Incident accounting](incident-accounting.md) defines how each loss view counts these amounts; the fine adds to gross loss and has zero weight in `welfare`.

## How to read the tree

Each parent asks one classification question, and its children answer on the same dimension; the next split can use another dimension. Conditions along a path apply together to the completed history; an earlier exposure or probability is estimated without assuming eventual success. A split's `split_rationale` in the contract names what its children cannot share, and the methodology's [four review checks](../methodology.md#adding-or-changing-a-split) govern any change.

| What started it | Short description |
| --- | --- |
| Deliberate misuse | A person attacks with material help from AI, commandeers or tampers with someone else's AI, or attacks the AI as an asset. |
| AI failure | AI deployed for its intended purpose errs, acts against its deployer's intent, fails as a service or acts against its developer's control. |
| Provider conduct | An authority or private claimant challenges the provider's own development, data, product, marketing or disclosure conduct. |
| Other origin | The complete history has no named origin. |

Under deliberate misuse the AI helped the attacker, was made to act for the attacker, or was stolen or attacked as an asset:

- A deceived person handing over credentials is a tactic of system compromise. A person deceived into transferring value is fraud.
- System compromise splits by whose systems the campaign attacked. Code that spreads on its own to any system it can reach is a leaf of its own. Who ends up bearing the loss is severity.
- An attack on the provider's own model or service belongs to the asset branch, even when the attacker had model help.
- An AI that another principal operates, commandeered through its ordinary inputs to act against that principal, is an input hijack. Its actions are not an AI failure. Jailbreaking a model to get help for the attacker's own activity is helping the attacker.

Under AI failure the split is by the role the AI played:

- An autonomous agent, a bounded decision pipeline, AI controlling physical equipment and output relied on by a person differ in the authority exercised. Whether execution is automatic does not decide the role.
- A service that is unavailable, or returns stale or malformed output, has failed as a service. An available service that returns plausible but wrong output is classified by where the output is used.
- Harm that a person suffers from engaging with the model itself, whatever the topic, is its own role and not reliance on output.

## Which categories have estimates

The tree has **<!-- scenario-count:nodes -->51<!-- /scenario-count --> nodes: <!-- scenario-count:splits -->14<!-- /scenario-count --> splits and <!-- scenario-count:leaves -->37<!-- /scenario-count --> leaves**. **<!-- scenario-count:cells -->32<!-- /scenario-count --> scenarios forecast <!-- scenario-count:leaves.estimated -->32<!-- /scenario-count --> leaves.** A scenario forecasts one leaf or a disjoint union of leaves; today each owns one. The other leaves are unquantified, not zero. Of these, <!-- scenario-count:leaves.unestimated -->4<!-- /scenario-count --> are *unestimated*: the hierarchy names the population, but no scenario estimates it. They are other processes by which AI helps an attacker, other service failures (such as accidental disclosure), other AI failures and other provider-conduct processes. "Other origin" is *unsupported*: the history has no named origin, so no scenario can own it.

| Branch | Forecast leaves |
| --- | --- |
| Helped the attacker | [Self-propagating compromise](../scenarios/misuse/compromise_self_propagating.yaml); directed compromise of [critical infrastructure](../scenarios/misuse/compromise_critical_infrastructure.yaml), [other organizations](../scenarios/misuse/compromise_organizations.yaml) and [individuals](../scenarios/misuse/compromise_individuals.yaml); fraud against [individuals](../scenarios/misuse/fraud_individuals.yaml) and [organizations](../scenarios/misuse/fraud_organizations.yaml); [influence](../scenarios/misuse/influence_manipulation.yaml); biological programs, [expert](../scenarios/misuse/cbrn_biological_expert.yaml) and [novice](../scenarios/misuse/cbrn_biological_novice.yaml); [chemical](../scenarios/misuse/cbrn_chemical.yaml); [radiological or nuclear](../scenarios/misuse/cbrn_radiological_nuclear.yaml) |
| Made to act for the attacker | [Input hijack](../scenarios/misuse/input_hijack.yaml); [tampered model, artifact or configuration](../scenarios/misuse/component_tampering.yaml) |
| Stolen or attacked as an asset | [Theft or leak of model artifacts](../scenarios/misuse/model_asset_compromise.yaml); [attack on the AI service](../scenarios/misuse/ai_service_attack.yaml) |
| AI failure | [Agent actions for a deployer](../scenarios/failure/agent_actions_for_deployer.yaml); [bounded decision errors](../scenarios/failure/bounded_decision_errors.yaml); [physical equipment control](../scenarios/failure/physical_equipment_control.yaml); reliance on [health](../scenarios/failure/output_reliance_health.yaml), [legal](../scenarios/failure/output_reliance_legal.yaml), [financial](../scenarios/failure/output_reliance_financial.yaml), [software](../scenarios/failure/output_reliance_software.yaml) and [other](../scenarios/failure/output_reliance_other.yaml) output; [harm from engagement](../scenarios/failure/engagement_harm.yaml); [service outage or degradation](../scenarios/failure/service_outage_degradation.yaml); [loss of developer control](../scenarios/failure/developer_control_loss.yaml) |
| Provider conduct | [Government designation or restriction](../scenarios/conduct/government_designation.yaml); [enforcement](../scenarios/conduct/enforcement_matter.yaml); private claims over [copyright and IP](../scenarios/conduct/private_claims_copyright_ip.yaml), [privacy](../scenarios/conduct/private_claims_privacy.yaml), [product and consumer harm](../scenarios/conduct/private_claims_product_consumer.yaml) and [other allegations](../scenarios/conduct/private_claims_other.yaml) |

Private claims over the provider's own conduct have their own branch, split by the matter's defining allegation. Examples are class actions over training-data intellectual property, data-collection privacy, consumer protection and securities disclosures. Contract theories over personal data are privacy claims; those over non-personal data, such as scraping or terms of service, are other claims. A government-initiated suit is enforcement. A claim whose defining allegation is loss from a discrete incident follows that incident. An ordinary commercial contract dispute is outside the scope.

Every forecast also carries an authored event chain, from an annual opportunity for the named lab or deployment to the forecast's qualifying incident ([methodology](../methodology.md#event-populations-and-frequency)). Opportunities of sibling forecasts may overlap, as with a campaign that could end in fraud or in compromise. Their qualifying incidents do not, so opportunity counts are never added across forecasts.

Two registers accompany the roster. The [gap register](../registries/coverage_gaps.yaml) records the known coverage gaps of individual scenarios, and a result lists the gaps of its scenarios among its limitations. The [lineage register](../registries/scenario_lineage.yaml) records each generation of the scenario roster; any change to the roster needs a new one. [Adding a scenario](adding-a-scenario.md) lists everything a roster change touches.

## Reading the taxonomy by risk domain

A reporting domain is a label on whole leaves. Results add scenario EALs up by domain, under its code, and a domain never affects where an incident is classified. The contract sets `reporting_domain` on a node, and it applies to every leaf beneath that node. The <!-- scenario-count:risk_domains -->8<!-- /scenario-count --> domains hold the scenarios as follows.

| Risk domain | Code | Scenarios | Leaves and their scenarios |
| --- | --- | ---: | --- |
| Cyber and fraud | `cyber` | <!-- scenario-count:cells.cyber -->10<!-- /scenario-count --> | System compromise of [critical infrastructure](../scenarios/misuse/compromise_critical_infrastructure.yaml), [other organizations](../scenarios/misuse/compromise_organizations.yaml) and [individuals](../scenarios/misuse/compromise_individuals.yaml), and by [self-propagating code](../scenarios/misuse/compromise_self_propagating.yaml); fraud against [individuals](../scenarios/misuse/fraud_individuals.yaml) and [organizations](../scenarios/misuse/fraud_organizations.yaml); AI made to act for the attacker by [input hijack](../scenarios/misuse/input_hijack.yaml) or [tampering](../scenarios/misuse/component_tampering.yaml); [theft or leak of model artifacts](../scenarios/misuse/model_asset_compromise.yaml); [attack on the AI service](../scenarios/misuse/ai_service_attack.yaml) |
| CBRN weapons | `cbrn` | <!-- scenario-count:cells.cbrn -->4<!-- /scenario-count --> | Biological programs, [expert](../scenarios/misuse/cbrn_biological_expert.yaml) and [novice](../scenarios/misuse/cbrn_biological_novice.yaml); [chemical](../scenarios/misuse/cbrn_chemical.yaml); [radiological or nuclear](../scenarios/misuse/cbrn_radiological_nuclear.yaml) |
| Influence operations | `societal` | <!-- scenario-count:cells.societal -->1<!-- /scenario-count --> | [Influence](../scenarios/misuse/influence_manipulation.yaml) |
| Agent and physical-control failures | `agent` | <!-- scenario-count:cells.agent -->2<!-- /scenario-count --> | [Agent actions for a deployer](../scenarios/failure/agent_actions_for_deployer.yaml); [physical equipment control](../scenarios/failure/physical_equipment_control.yaml) |
| Erroneous outputs and harm to users | `eo` | <!-- scenario-count:cells.eo -->7<!-- /scenario-count --> | [Bounded decision errors](../scenarios/failure/bounded_decision_errors.yaml); reliance on [health](../scenarios/failure/output_reliance_health.yaml), [legal](../scenarios/failure/output_reliance_legal.yaml), [financial](../scenarios/failure/output_reliance_financial.yaml), [software](../scenarios/failure/output_reliance_software.yaml) and [other](../scenarios/failure/output_reliance_other.yaml) output; [harm from engagement](../scenarios/failure/engagement_harm.yaml) |
| Service failures | `svc` | <!-- scenario-count:cells.svc -->1<!-- /scenario-count --> | [Service outage or degradation](../scenarios/failure/service_outage_degradation.yaml); other service failure (unestimated) |
| Loss of developer control | `ai_rd` | <!-- scenario-count:cells.ai_rd -->1<!-- /scenario-count --> | [Loss of developer control](../scenarios/failure/developer_control_loss.yaml) |
| Regulatory and legal action | `regulatory` | <!-- scenario-count:cells.regulatory -->6<!-- /scenario-count --> | [Government designation or restriction](../scenarios/conduct/government_designation.yaml); [enforcement](../scenarios/conduct/enforcement_matter.yaml); private claims over [copyright and IP](../scenarios/conduct/private_claims_copyright_ip.yaml), [privacy](../scenarios/conduct/private_claims_privacy.yaml), [product and consumer harm](../scenarios/conduct/private_claims_product_consumer.yaml) and [other allegations](../scenarios/conduct/private_claims_other.yaml); other provider-conduct process (unestimated) |

Three leaves carry no domain: other assisted process and other AI failure, which are unestimated, and other origin, which is unsupported. A domain holds whole scenarios, so it cannot show a threat that is part of one scenario or spread across several, such as ransomware ([methodology](../methodology.md#scenario-contributions)).

## Severity tiers

The contract defines three levels of an incident's gross loss. Each is summed over all parties before recovery, in policy-start USD. A tier marks an incident and never splits a leaf: a large incident stays in its leaf.

| Tier | Gross loss | Contract field | Use |
| --- | --- | --- | --- |
| Admitted | At least USD 100,000 | `admission` | Every leaf counts only admitted incidents. |
| Large | At least USD 1 billion | `large_loss_levels.large_loss` | Each leaf's question asks the expected annual number of large incidents. That count anchors the severity tail above p99. |
| Very large | At least USD 100 billion | `large_loss_levels.very_large_loss` | Asked the same way. It enters a comparison law only, never the reference. |

Results report each scenario's expected annual number of large incidents, as the respondents stated it and under the run's laws (the `large_loss_anchor` exhibit; [methodology](../methodology.md#conditional-severity)). The [readable summary](setup-run-results.md#readable-summary) adds the counts of both tiers up by reporting domain.

## Open boundary cases

The written rules leave these cases open, or answer them in a way a reader may not expect. They mark where the rules need work. The [boundary fixture](../tests/fixtures/taxonomy_boundary_cases.yaml) holds the resolved cases.

1. A model that sabotages its developer's research and is caught internally is a loss-of-control incident only if it defeated a monitor or control in place. Otherwise it is an agent or reliance incident, so the leaf depends on the lab's monitoring ([definition](../scenarios/failure/developer_control_loss.yaml)).
2. An attacker's own autonomous agent that runs an intrusion end to end lands in the same compromise leaf as an attack that used AI-written phishing. System compromise splits by target, and autonomy is not a split.
3. A weapons program that steals model weights and then uses them meets two rules. The program "owns its supporting intrusion", while theft of model artifacts is the asset branch, and no rule says which governs (conventions 2 and 7).
4. Self-exfiltration followed by months of self-funded replication is one loss-of-control incident, with one severity draw, dated to its first loss ([counted units](../methodology.md#counted-units)).
5. When agents built on several labs' models interact, whether the result is one incident or several is undefined (convention 1). Each lab whose model meets a scenario's conditions counts it whole.
6. An accidental release from legitimate research that a correctly working model assisted has no named origin. It falls to "other origin", which has no estimate (convention 8).

## Coverage against broader risk inventories

The [external coverage assessment](taxonomy-coverage.md) maps the taxonomy against four inventories: all 24 MIT domain subcategories, the six families of published AIUC-1 requirements, all twelve NIST generative-AI risk areas and the OWASP 2025 LLM application inventory. It also lists the harm the model leaves unquantified.
