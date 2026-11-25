# Taxonomy coverage and external reference points

This page lists the harm the model leaves unquantified and maps the [taxonomy](taxonomy.md) against four external risk inventories. The mappings are this repository's reading of each inventory. Their entries overlap, so mapped rows must not be summed, and a mapped control or capability is not another loss event. The [gap register](../registries/coverage_gaps.yaml) records the known gaps in what individual scenarios estimate.

## Research scope and unquantified harm

The research target is harm to every affected party. The model estimates a selected set of incidents involving the named providers. "Unquantified" means relevant to the research target but without a comprehensive estimate; it does not mean zero. A narrow incident forecast can sit beside a broader unquantified gap.

| Population or consequence | Current treatment and numerical boundary |
| --- | --- |
| Consumer advice, companion use and child safety | The output-reliance leaves count admitted incidents of advice acted on. The engagement-harm leaf counts admitted incidents of psychological harm from engaging with the model, whatever the topic. Smaller incidents of consumer advice, emotional injury, dependency or self-harm are the unestimated leftover, and nonmonetary harm is unvalued. |
| Privacy, surveillance and nonconsensual disclosure | Compromise, bounded-decision and privacy-claim incidents include related losses. Accidental provider disclosure is the unestimated other service failure. Standalone privacy invasion, sensitive inference and nonmonetary dignity loss lack a comprehensive estimate; a fine does not value the affected person's privacy. |
| Creative work, intellectual property and identity misuse | Real response costs, productive losses and personal injury count. The private-claims leaves estimate the provider's loss from claims over its training, licensing or data practices, where settlements are transfers. The underlying creative and cultural consequences are unquantified. |
| Copied capability and a provider's other trade secrets | The theft leaf counts takings of model artifacts: weights, checkpoints, adapters, training or fine-tuning data and model files. Copying a model's behavior through the provider's service, by distillation or extraction, belongs to no leaf. Neither do takings of a provider's other code, documents and know-how. |
| Discrimination and worker harms | Bounded-decision errors, including eligibility decisions, apply to a provider's own workforce and to customers' workforces on the same terms. Unequal opportunity, exclusion, harassment and surveillance outside those incidents are incompletely quantified. |
| Political manipulation and institutional harm | Identifiable response, security, operational and injury consequences of admitted political campaigns belong to the influence leaf. Diffuse damage to political participation, institutional trust or democratic legitimacy is unquantified. |
| Economic transition and concentration | No estimate covers labor displacement, degraded job quality, concentration of power, cultural displacement or macroeconomic transition. These effects need explicit counterfactuals and periods. |
| Environmental and infrastructure footprint | Incident-specific physical harm can fall within a scenario. The routine energy, water, materials and emissions effects of training and deployment have no forecast. |
| Autonomy, relationships and dignity | Identifiable injury and response costs can be valued within an incident. Agency, relationships and dignity have no monetary valuation, and damages awards do not stand in for one. |
| AI welfare and nonhuman welfare | No welfare measure covers potentially sentient AI or animals. |
| Civilization-scale and very long-run effects | The loss-of-control and mass-harm scenarios do not estimate extinction, irreversible political change or all future welfare. Finite monetary severities and the origin-year temporal basis limit them. |

## MIT domain taxonomy: all 24 subdomains

Labels paraphrase every subdomain of the [MIT AI Risk Repository](https://airisk.mit.edu/risks), accessed 2026-09-12; identifiers are MIT's. The comparison is with the domain classification, not with each incident or risk in MIT's database.

| MIT ID and concern | Model representation | Material limit |
| --- | --- | --- |
| 1.1 Discriminatory treatment or portrayal | Bounded-decision errors, such as eligibility and content moderation, and output reliance. | Broad representational and opportunity harms are unquantified. |
| 1.2 Harmful content exposure | Influence campaigns, bounded content-moderation errors and engagement harm, which counts psychological harm from engaging with the model on any topic. | Conversational consumer harm below the admission threshold, and nonmonetary harm, are not forecast. |
| 1.3 Group-dependent effectiveness | Errors and loss transitions in the applicable reliance or deployment scenario. | Group disparities and lost benefits are not separately estimated. |
| 2.1 Intrusion into personal information | Compromise of individuals and organizations, input hijack and privacy claims; accidental service disclosure is unestimated. | Non-breach privacy and sensitive inference are incompletely valued. |
| 2.2 Exploitable system weaknesses | System compromise, input hijack, tampered artifacts, model-asset compromise and software reliance. | A vulnerability without realized harm is not a terminal loss; incidents below USD 100,000 are unestimated. |
| 3.1 Incorrect information | Output reliance in five fields and bounded decision errors. | Direct consumer misinformation and belief changes are incompletely represented. |
| 3.2 Degraded shared information environment | Influence campaigns and bounded content-moderation errors. | Diffuse epistemic and institutional harm has no comprehensive estimate. |
| 4.1 Scaled influence and surveillance | Admitted influence campaigns, including political response costs. | Campaigns below the admission threshold and unvalued institutional effects limit coverage. |
| 4.2 Hostile cyber and weapon activity | System compromise and the four CBRN leaves; other AI-assisted processes are unestimated. | No complete conventional-weapons, surveillance or mass-harm census. |
| 4.3 Deception for illicit advantage | Fraud against individuals and organizations, compromise gained by deception, and influence. | Incidents below USD 100,000 are unestimated; personal abuse is incomplete. |
| 5.1 Unsafe trust and dependence | Output reliance and harmful actions under delegated authority. | Consumer dependence and relationship harms lack a dedicated estimate. |
| 5.2 Diminished personal control | Particular harmful autonomous actions and decisions. | Loss of agency is not comprehensively monetized. |
| 6.1 Concentrated power and gains | Particular manipulation or governance incidents. | Market power and the distribution of AI benefits are unquantified. |
| 6.2 Work and income disruption | Employment-related bounded decision errors. | Structural job displacement and changes in work quality are unquantified. |
| 6.3 Displacement of human creative value | Copyright and IP claims against the provider. | Economy-wide cultural and creative displacement is unquantified. |
| 6.4 Competitive deployment pressure | A possible driver of exposure, safeguards and loss-of-control judgments. | Not a standalone loss. |
| 6.5 Inadequate governance | A contributor to modeled incidents; enforcement matters, designations and private claims over provider conduct have their own leaves. | Fine and settlement size measure neither governance quality nor total underlying harm. |
| 6.6 Ecological resource burdens | Qualifying incident-related physical and response loss. | Routine resource use and environmental externalities have no dedicated estimate. |
| 7.1 Conflicting AI objectives | Loss of developer control and harmful agent actions for a deployer. | Scenario coverage does not imply complete long-run consequence coverage. |
| 7.2 Harm-enabling capabilities | Event-chain stages and capability responses for misuse, CBRN and autonomous pathways. | Capability is a causal input, not an independently additive loss. |
| 7.3 Performance failures | Output-reliance, bounded-decision, agent, physical-control and service scenarios according to role, each counting one shared defect at many deployments as one incident. | No empirical guarantee across every task, user or distribution shift. |
| 7.4 Opaque decision processes | A contributor to detection, review and response assumptions. | Opacity alone has no separately estimated dollar loss. |
| 7.5 Interests of AI entities | No dedicated numerical representation. | Moral status and valuation are outside the numerical model. |
| 7.6 Interacting AI systems | Agent cascades and shared defects inside one incident, hijacks through agent-to-agent messages, the nearest-AI convention and joint annual-loss assumptions. | Not every conflict, collusion or emergent interaction is quantified ([open boundary cases](taxonomy.md#open-boundary-cases)). |

## AIUC-1 requirements

The public [AIUC-1 standard](https://www.aiuc-1.com/) lists controls, not loss categories, so its families map to the model inputs their controls bear on. The model applies no loss reduction for certification.

| Requirement family | Use in the model |
| --- | --- |
| [A001–A008: data and privacy](https://www.aiuc-1.com/data-and-privacy) | Control points for access, disclosure and IP pathways; the privacy and IP gaps above remain. |
| [B001–B010: security](https://www.aiuc-1.com/security) | Safeguards for adversarial inputs, privileges, deployment protection and generated code, chiefly bearing on the input-hijack chain stages. Several can constrain the same incident; do not multiply reductions without support. |
| [C001–C012: safety](https://www.aiuc-1.com/safety) | Contextual risk definitions, prevention, review, monitoring and evaluation. A product-specific prohibited topic is not automatically a welfare loss. |
| [D001–D004: reliability](https://www.aiuc-1.com/reliability) | Error and tool-use controls mapped to the relevant event-chain stages, including the execution stages of shared defects. Evaluation failure rates need opportunity and consequence bridges before they become loss probabilities. |
| [E001–E017: accountability](https://www.aiuc-1.com/accountability) | Responsibilities, incident response, records, supplier oversight and disclosure affect prevention or recovery. E007 and E014 are retired entries. |
| [F001–F002: society](https://www.aiuc-1.com/society) | Cyber and CBRN misuse safeguards. |

## Other primary inventories

The twelve risk areas of the [NIST Generative AI Profile, AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), sections 2.1–2.12, are mapped by their definitions, not by compliance with NIST's action catalogue:

| NIST section and abbreviated concern | Current representation or gap |
| --- | --- |
| 2.1 CBRN assistance | Four CBRN leaves by hazard and, for biological, program expertise; capability assistance is upstream of an incident. |
| 2.2 Fabricated information | Output reliance and bounded decisions; incidents below the admission threshold remain a gap. |
| 2.3 Dangerous or hateful material | Influence, content-moderation and health-reliance incidents; direct personal injury incomplete. |
| 2.4 Personal information | Compromise, bounded-decision and privacy-claim incidents; standalone privacy gap. |
| 2.5 Environmental effects | Routine footprint unquantified. |
| 2.6 Biased or homogeneous outputs | Bounded decision errors; broader disparity and common-model exposure incomplete. |
| 2.7 Human–AI relationships | Specific reliance pathways; dependence and autonomy incomplete. |
| 2.8 Information integrity | Influence and reliance; institutional effects incomplete. |
| 2.9 Cyber security | Compromise, hijack, tampering and asset-attack leaves; weaknesses alone not realized loss. |
| 2.10 Intellectual property | Copyright and IP claims against the provider are estimated as provider loss, with settlements as transfers; other IP welfare consequences have no dedicated forecast. |
| 2.11 Abusive imagery | Personal and societal injury gap; lack of a payment or claim does not erase harm. |
| 2.12 Component integration | Tampered artifacts, physical equipment control and shared defects inside agent and bounded-decision incidents; no additive loss for integration itself. |

The ten entries of the [OWASP 2025 LLM application inventory](https://genai.owasp.org/llm-top-10/) are overlapping mechanisms or weaknesses, not ten estimates:

- LLM01 prompt injection, and LLM08 retrieval weaknesses exploited through injected content, are the input-hijack leaf when they commandeer a system someone else operates.
- LLM03 compromised dependencies and LLM04 training or model contamination are routes to a tampered model, artifact or configuration, or to system compromise, once harmful action is established.
- LLM02 disclosure and LLM07 prompt disclosure count only through actual consequences.
- LLM05 unsafe output consumption, LLM06 excess authority and LLM09 misleading output distinguish reliance, bounded execution and autonomous action.
- LLM10 resource exhaustion can produce incremental compute bills and service disruption; ordinary beneficial compute is not an incident loss.

Across these comparisons, the largest unquantified populations of societal harm are direct consumer and interpersonal harms, standalone privacy and identity harms, economic transition, environmental footprint and diffuse institutional effects.

## Checks against reported incidents

[Externally sourced cases](../tests/fixtures/taxonomy_external_cases.yaml) are short narratives of reported incidents, written without the taxonomy's vocabulary, each with an authored interpretation. `tests/test_taxonomy_external_cases.py` replays each interpretation through the hierarchy, to a leaf and scenario or to the missing fact that blocks classification. The cases illustrate boundaries and are not a sample. To test the definitions, give a reader only each case's `source` and `narrative` and compare the answer with `interpretation`; an ambiguity this exposes should change the definitions.
