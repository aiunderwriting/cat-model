# Threat models

A threat model is research on one scenario that its questions quote: who causes its incidents, the necessary steps to a counted incident, what the AI system changes and what limits each step, what defenders do, how outcomes differ in severity, which published analogues exist and how loss reaches the lab. It cites a source for what it reports and states no probability or loss estimate. The register, [`data/threat_models.yaml`](../data/threat_models.yaml), holds one for 27 of the 32 scenarios.

Research is a separate stage because respondents answer without search or tools; the panel still supplies every number.

## The stage

1. **Brief.** `uv run --frozen python scripts/threat_model.py brief SCENARIO_ID --research-through YYYY-MM-DD` prints one scenario's research task.
2. **Research.** An agent with retrieval answers with a YAML draft that includes `drafting` notes on how it read each source.
3. **Check.** `scripts/threat_model.py check DRAFT.yaml` checks the draft against the contract and prints it as a question quotes it.
4. **Verify.** The contract checks form only. Check each finding against its source, check that no step says how it is done, then add the draft to the register without its notes.
5. **Use.** A source bundle whose recipe lists the register captures it as `evidence/threat_models.yaml`, and each scenario's direct, chain and cross-check questions quote its threat model after the source excerpts. An answer may cite its ID in `evidence_used`, which `capability-risk validate` counts (`reasoning_use`).

## What the contract enforces

- Each source is an https URL or a citation naming its year, published no later than the research cutoff and cited by a part of the threat model, with a finding: what the research read in it.
- A part states a figure only if the finding of a source it cites does. A part without a source is shown as "a judgment of the analysis" and states none, like the summary and the key uncertainties. Dates, numbered provisions and identifiers are not figures.
- The analysis states no odds in words ("one in five").
- A pathway begins with its starting condition and marks each step `necessary` (no incident is counted without it) or `escalation` (it only enlarges one).
- Text holds no headings, markup, links or chat-role labels and takes at most 3,000 words in a question, which quotes no finding.
- A threat model records the digest of its scenario's definition. Once that changes, library validation and bundle capture refuse it until it is checked again and the digest updated.
- A study shows it only if its research cutoff is within the study's information cutoffs.
- The register holds at most one per scenario, and a source cited by several carries one publication date.
- An entry needs no `drafting` or `review` record; a bundle captured from a register that kept them still loads and asks the same questions.

## Limits

- **Five scenarios have none:** `compromise_individuals`, `cbrn_biological_expert`, `cbrn_biological_novice`, `cbrn_chemical` and `cbrn_radiological_nuclear`.
- **Lab-side evidence comes mostly from one developer.** All 27 threat models cite Anthropic's sites and 14 cite OpenAI's, so their account of safeguards, misuse and model capability is mostly Anthropic's.
- **The [event chain](../methodology.md#event-populations-and-frequency) does not follow the pathway step for step.** Its question asks about the scenario file's stages, then the probability that an opportunity which passed them becomes a counted incident. Pathways separate steps that this last probability joins, such as the loss reaching the admission threshold and the tie to the lab's model; where a threat model says the chain has no stage for one, that probability covers it.
- **Category questions quote only summaries** of the threat models beneath them.

[Limitations](limitations.md) covers how a shared analysis can anchor a panel.

## Comparing with a published expert threat model

1. Write the published threat model in the register's form: the reference.
2. Produce a candidate blind to it: research stops before its publication and excludes it and its coverage.
3. Map each reference part to the candidate part that states the same thing, or to none.
4. `scripts/threat_model.py compare REFERENCE.yaml CANDIDATE.yaml MAPPING.yaml` counts the steps, severity modes and actors the candidate covers, misses and adds.

The mapping is a judgment; the script only counts. The repository holds no such comparison.
