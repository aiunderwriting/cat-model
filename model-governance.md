# Model Governance

The model is experimental research software, maintained by AIUC and released under the [Apache License 2.0](LICENSE). The [intended-purpose statement](model-intended-purpose.md) says what the model is for, and the [release process](RELEASE.md) lists the records each study keeps.

## Authorities and ownership

The repository's code, scenario definitions and hierarchy contract define the method, and the documentation describes it. Each datum is authored once, and frozen copies are generated from it with checksums. No source-bundle content may derive from elicitation output. Small public references can live in Git. Restricted or large evidence stays in access-controlled storage outside it, and secrets never enter it. A synthetic fixture never becomes study evidence by being copied or restamped.

## Responsibilities

- AIUC maintains the model. Developers make every change through a pull request.
- Study operators choose the respondent models ([model profiles](runbook.md#model-profiles)), curate evidence, freeze complete inputs, keep every collection attempt and result, and report the outputs that are unavailable.

## Change controls

Changes follow [CONTRIBUTING.md](CONTRIBUTING.md) and [TESTING.md](TESTING.md).

- A material numerical change needs a before-and-after comparison with fixed seeds, and says which conclusions move.
- A change to an assumption or to the model's structure updates its row of the [assumption register](assumptions.md): the rationale, the uncertainty and the required sensitivity.
- A contract version identifies one meaning. Do not reuse it for different captured evidence.
- Population and counting definitions are authored once, in the scenario YAML, and no run generates or changes one. Collection and import refuse a node that departs from its scenario's current definition ([runbook](runbook.md#2-context-bound-labdeployment-questions)).
