# Model Intended Purpose

## Intended purpose

The model has two purposes:

- to map the monetary harm that AI-related incidents cause across society;
- to estimate what those incidents cost a named lab.

Both are research uses: estimating the annual loss of selected populations, comparing risks and finding which assumptions matter most.

The headline measures are societal harm (the `welfare` view) and the named lab's economic loss (`lab_loss::<lab_id>`), reported side by side. [Incident accounting](docs/incident-accounting.md) defines every view.

The hypothetical-lab study asks about one named lab, AI model and deployment along a forecast of public-frontier capability. The same questions at fixed ECI scores support conditional what-if analysis. Financial results need authored insurance terms.

## Interpretation and exclusions

The model does not provide:

- an insurance premium, reserve or capital requirement;
- an actuarial or legal opinion, or a regulatory filing;
- a decision to deploy a model;
- a measure of how well a mitigation works, which needs a counterfactual that a sensitivity result does not supply;
- a value for non-monetary harm, diffuse macroeconomic effects or AI's benefits ([limitation 7](docs/limitations.md#7-the-model-quantifies-selected-populations-only)).

Every communication of a result states the populations it includes and leaves out, with their exposure, units and admission threshold. It states the study year, the currency and valuation basis, and the respondent models and sources it rests on. It also states any financial terms, which comparisons ran, and the [limitations](docs/limitations.md) that apply.

## Research publication

Published research may rest on reasoned judgment with sparse evidence or incomplete validation, provided it states its assumptions, the comparisons that ran and its limitations. Correct a mathematical contradiction, such as an impossible probability, inconsistent units or a double-counted component, before presenting a result, and keep the original evidence. Judgments that disagree may stand as alternatives. Publishing captured evidence needs rights clearance and a check for hazardous detail ([limitation 11](docs/limitations.md#11-captured-evidence-can-carry-third-party-rights)).

## Status

The model is experimental research software, released under the [Apache License 2.0](LICENSE). Its results are research estimates, not insurance pricing or actuarial work product ([limitation 10](docs/limitations.md#10-results-are-research-estimates-not-actuarial-work-product)).
