# cat-model

An experimental catastrophe model for AI risk, built by [AIUC](https://aiuc.com), the Artificial Intelligence Underwriting Company.

In insurance, a catastrophe model estimates the probability distribution of annual loss from rare, severe events. This one does that for incidents involving a frontier AI lab's models: misuse that a model materially assists, failures of the models, and the lab's own conduct. A panel of large language models (LLMs) judges how often incidents happen and what they cost, and a simulation turns those judgments into loss distributions for one study year.

Every result reports two measures of the same simulated incidents:

- **Societal harm** (the `welfare` view): the monetary loss across everyone an incident affects, each loss counted once. It adds resource costs, injury valuations and appropriated assets.
- **A named lab's loss** (the `lab_loss::<lab_id>` view): that lab's own costs, fines, compensation paid to others and lost business.

The software is experimental, and its interfaces change without notice. The study it ships describes a stipulated hypothetical frontier lab. The judgments behind every number come from LLMs and cannot in general be backtested; the [limitations](docs/limitations.md) say how to read an output.

## What a result measures

A result is narrower than the catastrophic risk of frontier AI. It is the monetary loss of the incidents a study selects, for one lab and one study year.

| | A result counts | It is not |
| --- | --- | --- |
| Incidents | The scenarios a study selects, for one named or hypothetical lab's models. An incident counts when its gross loss is at least USD 100,000. | A total for every lab or for all AI. A population without an estimate is reported as unquantified, not as zero. |
| Attribution | The whole loss of each counted incident. An attack that used a model without material assistance is not counted. | The loss that AI added. Nothing that would have happened without the model is subtracted. |
| Time | Each incident's eventual loss, in the year the incident begins. | The cash paid or lost in a calendar year. |
| Societal harm | Money: resource costs, injury and death valued as compensation, and appropriated assets. | A complete measure of welfare. No value per fatality is stipulated, so respondents value injury inside each loss judgment. Non-monetary harm, diffuse economic effects and AI's benefits are outside it, and no scenario values extinction or irreversible political change. |
| The lab's loss | The lab's own costs, fines, compensation paid to others and lost business, before recoveries. | An insured loss. Payments under a policy need authored terms. |

[Limitation 7](docs/limitations.md#7-the-model-quantifies-selected-populations-only), [incident accounting](docs/incident-accounting.md) and [taxonomy coverage](docs/taxonomy-coverage.md) give the detail.

## How it works

1. **Scenarios define what is counted.** Each scenario is one leaf of a [loss taxonomy](docs/taxonomy.md). It defines what counts as one incident, how its loss is measured, and a short event chain used as a second route to its frequency. An incident counts when its gross loss is at least USD 100,000.
2. **A panel of LLMs gives quantile judgments.** For each scenario the respondents state quantiles of the expected annual number of incidents, of the loss of one incident and of the lab's own loss ([example questions](docs/example-questions/README.md)). They answer from the question, dated evidence excerpts and, for most scenarios, a [threat model](docs/threat-models.md) researched beforehand.
3. **Questions are conditioned on AI capability.** Capability is measured by [Epoch AI's Epoch Capabilities Index](https://epoch.ai/eci). A study asks every question along several trajectories of a forecast of the public frontier and weights the answers by that forecast ([capability inputs](docs/capability-frontiers.md)).
4. **Authored assumptions complete each law.** Severity tail shapes and caps, count families and loss accounting come from the scenario definitions and the [assumption register](assumptions.md).
5. **Simulation produces the distribution.** Incidents arrive as a Poisson process and each draws its loss. The model mixes over respondents, their stated uncertainty and the capability trajectories, then reports each view's expected annual loss (EAL) and annual loss distribution.

The [methodology](methodology.md#model-on-one-page) gives the equation and a worked example.

## Quickstart

You need Linux or macOS, Python 3.13 or 3.14 and [uv](https://docs.astral.sh/uv/) 0.9.17 or newer. Windows is not supported.

```sh
git clone https://github.com/aiunderwriting/cat-model.git
cd cat-model
uv sync --frozen
```

**Run the engine on a hand-checkable example.** The [golden Poisson example](config/setups/golden-poisson/README.md) is the smallest study whose results can be checked by hand. Its inputs are stipulated, so it checks the software and says nothing about risk.

```sh
uv run --frozen catmodel setup build \
    --spec config/setups/golden-poisson/setup.json \
    --plan config/setups/golden-poisson/plan.json \
    --setup-id golden-poisson --output /tmp/golden-setup
uv run --frozen catmodel run --setup /tmp/golden-setup --output /tmp/golden-run
uv run --frozen catmodel results summary \
    --results /tmp/golden-run/results.json --output /tmp/golden-summary
```

`/tmp/golden-summary/summary.md` reports an expected annual loss of 14: incidents arrive at 0.5 a year and cost 28 on average.

**Run the whole pipeline offline.** `scripts/offline_demo.py` takes one scenario from frozen questions to a results summary in about a minute, with no network or API key. A fake respondent supplies the answers, so the output shows what each step reads and writes and says nothing about risk. Run it from a clean checkout, into a new directory outside it:

```sh
uv run --frozen python scripts/offline_demo.py /tmp/offline-demo
```

**Run a study with real models.** A collection reads provider API keys from environment variables or a `.env` file (`.env.example` lists the names) and takes its respondent models from a [model profile](runbook.md#model-profiles). The [runbook](runbook.md#running-the-hypothetical-lab-study) covers planning, a dry run and collection, from a study of a few scenarios to the full hypothetical-lab study.

## Scenario library

Each scenario is a leaf of the loss taxonomy, a tree. The tree splits losses by how they arise: what started it, what the AI did, and what process caused the loss. Each incident is counted once, with all its losses, under the process that caused most of them.

The catalog holds <!-- scenario-count:cells -->32<!-- /scenario-count --> scenarios, one for each estimated leaf of the [hierarchy contract](contracts/scenarios/hierarchy-contract.yaml): <!-- scenario-count:origin.deliberate_misuse -->15<!-- /scenario-count --> under deliberate misuse, <!-- scenario-count:origin.ai_failure -->11<!-- /scenario-count --> under AI failure and <!-- scenario-count:origin.provider_conduct -->6<!-- /scenario-count --> under provider conduct. Each belongs to one of <!-- scenario-count:risk_domains -->8<!-- /scenario-count --> risk domains, such as cyber, CBRN weapons and loss of control, and results add expected annual loss up by domain. A further <!-- scenario-count:leaves.unestimated -->4<!-- /scenario-count --> leaves are unestimated and <!-- scenario-count:leaves.unsupported -->1<!-- /scenario-count --> is unsupported. In <!-- scenario-count:normalisation_bases -->16<!-- /scenario-count --> scenarios an exposure base only normalises the lab's size; the [coverage-gap register](registries/coverage_gaps.yaml) records each.

The [taxonomy guide](docs/taxonomy.md) and the [catalog](scenarios/scenario_catalog.json) show every scenario, and [taxonomy coverage](docs/taxonomy-coverage.md) lists the harms the model does not quantify. `docs/taxonomy-explorer.html` is an interactive view; open it in a browser from a clone.

## Documentation

- [Methodology](methodology.md): the model on one page, then the full calculation.
- [Limitations](docs/limitations.md): what to keep in mind when reading any output.
- [Assumption register](assumptions.md): each assumption, its evidence status and the sensitivity that challenges it.
- [Runbook](runbook.md): evidence, collection and execution commands.
- [Setup, run and results](docs/setup-run-results.md) and [reading results](docs/research-results.md): the files a run writes and how to interpret them.
- [Taxonomy](docs/taxonomy.md), [threat models](docs/threat-models.md), [capability inputs](docs/capability-frontiers.md) and [incident accounting](docs/incident-accounting.md).
- [Architecture](ARCHITECTURE.md): the map of the code, from the equation to the modules. [Glossary](docs/glossary.md): terms of art.
- [Intended purpose](model-intended-purpose.md), [governance](model-governance.md), [contributing](CONTRIBUTING.md), [testing](TESTING.md) and [security](SECURITY.md).

## About AIUC

The Artificial Intelligence Underwriting Company ([AIUC](https://aiuc.com)) underwrites AI risk. Its current product is [AIUC-1](https://github.com/aiunderwriting/AIUC-1-Changelog) certification for AI agents, and its long-term aim is to [underwrite superintelligence](https://underwriting-superintelligence.com). Questions about the model go to tom@aiuc.com.

## Citing

[CITATION.cff](CITATION.cff) gives the citation metadata. A reported result should also name the code revision its setup records.

## License

AIUC's code, data and documentation in this repository are licensed under the [Apache License, Version 2.0](LICENSE); [NOTICE](NOTICE) holds the copyright notice and the attributions that the committed third-party data require. Third-party data and documents keep their own licenses and attribution requirements, recorded in each source's provenance file; for example, the Epoch AI data in [`data/capability/`](data/capability/README.md) is licensed under CC BY 4.0.
