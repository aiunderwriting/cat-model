# Architecture

This page maps the code for a new contributor. Read the [README](README.md) first. The [methodology](methodology.md) explains the calculations, and the [glossary](docs/glossary.md) defines the terms used here.

## The pipeline

Evidence flows one way: each stage writes immutable files, and later stages only read them. `scripts/run_hypothetical_lab_study.py` runs every stage of the hypothetical-lab study in order.

| Stage | Command | Runs in | Contract it writes | On disk |
| --- | --- | --- | --- | --- |
| Source bundle | `package build-source-bundle` | `catmodel.evidence.source_inputs` | `SourceBundleManifest` | `manifest.yaml`, `provenance.yaml` and the captured files |
| Study plan | `capability-forecast public-study`; `capability-risk build-instrument` for one instrument | `catmodel.setup.public_study`; `catmodel.setup.contextual_inputs` | `PublicFrontierForecast`, `PublicFrontierQuadrature`, `ContextualRiskInstrument` | a study's `public-study.json`, `forecast.json`, `public-frontier.json` and `instruments/<node>-<status>.json`; or the `--output` file |
| Collection | `capability-risk collect`; `validate` | `catmodel.collection.contextual_collection`; `catmodel.setup.contextual_inputs` | `ContextualRiskArtifact` | `contextual-risk.json` and `calls/` in each collection directory, under `collections/` for a study |
| Setup | `setup import`; `setup build` | `catmodel.setup.public_study` or `contextual_import`; `catmodel.setup.store` | `RunSetup`, `ExecutionPlan` | `setup-manifest.json`, `setup.json`, `alternatives/`, `execution-plan.json` and `evidence/` |
| Run | `run` | `catmodel.results.execution` | `RunCapture`, `NativeResults` | `capture/run-capture.json`, `capture/arrays/`, `results.json`, `results.diagnostics.json` and `tornado.svg` |
| Replay | `results replay` | `catmodel.results.execution` | `NativeResults` | `results.json` and `results.diagnostics.json` |
| Summary | `results summary` | `catmodel.results.results_summary` | none; it reads `NativeResults` | `summary.md` and `summary.csv` |
| Export | `results export` | `catmodel.results.results_export` | `ResultsExport` | `export.json`, three capability-scenario files and checksummed copies under `native/` |

Every command that writes takes `--output`, a path that must not exist yet; collections resume in their own directory instead. Two things stand beside the pipeline. `scripts/respondent_calibration.py` (`catmodel.collection.respondent_calibration`) fits a respondent spread and level bias on known-answer questions, a diagnostic that no stage reads. An answer table (`--answers`, [runbook](runbook.md#answer-table)) keeps every provider response by its exact prompt, so a later collection sends only what the table cannot answer.

The [workflow map](docs/workflows.md) lists every command and its tier, the [runbook](runbook.md) gives the commands in full, and [Setup, run and results](docs/setup-run-results.md) describes every file.

## From the equation to the code

This table maps each line of the [methodology's one-page equation](methodology.md#the-equation) to the code that implements it in the main pipeline. Modules are under `catmodel`; a name with a leading underscore is private to its module.

| Equation | Implemented in |
| --- | --- |
| `L = Σ_s Σ_i X_{s,i}`, a year's loss in each view | `simulation.incident_runtime.simulate_incident_state` draws every scenario's incidents and sums each year's losses by view. |
| `N_s ~ Poisson(Λ_s)` | `simulation.incident_runtime._origin_counts` draws each year's count by inversion from one uniform, with `probability.count_inversion.poisson_counts`. |
| `Λ_s = Q_{r,s}(Φ(k_s a_{s,y}); c)` | `elicitation.epistemic.stretched_score_quantiles` reads the count law through its five quantiles at a stretched score, and `epistemic_mean` gives its mean `Λ̄_s`. `simulation.incident_runtime.parameter_scores` draws the year's score `a_{s,y}`, and `score_multiplier` gives the year's rate as a multiple of `Λ̄_s`. |
| `Q_{r,s}`, the final count law | `elicitation.chain_law.chain_count_law` composes the chain count, and `pooled_count` pools it with the direct count; `elicitation.eal_crosscheck.pooled_answer` applies the pool to the cross-checked answer. |
| The taxonomy parents' pool, a factor on `Q_{r,s}` | `setup.contextual_hierarchy.hierarchy_outcome` pools each parent with what is beneath it (`elicitation.contextual_hierarchy.pool_parent`). `setup.contextual_setup.contextual_respondent_selections` scales each leaf's count by its factor. |
| The capability condition `c` | `forecast.public_frontier_quadrature.public_frontier_quadrature` places the trajectory nodes. At fixed anchors, `setup.contextual_node.bracket_anchors` and `interpolated_rate` interpolate between them. |
| `X_{s,i} = Σ_j X_{s,i,j}`, one member and one draw | `setup.contextual_node._node_law` gives every incident one member. `simulation.incident_linear.draw_single_member_events` draws its components once. |
| `X_{s,i} ~ F_{r,s}(·; c)`, split into components `j` by shares | `probability.distributions.Distribution` is the severity law: spline body, GPD tail, large-loss anchor and cap. `setup.contextual_node._shared_total` builds it from the answers, and `_branch_outcomes` and `_component` split it by the elicited shares. |
| `lab loss = Σ min(provider-borne + η · other, lab_cap) + …` | `setup.contextual_node._component` gives each component its lab weight, one if provider-borne and otherwise the share `η` from `_lab_compensation_share`; `_lab_losses` sets the cap. `simulation.incident_linear.linear_values` applies the weights and caps each incident. |
| `H_s`, the annual business loss | `setup.contextual_node._lab_annual_business_loss` builds its law. `simulation.incident_runtime._add_annual_lab_losses` draws it once in each year with at least one incident. |
| `gross`, `welfare` and `net_resource_cost` | `simulation.components.component_view_weight` weighs a component in each view (`economic_loss.welfare_weight` for welfare). `simulation.incident_moments.annual_views` adds `H_s` to the gross and lab views only. |
| `Pr(L ≤ x) = Σ_z w_z · Pr(L ≤ x \| z)` | `simulation.capability_mixture.CapabilityMixture` mixes the states' simulated years by their weights (`samples`, `metrics`). |
| `w_z = ω_d · v_r` | `setup.public_frontier.public_frontier_setup_fields` sets each joint state's weight: the node weight `ω_d` from `public_frontier_quadrature`, times the respondent weight `v_r` from `setup.excluded_samples.pooled_weights`. |
| `EAL = Σ_z w_z Σ_s [Λ̄_s E[X_s \| z] + …]` | `simulation.incident_moments.analytic_linear_eals` computes each state's analytic means, with `at_least_one_probability` for the business-loss term and `capped_episode_lab_mean` for a capped lab mean. `simulation.capability_mixture.native_mixture_mean` mixes them over the states. |

`setup.contextual_node.compile_contextual_node` compiles one node's answers into its incident law, and `setup.public_frontier` assembles the laws of every node and respondent into joint states.

## Repository layout

| Path | Holds |
| --- | --- |
| `scenarios/` | The scenario library: one YAML file per scenario under `misuse/`, `failure/` and `conduct/`, shared rosters and tail classes, and the generated `scenario_catalog.json`. |
| `contracts/scenarios/` | The hierarchy contract: the loss taxonomy, its admission threshold and the large-loss levels. |
| `registries/` | The coverage-gap register and the scenario lineage register. |
| `config/` | Stipulated example setups (`config/setups/`) and the hypothetical-lab study's context (`config/studies/`). |
| `data/` | Committed inputs: the Epoch ECI table and forecast template, source cards, the threat-model register, example model profiles, the world-output reference and calibration seeds. |

The wheel packages `config/`, `scenarios/`, `contracts/`, `registries/` and parts of `data/` as resources, which `catmodel.resources` locates.

## Package map

The import-linter contract "Pipeline stages" in `pyproject.toml` puts every package and top-level module of `src/catmodel/` in exactly one of 10 layers. A module may import modules in lower layers, never in higher ones, and packages separated by `|` within a layer may not import each other. Imports under `if TYPE_CHECKING:` do not count. `uv run lint-imports` checks the contracts. `tests/test_import_contract_config.py` checks that every module has exactly one layer and that no module imports a private name from another module; a package may import its own private submodules.

The layers follow, top first. Within a package, modules whose names say what they hold are not listed.

**1. `catmodel.cli`.** The `catmodel` command. Each command parses its options and hands them to a function in a lower layer. The root group defines the global options `--profile` and `--workers`; `_workflow` holds `setup`, `run` and `results`, and `_helpers` shared options and checks.

**2. `catmodel.collection | catmodel.results`.** The two ends of the pipeline: asking respondents, and running a setup and reporting on it. Neither imports the other.

`catmodel.collection` is the only package that calls a provider.

- `contextual_collection`: collection of one instrument or a whole study, asking each leaf the questions its paths in `catmodel.elicitation.question_flow` yield.
- `frozen_call_collection` and `collection_scheduler`: one call collected with strict parsing and retries, and one rate-adaptive lane per model.
- `collection_workers`: a study's contexts divided among collector processes, each with its share of every model's ceiling, because one process serves its requests on one CPU core.
- `answer_table`: the append-only store of provider responses that a collection reads before each request and adds to after it.

`catmodel.results` runs a setup, reduces its capture and reports.

- `execution`: running a setup, loading and checking a capture, and replaying it.
- `reduction`: reduces captured draws under target weights into a `ReducedResults` report, which execution completes into a `NativeResults` report.
- `capability_storage`: stores and restores the capture's numeric arrays.
- `conditional`: conditional kernels read from a capture for what-if analysis.

**3. `catmodel.setup`.** From a study plan and its captured answers to a run setup.

- `public_study`: forecast preparation, public-study planning and public-study import.
- `contextual_import`: fixed-anchor import.
- `contextual_inputs`: builds instruments, and loads collections by replaying their raw responses along the same paths of `question_flow` that collection asked.
- `public_frontier`: turns a public study's replayed captures into setup inputs.
- `contextual_setup`: each kept respondent's selection of final answers and the factors of the taxonomy parents' pools, which both setup compilers read, and the setup of a fixed-anchor collection.
- `contextual_node`: compiles one node's answers at a fixed coordinate into its admitted incident law under one `CompileOptions`, and a leaf's EAL.
- `contextual_compilation`: the frequency-estimate and large-loss-anchor exhibits of a contextual setup.

**4. `catmodel.calibration | catmodel.evidence | catmodel.forecast | catmodel.simulation`.** Four packages that do not import each other.

`catmodel.simulation` compiles incident laws and simulates them.

- `incident`: compiles a joint state's laws into marked incident laws and its conditional key.
- `incident_runtime`: simulates one joint state's incidents under the frozen controls, and decides each view's availability from the compiled laws' structure.
- `incident_linear`: event-addressed draws of single-member incidents, and the linear view coefficients of a mark's components.
- `incident_moments`: analytic means and proofs of zero loss.
- `small_events`: sums a year's small incidents by their exact first two moments.
- `capability_mixture`: integrates conditional annual losses over joint states and computes mixture statistics.
- `tail_gate`: the convergence gate ([The tail gate](#the-tail-gate)).
- `execution_budget`: counts simulation work and identifies each conditional simulation.
- `components`: a component's weight in each loss view, its positive support, episode lab-loss caps and defense-loading states.
- `authored`: the loss terms only hand-authored setups use, payment kernels (`payments`).

`catmodel.forecast` holds the public-frontier forecast, its data, its backtest and its trajectory quadrature, and `catmodel.calibration` the known-answer questions and the respondent spread fit. `catmodel.evidence` holds the bundle manifest, its snapshot and its validation (`source_bundle`), and the builder of a source bundle from a recipe (`source_inputs`). The builder also writes `provenance.yaml`, its record of the commit it ran at and the digests of `assumptions.md`, the hierarchy contract and the recipe. Nothing authenticates that record, so intake cannot tell a built bundle from one assembled by hand.

**5. `catmodel.elicitation`.** What respondents are asked and how their answers are read.

- `question_text` and `contextual_risk`: render the questions respondents read and validate their answers.
- `question_flow`: every kind of question with its parser, response model and remaining turns, and each leaf path's sequence: which follow-up an answer triggers.
- `contextual_frequency` and `chain_law`: the direct and event-chain frequency judgments and their pool.
- `epistemic`: the count law read from a respondent's five quantiles.
- `large_loss`: the large-loss anchor's share and tail band, and the second point of the two-anchor comparison.

**6. `catmodel.contracts`.** The typed contracts every stage shares.

- `run_setup`: `RunSetup`, its scope and the setup manifest. `_setup_codec` and `_setup_validation` encode `setup.json` and check the setup's inputs against each other.
- `law_overrides`: the laws each alternative puts in place of reference laws, held in memory for an authored setup and in the alternative's own file for a saved one, which reads one arm at a time.
- `incident`: incident laws, joint states, alternatives, comparison requirements and the `ExecutionPlan`, with its default controls.
- `contextual_risk`: instruments, answers and the collection artifact.
- `results` and `output`: `NativeResults`, every field of both results files typed, its schema version and reason codes, and the export index. `results_schema` generates the results files' JSON Schema documents from those types.
- `native_severity`: the stored form of the severity laws.
- `shared_fields`: the base of the frozen contracts a setup holds in bulk, whose instances share one set of field names to save memory.

**7. `catmodel.probability`.** Laws and numerical kernels that know nothing of scenarios or studies.

- `distributions`: the severity laws (a quantile-spline body, a GPD tail and caps) and their numerical domain.
- `percentiles`: the weighted quantiles of a loss sample from one sort, the kernel of every VaR and median.

**8. `catmodel.config | frozen_calls | scenarios | worker_pool | worktree`.**

- `config`: model profiles, collection settings and provider credentials.
- `frozen_calls`: strict replay of stored provider calls, and the parser identity each records.
- `scenarios`: the scenario library. `taxonomy_identity` tracks the roster's lineage, `base_rates` reads the source cards, and `threat_models` holds the threat-model contract, its register and the research brief.
- `worker_pool`: worker processes whose results do not depend on their number.
- `worktree`: the running code's identity, a Git commit and tree state or the installed version.

**9. Foundations.**

- Files and identities: `artifacts`, `hashing`, `json_io` and `yaml_io` (the JSON and YAML readers, which refuse a key repeated within one mapping), `path_safety`, `resources`, `output_paths` and `stored_files`, which holds files by name and leaves the large ones on disk, each read only while it has its recorded digest.
- Shared vocabulary: `identifiers`, `dateutil`, `economic_loss` (monetary bases and welfare weights), `assumption_vocabulary` and `logging`.
- `linear_algebra_threads`: one BLAS thread per process, set before NumPy loads. The CLI calls it as it is imported, and a script that loads NumPy without the CLI calls it first.
- `llm`: the provider transport over LiteLLM.

**10. `catmodel.validation_errors`.** Stable codes and locations for invalid answers. A retried question's feedback names them, never the error message.

Three narrower contracts in `pyproject.toml` add to the layers:

- Only the CLI imports click.
- Only `catmodel.collection` uses the provider transport: no other package imports `catmodel.llm`, LiteLLM or a provider SDK.
- The `capability-*` CLI module and `catmodel.collection` do not import the source-bundle builder.

## Cross-cutting ideas

### Identities and hashing

Most identities are SHA-256 digests of canonical JSON: sorted keys, compact separators and no NaN (`catmodel.hashing`). No identity records when an artifact was made or how many workers made it, so the same inputs, code and environment give the same identities. The environment a setup's identity covers is the code's provenance and what can change a number (`catmodel.setup.store.setup_code_provenance`); the scientific fingerprint covers none of it.

| Identity | What it covers | Where it is recorded |
| --- | --- | --- |
| Source bundle | `manifest.yaml`, which lists each file's SHA-256 and source provenance. `provenance.yaml` stays outside it. | Each instrument (`source_bundle_manifest_sha256`). |
| Instrument | The whole frozen instrument, respondents included. | The collection plan and artifact (`instrument_sha256`). |
| Frozen call | Its key (the call ID and the respondent), its prompt digest and its parser protocol: the parser's name, the response schema and the parsing code. | `calls/`, `acquisition/` and `parser-protocols/` in the collection directory. |
| Setup | `setup-manifest.json`: the code provenance and each file's checksum. | The capture (`originating_setup_sha256`). |
| Scientific fingerprint | The setup's scientific inputs, without disclosure prose, the setup ID or the execution plan. | The capture (`originating_scientific_sha256`) and each result's `analysis_spec`. |
| Conditional key | One joint state's compiled laws under one arm, with its populations, entities, period and deployments. | The capture, for each arm and state. |
| Capture | `run-capture.json` without its own digest. Each array file is named by its SHA-256. | Each result (`content_sha256`). |
| Result | `analysis_spec`: the scientific and plan digests, the capture's provenance, the target weights, the tail levels and the reducing code. | `analysis_sha256` in `results.json`. |
| Reason | The first 12 hexadecimal digits of the SHA-256 of its text. | The `reasons` table of each results file. |

Two arms whose state has the same conditional key, years and controls share one simulation of it.

### Replay re-derives and compares

A digest proves that bytes did not change, not that they are right, so each reader re-derives what it can and refuses a mismatch.

- **Collections.** `capability-risk validate` and `setup import` run every stored raw response through the current parser and refuse an answer that the replay does not reproduce. Collection and replay share one test of a failed validation attempt (`frozen_calls.validated_response`), so collection never stores an attempt that replay cannot re-derive.
- **Setups.** Building a setup decodes the files it is about to write and compares them with the composed inputs. Loading one checks every checksum and refuses untracked files.
- **Captures.** `results replay` and `results export` check the capture's digest and file list, recompile each state's laws from the setup and refuse a conditional key that differs. They also re-derive each state's support, analytic EALs, work counts, small-event splits and tail gate rounds; a closed-form EAL must match to a relative 10⁻⁹.
- **Results.** `load_results` refuses another schema version, a file that does not match its JSON Schema document (`contracts/results/`), and an `analysis_sha256` that is not its `analysis_spec`'s digest. Loading an export requires its results to reproduce from its capture, each number within a relative 10⁻⁹.
- **Versions.** A reader refuses a saved file whose version marker (`schema_version`, `bundle_schema_version`, or `format` in `export.json`) is absent or not exactly the current value (`json_io.require_marker`).

### Determinism

Every step but collection is deterministic.

- From one commit and its committed inputs, the bundle, forecast, study plan and frozen prompts rebuild byte for byte.
- Every simulation draw comes from a Philox stream keyed by a hash of the seed, the joint state, the risk unit (usually an origin) and a named purpose (`catmodel.probability.keyed_streams`). A year's draws therefore depend only on the plan seed, the state and the year, not on batch size, the requested views, other origins or the number of years simulated.
- Each joint state's seed derives from the state's ID and the plan seed, which import fixes at 42, so two imports of the same captures simulate identical reference draws.
- Each year's count is drawn by inversion from one uniform (`catmodel.probability.count_inversion`), so an arm that raises a rate draws counts comonotone with its control's, and arm effects are paired.
- On the same platform and dependency versions, the setup, the capture and the results do not depend on the worker count (`--workers`), nor a collection's frozen files on its number of collector processes.
- Across CPU types and BLAS builds, the last bits of long sums can differ, which export tolerates at a relative 10⁻⁹. A decision that compares a computed value with a threshold uses a tie tolerance or the stated decimal (`catmodel.probability.numerics`).
- Collection is not deterministic: providers do not guarantee repeatable sampling. It takes no seed, and the order of requests never reaches a capture.

### The tail gate

The tail gate (`catmodel.simulation.tail_gate`; [`NUM-TAIL-GATE`](assumptions.md#num-tail-gate)) decides whether a tail statistic can be quoted. After each round, an arm whose driver views have not converged doubles its years per state, until its state-years would pass its cap; a statistic still unconverged is reported `withheld`. The capture records every round, and replay checks them ([methodology](methodology.md#numerical-execution-and-outputs)).

### Forecast quadrature and conditioned questions

`capability-forecast public-study` fits the public-frontier forecast (`catmodel.forecast.public_frontier`) and integrates over the frontier's progress with a fixed rule (`catmodel.forecast.public_frontier_quadrature`; [trajectory quadrature](methodology.md#trajectory-quadrature)). Each node gets one instrument per source status, and its path conditions every question at that node: `catmodel.elicitation.question_text` renders it as a dated table of each lab's ECI. `setup import --study` makes one joint state per node and respondent, weighted by the node's weight. `public-frontier.json` freezes the nodes and weights, and import refuses a study whose rule it cannot reproduce.

### Governed constants and the assumption register

- A governed constant lives in the module that uses it, with its evidence status in its docstring or comment and, for most, the [assumption register](assumptions.md) entry that governs it: `DEFAULT_YEARS_PER_STATE` in `catmodel.contracts.incident` cites [`NUM-BUDGETS`](assumptions.md#num-budgets).
- Authored judgments live in data files, such as the tail classes and their shapes in `scenarios/severity_governance_policies.yaml`.
- `catmodel.setup.sensitivity_catalog` maps register IDs to the comparison families that challenge them, with the reason whenever a required challenge is unavailable. `tests/test_assumption_register_sensitivity.py` keeps the catalog and the register in step.
- Every source bundle holds a register of its own, `assumptions/assumption_register.yaml`, whose list may be empty ([runbook](runbook.md#1-evidence-and-scope)).

### Write-once outputs

`catmodel.artifacts` publishes each file whole or not at all, and never replaces one. Most commands stage a new directory beside its destination and rename it into place only when it is complete. `catmodel.output_paths` refuses an output inside an existing bundle, setup, capture or results directory.

## Where to start

### Adding a scenario

Follow [Adding a scenario](docs/adding-a-scenario.md). A new scenario has no answers, so a study must be collected again to include it.

### Changing a law

[From the equation to the code](#from-the-equation-to-the-code) says where each law lives.

1. Keep the analytic means in `catmodel.simulation.incident_moments` consistent with the law. Results report the analytic EAL, and replay re-derives it.
2. Test against an independent oracle ([TESTING.md](TESTING.md)).
3. A material numerical change needs a before-and-after comparison ([change controls](model-governance.md#change-controls)).

A change that moves an analytic mean makes older captures fail replay, so their setups must be run again.

### Adding a results field

1. Declare the field in its TypedDict in `catmodel.contracts.results` with its type, `NotRequired` if a report can lack it, and a `Literal` for a status or other value from a fixed set. A record is closed, so a field missing from its TypedDict fails the schema checks below.
2. Choose its file by its section. `results.json` holds the headline, the sections of `HeadlineResults`. `tests/test_results_schema.py` pins its size, so add a field there only when a reader of headline numbers needs it. A report-only analysis is an exhibit, a section of `ResultsExhibits`, and a value that only view diagnostics need goes in the `diagnostics` section; `results.diagnostics.json` holds both.
3. Name a reason key `reason`, or end it in `_reason` or `_reasons`. The results file then stores each reason as a code in its `reasons` table.
4. Raise `RESULTS_SCHEMA_VERSION` in the same contracts module. Any change to the fields, types or meaning of either results file raises it, an added field included.
5. Regenerate the files' JSON Schema documents in `contracts/results/` with `uv run python scripts/generate_results_schemas.py` and commit them. `tests/test_results_schema.py` fails until they match the contracts.
6. Name the operands of a new difference or standard error in the tolerance tables of the export (`_scale_of` in `catmodel.results.results_export`), which compare an export's stored results with a replay, number by number, when the export is read.
7. Say in the pull request what changed. The regenerated schemas are each field's reference; [Results files](docs/setup-run-results.md#results-files) names only the sections, so add a new section there.

### Adding a CLI option

1. A `setup import` option that only public studies take belongs in `_PUBLIC_STUDY_ONLY` in `_workflow`, so `--run-directory` refuses it.
2. A value that changes what is computed must reach the setup or its plan, and so their identities. An operational value, such as the worker count, must not.
3. Update the command map in the [workflow map](docs/workflows.md) and the [runbook](runbook.md). Test the option through the CLI, as `tests/test_workflow_cli.py` and `tests/test_contextual_risk_cli.py` do. `tests/test_cli_stages.py` pins every command and option name, so an addition is a deliberate change there too.

## Tests

[TESTING.md](TESTING.md) sets the testing policy and lists the repository checks. Tests are in `tests/`, and their helpers and independent oracles in `test_support/`.
