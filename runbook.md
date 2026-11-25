# Model Runbook

This page gives the commands for each step of a study: capture a source bundle (section 1), plan context-bound questions about one named lab, AI model and deployment (2), collect and validate the answers (3), import a setup and run it (4), derive results from the capture (5) and report them (6). The [workflow map](docs/workflows.md) lists every command, and [setup, run and results](docs/setup-run-results.md) describes the files. Collections take their respondent models from a [model profile](#model-profiles); for development, use the packaged `--profile test` or `--profile dev`.

## 1. Evidence and scope

Freeze upstream inputs once, from a clean committed checkout:

```sh
uv run --frozen catmodel package build-source-bundle \
  --recipe <recipe.yaml> --output <new-directory>
```

The recipe's `files:` map names each bundle path and the file beside the recipe that it captures. Each `sources:` entry names a captured file (`snapshot_path`) and a YAML file beside the recipe (`metadata_file`) with the source's provenance: URI, vintage, transformations, loader, `qa_status` and `license_access_basis`. The bundle's identity is the SHA-256 of `manifest.yaml`, which every instrument binds. `provenance.yaml` records the commit and input digests outside that identity, and nothing authenticates it. A record cites a captured file as a hash-pinned `evidence_file:` reference, which intake checks. Elicitation output never belongs in a bundle.

A bundle holds:

- `evidence/source_cards.yaml` (`SourceCardsFile` in [`base_rates.py`](src/catmodel/scenarios/base_rates.py)): the dated excerpts shown to respondents and, for every scenario, the cards its questions show. Each card cites a source in `sources`, as an https URL or a citation that names its year, and holds only what its sources report.
- `assumptions/assumption_register.yaml`, whose list may be empty.
- Optionally `evidence/threat_models.yaml`, the [threat-model](docs/threat-models.md) register, which the scenarios' questions then quote after their source cards.

The hypothetical-lab study's bundle copies [`data/elicitation_source_cards.yaml`](data/elicitation_source_cards.yaml) and [`data/threat_models.yaml`](data/threat_models.yaml). Its assumption register is empty: its assumptions are in [`assumptions.md`](assumptions.md) and its stipulations in the study's own files.

Scenarios and the taxonomy belong to the repository, not to a study ([adding a scenario](docs/adding-a-scenario.md)); a changed leaf population needs newly collected answers.

## 2. Context-bound lab/deployment questions

For one named lab and deployment, write `context.json` with the `ContextualRiskInstrument` fields except `respondents` and `source_bundle_manifest_sha256`, which the builder supplies.

- **Anchors.** The anchors must cover the integer range and include the exact reference score: answers are interpolated between anchors and never extrapolated.
- **Scales.** Optional `path_linked_exposure` lists deployment scales, each with an as-of level and a study-year median-path level that may not be lower, and a `max_growth_multiple` above 1. `public-study` renders each `{scale_id}` placeholder in `use_population` per trajectory node (`POP-HYPOTHETICAL-LAB`). A context with fixed scales states them in `fixed_exposure`. Every scale that an event chain's base count reads must be stipulated.
- **Source statuses.** A context for a public study maps `public` and `not_yet_public`, in `source_statuses`, to the same subset of `model`, `deployment`, `deployment_binding`, `capability_access`, `exposure`, `fixed_controls` and `fixed_adoption`. It may omit `capability_scale`, `capability_domain_id`, `reference_eci`, `integer_min`, `integer_max` and `anchor_scores`, which `public-study` takes from the forecast; `build-instrument --source-status` needs them stated.
- **Scenarios.** List `nodes` in the context or repeat `--scenario`; `--scenario all`, alone, selects the complete catalog. More than one node needs an `independent_node_origins_rationale`.

```sh
uv run --frozen catmodel --profile test capability-risk build-instrument \
  --source-bundle source-bundle --context context.json \
  --scenario all --output contextual-instrument.json
uv run --frozen catmodel --profile test capability-risk collect \
  --instrument contextual-instrument.json --source-bundle source-bundle \
  --run-directory contextual-run --dry-run
```

A dry run calls no provider and freezes every question in `calls/*/request.json`; a live collection omits `--dry-run`. Collection, validation and import rebuild each repository scenario's node from the current YAML and refuse one whose population, counted unit or event chain differs.

For each node, coordinate and respondent, collection asks the direct question and the event-chain question in separate calls. The complete catalog also asks about each of the <!-- scenario-count:parents -->14<!-- /scenario-count --> taxonomy categories above an estimated scenario: <!-- scenario-count:questions -->78<!-- /scenario-count --> initial questions per coordinate and respondent (<!-- scenario-count:cells -->32<!-- /scenario-count --> direct, <!-- scenario-count:cells -->32<!-- /scenario-count --> chain and <!-- scenario-count:parents -->14<!-- /scenario-count --> parent). Some answers trigger a follow-up call, and every call has up to three validation attempts ([methodology](methodology.md#event-populations-and-frequency)): budget about one and a half times the initial calls. A call that exhausts its attempts is recorded and collection goes on.

Validate the capture (section 3), then import it:

```sh
uv run --frozen catmodel setup import \
  --run-directory contextual-run --setup-id contextual-example \
  --output contextual-setup
```

This fixed-anchor import gives conditional what-if results, with a kernel at every modeled integer score and no comparison families or financial view ([captured evidence](docs/setup-run-results.md#captured-evidence)).

### Public-frontier trajectory studies

The main pipeline asks every question along trajectories of a forecast of the public frontier. Plan a study from a source bundle, a `PublicFrontierForecast` JSON and a context:

```bash
uv run --frozen catmodel --profile test capability-forecast public-study \
  --source-bundle source-bundle \
  --forecast public-frontier-forecast-template.json \
  --context context.json --scenario all \
  --assumptions public-assembly-assumptions.json \
  --output public-study
```

- **Forecast.** One general-ECI coordinate per named lab, all on one frozen scale ([fields and defaults](docs/capability-frontiers.md#joint-public-frontier-forecast)); set a `pace_cap` other than the default before planning. A forecast without `releases`, such as the [committed template](data/capability/public-frontier-forecast-template.json), takes them from Epoch's ECI table in the bundle, whose provenance must give its URI, retrieval time and CC BY license basis. A forecast that states its `releases` is used as authored: include each lab's leader already available before `trend_window_start`, and a release at or before as-of for every lab. You answer for each release's eligibility, date, score, benchmark error, license and credit, and for the `scale_anchors` and any `scale_correspondence`.
- **Epoch table.** To capture it, copy the [committed snapshot and its provenance file](data/capability/README.md) beside the recipe and add:

  ```yaml
  files:
    evidence/epoch_eci_scores.csv: epoch_eci_scores.csv
  sources:
    - snapshot_path: evidence/epoch_eci_scores.csv
      metadata_file: epoch_eci_scores.provenance.yaml
  ```

- **Context.** `source_statuses` describes the lab's public and not-yet-public source populations, and the command renders an instrument template for each. A source model's status is fixed when the incident originates, wherever the model is used. The policy period must start on or after the forecast's as-of.
- **Assumptions.** A JSON with nonblank `exposure_rationale`, `residual_independence_rationale` and `parameter_coupling_rationale`.
- **Nodes.** Seven interior trajectory nodes by default; `--nodes` takes another odd count of at least three. A no-progress node is added, so the default study has 16 contexts ([trajectory quadrature](docs/capability-frontiers.md#trajectory-quadrature)). A study that will have [variant studies](#hypothetical-frontier-lab-study) needs 7, 11, 15 and so on.

The command writes `public-study.json`, `forecast.json`, `trend-posterior.json`, `public-frontier.json`, `forecast-comparison.json` and the paired instruments.

Check the fit before collecting: each lab's `likelihood_release_ids` and `as_of_value` and the discrepancy in `trend-posterior.json`, and in `forecast-comparison.json` the leave-one-out residual RMS, which should be near one, and the staleness entries, which flag labs far behind their trend. `uv run --frozen python scripts/backtest_public_frontier.py public-study/forecast.json pace-backtest.json` scores candidate pace shapes; adopt another `pace_shape`, citing the file in `pace_rationale`, only when its loss difference exceeds about two standard errors.

Collect every context with one command, first with `--dry-run` and then without it (section 3), and import the study:

```bash
uv run --frozen catmodel --profile test capability-risk collect \
  --study public-study/public-study.json --source-bundle source-bundle --dry-run
uv run --frozen catmodel setup import \
  --study public-study/public-study.json --setup-id public-frontier \
  --output public-frontier-setup
```

Import freezes every catalog comparison family as a requirement and executes the default set ([context-bound comparisons](docs/setup-run-results.md#context-bound-comparisons)); repeat `--comparison-family <family>` for another selection, which replaces the default, or pass `--comparison-family all`. Each simulated arm adds at most one reference-equivalent run, so check `setup inspect` first. Results stay `partial` while a family is unselected or unavailable.

**Refining.** The run's `trajectory_quadrature` summary compares each EAL with the embedded rule, which has `(n − 1)/2` of the study's `n` nodes ([methodology](methodology.md#trajectory-quadrature)). A difference well beyond respondent-sampling noise calls for more nodes, and a noisy one for more samples per node. To refine, plan the same inputs at `2n + 1` nodes into a new directory: every earlier context reappears byte for byte. Collect with `--answers` naming an [answer table](#answer-table) that holds the first collection's responses, and only the new nodes' questions are asked.

#### Hypothetical frontier lab study

[`config/studies/hypothetical-frontier-lab/`](config/studies/hypothetical-frontier-lab/) stipulates the study's lab (`POP-HYPOTHETICAL-LAB`): `context.json` defines the lab, its deployments and their path-linked scales, and `public-assembly-assumptions.json` holds the assembly rationales. The context's period cutoffs equal the forecast template's as-of date, so refresh both together. [`authored-source/`](config/studies/hypothetical-frontier-lab/authored-source/) holds the bundle's recipe and assumption register. The recipe's paths resolve beside it, so stage it in one directory with copies of `data/capability/epoch_eci_scores.csv`, `data/capability/epoch_eci_scores.provenance.yaml`, `data/elicitation_source_cards.yaml` and `data/threat_models.yaml` before building the bundle.

By hand, plan with `public-study` at `--profile example` and `--nodes 15`, with the committed forecast template, this directory's context and assumptions, and `--scenario all`: 32 contexts. Collect, validate and import. Lab loss is one of the two reported measures, so select the lab-loss families too; naming any family replaces the default selection, so the command names the defaults as well:

```bash
uv run --frozen catmodel setup import \
  --study public-study/public-study.json --setup-id hypothetical-frontier-lab \
  --comparison-family public_forecast \
  --comparison-family model_pooling_leave_one_out \
  --comparison-family tail_shape --comparison-family uncapped_severity \
  --comparison-family exposure_scale --comparison-family safeguard_level \
  --comparison-family lab_loss_cap --comparison-family lab_loss_provider_only \
  --output hypothetical-frontier-lab-setup
```

Two default families need variant studies. `exposure_scale` halves and doubles every path-linked scale, and `safeguard_level` (`POP-SAFEGUARD-LEVEL`) replaces the stipulated safeguards with the `weaker` or `stronger` level of the context's `safeguard_levels`. Each variant is its own public study on the headline study's embedded rule, at three samples per model:

1. Plan it with the same `public-study` command plus `--variant half`, `double`, `weaker` or `stronger`, into its own directory, with `--nodes` at `(n − 1)/2`: 7 for a 15-node study. The default, 3, suits the default 7.
2. Collect its whole plan with `collect --study`, one study at a time, because each collection takes every model's full ceiling.
3. Validate each capture and add `--variant-study <directory>/public-study.json` to the import.

Import refuses a variant study whose forecast, period, nodes or instruments do not match the headline study's. It measures each variant arm against the headline captures reweighted to the embedded rule (`exposure_scale::embedded_control`, `safeguard_level::embedded_control`).

#### Running the hypothetical-lab study

`scripts/run_hypothetical_lab_study.py` plans, dry-runs or collects the study end to end. As configured, the study is the 15-node headline study with the two safeguard-level variant studies, at the `example` profile:

```sh
uv run --frozen python scripts/run_hypothetical_lab_study.py ../hfl-dry-run --variant weaker --variant stronger --dry-run
uv run --frozen python scripts/run_hypothetical_lab_study.py ../hfl-run --variant weaker --variant stronger --live
```

Exactly one of `--dry-run` and `--live` is required. A dry run builds the bundle (`source-bundle/`), plans the headline study (`public-study/`) and each variant study (`safeguard-weaker/` and so on), and freezes every initial prompt (`*/collections/*/calls/*/request.json`) without calling a provider. A live run also collects the answers, one study at a time, validates every context, imports a setup with `--comparison-family all` and the variant studies (`setup/`), runs it (`model-run/`), and writes `summary/` and `export/`.

| Option | Meaning |
| --- | --- |
| `--variant NAME` | Add a variant study (repeatable): `weaker` or `stronger` (in `safeguard-weaker/`, `safeguard-stronger/`), `half` or `double` (`exposure-half/`, `exposure-double/`). The import records a variant that is not named as unavailable. |
| `--scenario ID` | Ask about this scenario only (repeatable; default: the complete catalog). A selection short of the complete catalog asks no taxonomy parent question. |
| `--nodes N` | The headline study's interior nodes (default 15): 7, 11, 15 and so on with `--variant`, and any odd number from 3 without. |
| `--profile NAME`, `--profiles FILE` | The [model profile](#model-profiles) (default `example`). Every step asks every model of it. |
| `--source-bundle DIR` | Use an existing bundle instead of building one. |
| `--answers DIR` | Give every collection an [answer table](#answer-table). |
| `--resume` | Continue an earlier run of the same request. |

The script needs:

- A clean committed Git clone installed in place, with `uv sync --frozen` or `pip install -e .` (then `python` replaces `uv run --frozen python`), and a new output directory outside it: the bundle build refuses a dirty tree.
- For a live run, an API key for every model of the profile, in the environment or in a `.env` file in the working directory (`.env.example` lists the variables). The provider access must allow risk-assessment questions about chemical, biological, radiological and nuclear weapons, attacks on critical infrastructure and other catastrophic misuse. A provider may still refuse some answers: collection goes on, and import leaves a model out of a scenario it does not answer ([section 4](#4-build-a-setup-and-run-it)).

**Size.** The 15-node headline study has 32 contexts, in each of which the example profile's 15 respondents (three models, five samples) answer 78 initial questions: 37,440 initial calls. Each variant study adds 16 contexts with 9 respondents: 11,232 initial calls. The dry run above therefore freezes 59,904; allow about a quarter of an hour on four cores and 4 GB of disk. A live run sends about one and a half times as many calls (section 2), less what an answer table holds, and its setup needs more memory than a default one ([Choosing the worker count](#choosing-the-worker-count)).

**Resuming.** Rerun the same command with `--resume`; with `--live` it also continues a dry run's directory. Collections resume from their stored calls (section 3). Any other step is skipped when its output exists, validates and was built from the same request, which a record beside it holds (`<output>.request.json`). A changed request, such as another `--nodes`, an added variant or a changed committed input, needs a new output directory. Move aside an output that does not validate.

**What repeats.** Every step but collection is deterministic ([determinism](ARCHITECTURE.md#determinism)); providers do not guarantee repeatable sampling.

##### A small study

A few scenarios on three trajectory nodes, asked once of one model, are planned, collected, imported and run with the same commands. Write a profiles file whose profile has one model and `samples_per_question: 1` ([Model profiles](#model-profiles)), and build a source bundle as above or reuse the one a dry run of the script wrote. Then:

```sh
export CATMODEL_PROFILES=<profiles.yaml>
uv run --frozen catmodel --profile <profile> capability-forecast public-study \
  --source-bundle <source-bundle> \
  --forecast data/capability/public-frontier-forecast-template.json \
  --context config/studies/hypothetical-frontier-lab/context.json \
  --scenario <scenario-id> --scenario <scenario-id> \
  --assumptions config/studies/hypothetical-frontier-lab/public-assembly-assumptions.json \
  --nodes 3 --output <study>
uv run --frozen catmodel --profile <profile> capability-risk collect \
  --study <study>/public-study.json --source-bundle <source-bundle> --dry-run
uv run --frozen catmodel --profile <profile> capability-risk collect \
  --study <study>/public-study.json --source-bundle <source-bundle>
uv run --frozen catmodel setup import \
  --study <study>/public-study.json --setup-id <setup-id> --output <setup>
uv run --frozen catmodel run --setup <setup> --output <run>
uv run --frozen catmodel results summary \
  --results <run>/results.json --output <summary>
```

Three nodes and the no-progress node make eight contexts, so two scenarios ask 32 initial questions of one respondent. The first `collect` freezes them for reading and the second sends them. With one model and one sample, the leave-one-out comparison and the respondent-sampling standard errors are unavailable, and a three-node study cannot have variant studies.

`scripts/run_hypothetical_lab_study.py` runs the same chain in one command: `--scenario ID`, which repeats, selects scenarios, and without `--variant` it accepts `--nodes 3`. `scripts/offline_demo.py` runs it offline on synthetic answers.

#### Conditioning pilot

An optional check, before a full collection, of whether respondents' answers move with the capability trajectory, and with what in it ([limitation 6](docs/limitations.md#6-the-capability-forecast-rests-on-few-releases)). `scripts/conditioning_pilot.py` asks a few scenarios on a small node rule in three arms: `reference`, each question as planned; `fixed_scale`, every path-linked deployment scale at its median-path level on every node; and `placebo`, the trajectory table's lab columns in reverse order, which carries no information.

```sh
uv run --frozen python scripts/conditioning_pilot.py plan ../pilot --source-bundle ../hfl-dry-run/source-bundle --context config/studies/hypothetical-frontier-lab/context.json --scenario fraud_individuals --scenario output_reliance_software
uv run --frozen python scripts/conditioning_pilot.py collect ../pilot --dry-run
uv run --frozen python scripts/conditioning_pilot.py collect ../pilot --live
uv run --frozen python scripts/conditioning_pilot.py analyse ../pilot --output ../pilot-analysis
```

- `plan` writes a new directory without calling a provider: a public study on `--nodes` interior nodes (default 3) and `node-00`, every arm's instruments and `pilot.json`. `--forecast`, `--profile`, `--profiles` and `--samples` change its inputs. The plan is never imported.
- `collect` requires `--dry-run` or `--live` and behaves like any contextual collection (section 3); `--arm` collects one arm, and `--answers` uses an answer table. At the default rule and the `example` profile's 15 respondents, each arm makes 240 initial calls per scenario, or 720 per scenario for the three arms, so a six-scenario pilot freezes 4,320 initial calls, before follow-up calls and retries.
- `analyse` replays the captures and writes `pilot.md`, which defines every statistic and verdict, and `pilot.csv` into a new directory.

The verdicts are screens chosen as reading aids, not tests. The noise floor is the standard deviation of the log answer between one model's samples on one node. At the default rule with 15 respondents and the same noise in every group, the fitted change over the range has a standard error of about 0.35 noise floors and the placebo log ratio one of about 0.18; more samples per model (`--samples`) tighten a verdict.

## 3. Collection settings and validation

Collection is blind: the collectors use no web search, so models rely on the evidence in the prompts.

**Concurrency.** The profile's `max_concurrency`, or a provider's entry in `max_concurrency_by_provider`, is each model's ceiling of requests in flight, shared by every context of one `collect` run. A rate limit (a 429 or an exhausted token budget) lowers the model's limit by 10%, and later successes raise it back.

**Transport.** Every provider is called through LiteLLM (`PROVIDERS` in `src/catmodel/llm/litellm_provider.py`). OpenAI runs as a background Responses run with `store: true`, so the response is stored under the account's retention terms; the other providers are streamed. A lost background run or a cut stream is sent again and recorded as `connection_lost` (`NUM-CONNECTION-LOST`).

**Failures.** A request that fails after the transport's retries is frozen as an attempt with no response (`NUM-PROVIDER-FAILURES`):

- A transient failure (a lost connection, a 408, 429 or 5xx, or a deadline) fails only its call's (coordinate, respondent) group. The run goes on and ends nonzero, naming each incomplete context and failed call.
- A systematic failure (another client error, such as a 401) stops the run at once.
- A content-policy rejection is a `refusal`, which counts as a failed validation attempt.

**Resuming.** Fix the cause and rerun the same command in the same run directories: each failed call is sent again with the same prompt. A context that holds a call with three provider failures needs a new run directory, so wait out a sustained outage before rerunning. To collect one context afresh, move its capture directory (the plan's `capture` path, under `collections/`) aside. A changed profile or profiles file, model, reasoning effort, output ceiling, deadline, output budget, response schema or parsing code needs a new run directory; concurrency caps may change.

**Interrupting.** Send Ctrl-C or SIGTERM to the `catmodel` process, not to its collector processes: each collector then cancels its open background runs, which would otherwise keep generating and be billed. A run left open by a killed process can be cancelled with `POST /v1/responses/{run_id}/cancel` and the JSON body `{}`; the log gives each run id at `server-side run submitted`.

**Validating.** Replay every stored response offline:

```sh
uv run --frozen catmodel capability-risk validate --run-directory contextual-run
```

A response must be the answer's JSON object and nothing else, apart from one Markdown code fence around it. Validation checks that answers follow the contract, not that they are right. The report gives:

- `call_limits`: each model's output-limit stops, refusals, provider failures and `connection_lost` re-sends, and its queue and generation times as shares of the deadline (`NUM-CALL-LIMITS`). An output-limit stop, or a generation time near the deadline, calls for a higher limit in a new run directory.
- `unanswered_cases` and `exhausted_calls`; the raw attempts under `calls/` show the rule each response broke, or the provider's `stop_reason`.
- `unanswered_scenarios`: each leaf that a model left unanswered in more than 10% of the collection's samples, which import may leave the model out of ([section 4](#4-build-a-setup-and-run-it)).

**Output ceilings.** A provider that rejects a model's `max_output_tokens` fails that model's first call and stops the run. To check beforehand, with one live call per model:

```sh
uv run --frozen python scripts/probe_output_ceilings.py --profile test
```

`--model MODEL_ID` (repeatable) probes selected models. Probe a Google, Meta or OpenRouter model before its first collection: no example profile uses those providers, and their requests are tested only offline.

### What bounds throughput

The lowest of three bounds sets a collection's rate:

- **Each model's ceiling.** A model with ceiling `c` and a typical answer time of `t` seconds answers about `c / t` requests a second, and a collection ends when its slowest model has answered.
- **The provider's rate limits**, which lower a model's limit below its ceiling.
- **The collector's CPU.** One collector process serves its requests on one CPU core, and LiteLLM parses every streamed chunk in Python. Allow about 0.5 CPU-seconds per answered attempt, or about 120 attempts a minute per core.

`catmodel --workers N capability-risk collect --study` therefore collects in up to N collector processes (default: the CPUs available), dividing the contexts and each model's ceiling among them. The frozen files do not depend on N.

To size a collection:

1. Estimate the cores the ceilings need: the sum over the models of `c / t`, times the CPU-seconds per attempt.
2. Run that many collectors plus a quarter. The example profile at its ceilings needs five or six.
3. With fewer cores, lower the ceilings: above what the cores serve, a higher ceiling adds failures, not answers.
4. Watch the log in the first minutes. `collector bound by its CPU core, not its ceilings` means that a higher ceiling adds no answers. Repeated `event loop late` warnings, or `transient provider failure; retrying` warnings that name a connection timeout, mean that exchanges are failing for lack of CPU.

Memory grows with the requests in flight and the contexts being collected: allow about 6 GB for the 15-node headline study at the example profile's ceilings with four collectors, and about 0.4 GB for each further one.

### Model profiles

Every command that builds or collects an instrument reads its respondent models and collection settings from one profile of a profiles file. `catmodel --profile` names the profile. The file is the one that the `CATMODEL_PROFILES` environment variable names, else AIUC's example, [`data/example_model_profiles.yaml`](data/example_model_profiles.yaml). Its `example` profile is the hypothetical-lab study's panel: three frontier LLMs, from OpenAI, Anthropic and xAI, with five samples each. Other studies should bring their own file:

```yaml
profiles:
  mine:
    models:
      - provider: anthropic          # a ProviderName in src/catmodel/llm/models.py
        model_id: "claude-opus-5-5"  # unique in the profile
        reasoning_effort: high       # low, medium, high, xhigh or max
        max_output_tokens: 128000
    settings:                        # optional, as is each of its fields
      samples_per_question: 5        # 1 to 100
      max_concurrency: 20            # each model's requests in flight, 1 to 1,000
      max_concurrency_by_provider: {}  # a ceiling for one provider's models instead
      request_timeout_seconds: 600.0
```

Use it as `CATMODEL_PROFILES=my-profiles.yaml uv run --frozen catmodel --profile mine capability-risk collect ...`; the scripts take `--profile` and `--profiles FILE`. Loading validates every profile of the file (`ModelProfile` in [`config.py`](src/catmodel/config.py)), and every model of every profile needs `max_output_tokens`. An instrument asks every model of the profile at its `samples_per_question` (three in a variant study), and collection refuses an instrument whose models the profile lacks. `collection-plan.json` records the profile and, for a file other than the example, the file's path and SHA-256, so an edited file needs a new run directory.

### Respondent spread calibration

The calibration compares the respondents' count judgments with known outcomes ([methodology](methodology.md#respondent-spread-calibration)). It is a diagnostic: no setup reads its fit, and every study is imported at face value (`k = 1`, `b = 1`). `scripts/respondent_calibration.py` runs it from a source checkout, on the committed [seed package](data/calibration/respondent_spread/README.md):

```sh
uv run --frozen python scripts/respondent_calibration.py collect \
  --seeds data/calibration/respondent_spread/seeds.yaml \
  --run-directory calibration-run --profile example --dry-run
uv run --frozen python scripts/respondent_calibration.py fit \
  --run-directory calibration-run \
  --resolutions data/calibration/respondent_spread/resolutions.yaml \
  --output respondent-spread-fit.json
```

`collect` requires `--dry-run`, which freezes every prompt, or `--live`; `--profile` (default `example`), `--profiles` and `--answers` work as elsewhere. Use the profile of the study that the diagnostic is read beside, and collect again after any change of roster, reasoning effort, contract or output budget. `fit` checks each resolver script against its frozen digest, so run it from the checkout that holds the seed package; `--workers` does not change the fit.

### Answer table

A collection directory binds its responses to one instrument, roster and code revision. An answer table keeps the responses themselves, each keyed by its exact system and user prompt, by the provider, model, reasoning effort and output budget, and by the sample index, so they outlive those changes.

```sh
uv run --frozen catmodel --profile example capability-risk collect \
    --study STUDY/public-study.json --source-bundle BUNDLE --answers ANSWERS
```

With `--answers`, a request is sent only if the table holds no response for it, and every response a provider returns is added. A reused response is parsed and validated like a live one, and its attempt is marked `reused_from_answer_table`. After a change to one scenario's wording, a parser, the roster or the number of samples, a new collection asks only what the table cannot answer; a dry run prints how many frozen requests the table already holds. `catmodel answers add --answers ANSWERS COLLECTION...` adds the responses of earlier collections, including ones the current code can no longer import.

- A reused answer is not a new sample: a study that must be independent of an earlier one collects without the table.
- A refusal or an output-limit stop is reused like an answer.
- A provider can change the model behind an identifier; a row records only the identifier the provider reported.

## 4. Build a setup and run it

`setup import --study` is the main pipeline's intake and `setup import --run-directory` the fixed-anchor what-if intake (section 2). Inspect a setup, then run it:

```sh
uv run --frozen catmodel setup inspect --setup public-frontier-setup
uv run --frozen catmodel run --setup public-frontier-setup --output model-run
```

**Unanswered questions.** Import decides what an exhausted direct question leaves out ([pooling](methodology.md#pooling-and-parameter-uncertainty)). A model that lacks one scenario in more than 10% of its samples (`max_excluded_sample_share`) is left out of that scenario, and its samples take other models' answers to it (`POOL-UNANSWERED-SCENARIOS`). Any other sample that lacks a scenario is excluded (`POOL-EXCLUDED-SAMPLES`). A sample whose answers declare the lab's own loss unavailable is kept, and is left out of the lab-loss and gross views alone. Import refuses a study in which no model answers a scenario (collect with a model that does), a model does not answer more than half of the scenarios (use a profile without it), a model keeps fewer than half its samples at a node, or more than 10% of the samples are excluded. `setup inspect` reports the `excluded_samples` exhibit.

**Simulated years.** Import starts every state at 1,024 simulated years. The tail gate doubles an arm's years while its tails fail to converge, up to a cap on the arm's simulated state-years: 2,000,000 for the reference arm and 500,000 for each comparison arm that runs its own simulation (`NUM-TAIL-GATE`). A statistic still unconverged at the cap is `withheld`. A 1-in-N level needs about 160·N state-years of equal weight, so the reference cap reaches about 1-in-12,000 and the comparison cap about 1-in-3,000. The Python functions `import_public_frontier_study` and `import_contextual_setup` take other caps (`max_reference_state_years`, `max_comparison_state_years`), starting years (`years_per_state`), work allowances (`work_allowances`) and another `seed`.

**Work limits.** A state whose work exceeds its allowance per simulated year times its years aborts the run without a capture, naming the arm and state (`NUM-BUDGETS`). Import again with a larger `work_allowances`.

**Authored inputs.** `catmodel setup build --spec <inputs.json> --plan <plan.json> --evidence <directory> --setup-id <id> --output <new-directory>` freezes a hand-authored setup ([guide](docs/setup-run-results.md)).

A run writes `capture/`, `results.json`, its diagnostics file and `tornado.svg` ([results files](docs/setup-run-results.md#results-files)). For a readable report, run `catmodel results summary --results <results.json> --output <new-directory>`, which writes `summary.md` and `summary.csv` ([readable summary](docs/setup-run-results.md#readable-summary)).

### Choosing the worker count

`catmodel --workers N` sets the worker processes of `run`, `results replay`, `results export` and `setup import --study`, and the collector processes of `collect --study` ([What bounds throughput](#what-bounds-throughput)). The default is the CPUs available to the process. On the same platform and dependency versions, every number and identity is the same for every N; `results.json` records N under `execution_context`.

Memory grows with the simulated state-years and with the number of arms: a run holds every arm's annual draws while it derives its results, and replay and export load a capture whole. Allow about 13 GiB to run, replay and export a 15-node study's setup with the default comparison families and no variant studies. The setup with every family, which the study script imports, needs more.

If deriving the results fails after the simulation, the run moves its capture to the output directory and writes `INCOMPLETE.txt` beside it, naming the `results replay` command that derives the results. A run killed at that point leaves the capture whole in its staging directory beside the output (`.run-…/run/capture`); run `results replay --capture` on it where there is more memory. A killed import, run, replay or export can also leave a temporary directory beside its output (`.setup-arms-…`, `.compiled-…`), which can be deleted.

## 5. Derive results from a capture

`results replay` reduces the immutable capture again, without simulating:

```sh
uv run --frozen catmodel results replay --setup public-frontier-setup \
  --capture model-run/capture --output replay
```

- `--level` adds tail levels to the defaults, 0.9, 0.99, 0.995, 0.996 and 0.999.
- `--weights <file.json>` changes the joint states' weights and nothing else. The file maps every declared state ID to its target probability, or every required arm ID to such an object; a public study's setup needs the per-arm shape ([frozen workflow](docs/setup-run-results.md#frozen-workflow)). Weight on a state outside an arm's captured support is refused.

A changed law, ownership or occurrence grouping needs a new run.

`catmodel results export --setup <setup> --capture <run/capture> --output <new-directory>` writes a portable, checksummed export of the results the capture reduces to. With `--results replay/results.json` it exports a replay instead, and refuses a results file that does not reproduce from the same setup and capture. The export's copy of an imported setup leaves out the captured prompts, responses and parsed answers and lists each file by path and SHA-256, and the export still loads and verifies; `--include-answers` copies them ([portable results](docs/setup-run-results.md#portable-results-and-conditional-distributions)).

## 6. Reporting and failures

Before quoting a number, read in the results or the summary:

- `status`, `coverage` and `limitations`, and each statistic's status: a `withheld` or `unavailable` statistic is not zero ([frozen workflow](docs/setup-run-results.md#frozen-workflow));
- `respondent_sampling`: an effect with |z| below 2 is within respondent-sampling noise;
- `trajectory_quadrature`, before reporting a forecast comparison;
- the `eal_crosscheck` block of `frequency_estimate` and the plausibility checks (`NUM-PLAUSIBILITY`): an unresolved leaf, or a flagged share or count, calls for review of that leaf's judgments, not a rerun.

Report VaR, TVaR and occurrence (OEP) statistics as illustrative ([statistic statuses](docs/setup-run-results.md#frozen-workflow)), omitted populations as unquantified, and the absolute effect where a relative one has a zero or tiny denominator. The [intended-purpose statement](model-intended-purpose.md#interpretation-and-exclusions) lists what every communication of a result states.

Repair a checksum, population, acquisition or completeness failure at its authored source. Never restamp stale evidence, patch a generated manifest, insert zero answers or invent composition.
