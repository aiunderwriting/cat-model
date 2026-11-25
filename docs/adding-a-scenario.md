# Adding a scenario

A scenario forecasts one leaf of the [loss taxonomy](taxonomy.md), or a disjoint union of leaves, for a named lab and deployment. This guide lists everything a new scenario touches. One command validates the library and regenerates what derives from it:

```sh
uv run python scripts/validate_scenario_library.py        # report every error, with its file
uv run python scripts/validate_scenario_library.py --fix  # also regenerate the derived artifacts
```

The leaf must already exist: a new or changed leaf is a split change, made as [Adding or changing a split](../methodology.md#adding-or-changing-a-split) describes.

## Checklist

1. **Claim the leaf.** In the [hierarchy contract](../contracts/scenarios/hierarchy-contract.yaml), remove the leaf's `unestimated_disposition` and revise any `display_summary` or `display_prose` that says no scenario estimates it. Validation requires every leaf to be estimated by exactly one scenario or to carry an `unestimated_disposition`, never both.
2. **Write the scenario file** `scenarios/<origin>/<id>.yaml`, named after its `id`, starting from a scenario of the same origin. [The scenario file](#the-scenario-file) lists what to author and what the library supplies.
3. **Register the roster change.** A roster generation is registered in [`registries/scenario_lineage.yaml`](../registries/scenario_lineage.yaml) whenever the roster changes. Add a generation labelled as the next after the register's `current_taxonomy_version`, with the `scenario_roster_sha256` that the validator prints, make it `current_taxonomy_version`, add a `scope_expansion` transition that names the scenario, and set the scenario's `lineage.introduced_in` to the new generation. [The lineage entries](#the-lineage-entries) shows every field.
4. **Record coverage gaps.** If an exposure base has `basis: normalisation`, name the scenario in an `ungoverned_exposure_denominator` row of [`registries/coverage_gaps.yaml`](../registries/coverage_gaps.yaml); validation requires it. Record any other known gap there too: a result that includes the scenario lists its rows among its limitations.
5. **List the scenario's source cards**, the dated excerpts its questions quote as evidence. Under `scenarios` in [`data/elicitation_source_cards.yaml`](../data/elicitation_source_cards.yaml), map the scenario's `id` to the card ids in the order its questions show them. The question about each taxonomy category above the scenario quotes them too. Validation requires an entry for every scenario and for nothing else; an empty list, `[]`, shows none. A new card goes under `cards`, must be listed by a scenario and follows the rules in the file's header, which the tests check.
6. **Regenerate.** `--fix` rewrites [`scenarios/scenario_catalog.json`](../scenarios/scenario_catalog.json), the [taxonomy explorer](taxonomy-explorer.html) and the `scenario-count` markers in README.md, methodology.md, runbook.md, docs/taxonomy.md and docs/limitations.md, which validation otherwise reports as stale. Review the diff.
7. **Update the prose that names leaves**, which `--fix` does not rewrite: the forecast leaves and the unestimated leaves in [taxonomy.md](taxonomy.md#which-categories-have-estimates) and the [methodology's tree](../methodology.md#tree); the unestimated leaves in `HIER-REMAINDER` and the [coverage assessment](taxonomy-coverage.md); and the tail classes' members in [`severity_governance_policies.yaml`](../scenarios/severity_governance_policies.yaml), the methodology's [severity section](../methodology.md#conditional-severity) and `SEV-GPD-SHAPE`.
8. **Run the tests.** Library tests derive their counts from the library. Tests that pin leaves or scenarios by name, such as the taxonomy-export, external-case and taxonomy-parent tests and the pinned lists of normalisation bases and tail-class members, change with them; update them and say why in the pull request.

A new scenario has no answers: a context-bound study must be collected again to include it.

## The scenario file

Author these fields. The loader refuses an unknown field, and reports each error with the file and field it belongs to.

| Field | What to author |
| --- | --- |
| `id`, `name`, `node_ids` | The scenario's identity, with an `id` no other file uses, and the leaf or leaves it forecasts. A union's leaves share one origin, AI role and reporting domain; the reporting domain derives from the leaf. |
| `lineage.introduced_in` | The roster generation that added the scenario. |
| `description` | A short summary that does not restate the population contract. |
| `frequency_semantics` | `counting_identity`, `terminal_event_unit` and `year_rule`. The terminal unit states the admission threshold once, as `{{admission}}`; write no amount of money here, in the attribution nexus or in the chain's opportunity and stages. |
| `attribution_nexus` | `kind` and the `terminal_qualification` that ties the incident to the provider. |
| `severity_occurrence_scope` | The gross loss of one incident, with one `{{loss_component:<cell_id>}}` marker for each tracked composition cell and for no other. |
| `severity_governance` | `policy: governed_tail_shape`, a `tail_class` from [`severity_governance_policies.yaml`](../scenarios/severity_governance_policies.yaml) with a one-line `tail_class_rationale`, a `cap_posture` unless the class owns the endpoint, as the catastrophe class does, any `severity_scope_exclusions`, and any `evidence_refs`, each a URL, a citation or a repository file and heading. The class owns the tail shape. |
| `estimand_composition` | `roster: standard` (the whole-incident roster most scenarios use) or `roster: private_claims` from [`composition_rosters.yaml`](../scenarios/composition_rosters.yaml), optionally restating a cell's loss form under `loss_forms`; or the scenario's own `status`, `rationale`, `evidence` and `cells`. A cell with a `loss_type` and `attachment_basis` is tracked: its loss form renders at its marker in the severity scope. |
| `event_chain` | The `opportunity`, one exposure base for `public` and one for `internal` capability access, and one to four `stages` that neither restate the terminal unit nor repeat each other. |
| `facets` | Reporting classifications; cells where the AI helped an attacker use `deployment_control_mode: any_copy`. |
| `occurrence_unit_id` | A name for the counted unit. |
| `empirical_grounding` | What evidence exists for this population and what it does not establish. The catalog reports it. |

The library supplies the rest: the admission amount and severity floor from the hierarchy contract, the tail shape and endpoint from the tail class, the severity policy's classification and sensitivity range, and a named roster's cells.

### Exposure bases

Each exposure base names one base unit in the singular, a `count` and a one-sentence `derivation` that the respondent reads beside the count. A `derived` count reads one of the declared deployment scales, `consumer_weekly_users`, `paying_business_customers` or `lab_staff` (`EXPOSURE_SCALES` in `catmodel.scenarios.loader`), times an authored `factor`; a `fixed` count is one number, whose derivation says it is fixed by the scenario. Write a size unit as "1 million weekly active users of the consumer assistant" or "1,000 lab staff", with the derivation "divided by 1 million" or "divided by 1,000".

`basis` says what the count rests on:

- `judgment`: the base unit is the population from which opportunities arise, and the count is an authored estimate, such as a share of a scale, a fixed number or a scale taken as that population.
- `data`: a cited source measures the count or its share of a scale.
- `normalisation`: the base only measures the lab's size, because no scale counts the population from which opportunities arise. The opportunity rate carries the whole exposure judgment, and the scenario needs a coverage-gap row.

The README and limitations counts of cells with a normalisation base derive from this field.

## The lineage entries

The register's first generation is `r1`, and every later roster change registers the next: `r2`, `r3` and so on. A new scenario adds one generation and one transition. In the example, `<current>` is the register's `current_taxonomy_version` before the change and `<next>` is the label after it, which no registered generation uses:

```yaml
current_taxonomy_version: "<next>"

generations:
  # Earlier generations stay.
  - version: "<next>"
    established: "<date>"
    scenario_roster_sha256: "<digest the validator prints>"
    description: >-
      The <current> roster and <id>, which estimates <leaf>.

transitions:
  - transition_id: "add_<id>"
    kind: "scope_expansion"
    executed_in: "<next>"
    successor_ids: ["<id>"]
    evidence_staling: "none"
    rationale: >-
      No scenario estimated <leaf>; adding one changes no other scenario's population.
```

Every field shown for the generation and the transition is required, and dates are text, so quote them. A `scope_expansion` names no `predecessor_ids`; the other kinds, `split`, `merge`, `carve_out` and `refinement`, require them. `evidence_staling` records how much earlier evidence the change stales: `none`, `denominator` or `all`. Validation checks the digest, each entry's generation and each transition's scenario ids, but not that a roster change has a transition. [`taxonomy_identity.py`](../src/catmodel/scenarios/taxonomy_identity.py) defines these entries and `retired_scenarios`.

## Changing a scenario

A capture stays valid only while its questions are unchanged: reloading one refuses any node whose population, counting definition or asked chain (opportunity, exposure unit, count rule, derivation and stages) differs from the current file. Audit text that no question shows, such as an exposure base's `basis`, `rationale` and `source`, a cell's `rationale`, `empirical_grounding` and `facets`, can be revised without refusing earlier captures.

A scenario's [threat model](threat-models.md) is bound to its definition: once the description, counted unit, loss scope, attribution rule or chain changes, validation refuses the threat model and prints the digest to record once the threat model matches the new definition. A new scenario needs none; its questions then show only its source cards.

Renaming, splitting, merging or retiring a scenario changes the roster: register a generation with a `transitions` or `retired_scenarios` entry ([the lineage entries](#the-lineage-entries)), and update the coverage-gap rows and the source-card entries, which validation requires to name live scenarios.
