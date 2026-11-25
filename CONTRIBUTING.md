# Contributing

This repository is experimental research software maintained by AIUC. Interfaces change without notice, and backward compatibility is not a goal: a change updates every caller instead of adding compatibility shims. Open an issue to discuss a substantial change before starting it. Report a security vulnerability privately, as [SECURITY.md](SECURITY.md) describes, not in an issue. Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md); the maintainers' contact is tom@aiuc.com.

## Set up

You need Python 3.13 or 3.14 and uv 0.9.17 or newer, the first release that reads the relative `exclude-newer` cooldown in `pyproject.toml`. An older uv warns, ignores the project's uv settings and resolves the lockfile again without them. Check `uv --version` and upgrade with `uv self update`. CI runs the uv version that its workflows name.

```sh
uv sync --frozen --group dev
uv run pre-commit install  # optional commit hooks
```

The [quickstart](README.md#quickstart) checks the installation against a hand-derived example. `uv run python scripts/offline_demo.py` then takes a one-scenario study from questions to a summary in about a minute, with no API key: a fake respondent writes every answer, so its results say nothing about risk.

Then read [ARCHITECTURE.md](ARCHITECTURE.md). It maps the pipeline stages to the code, lays out the packages and their import layers, and says where to start on common changes.

## Before you open a pull request

Run the [repository checks](TESTING.md#repository-checks). CI's single required check, `ci-green`, must pass.

Tests make no LLM provider calls, and they read model profiles from AIUC's packaged example unless a test supplies its own file. While developing, collect with the example's `--profile test` or `--profile dev`, or with a [profiles file](runbook.md#model-profiles) of your own, and run a collection command with `--dry-run` unless you intend a live collection. The [output-ceiling probe](docs/workflows.md#scripts) makes one live call per selected model.

## Conventions

- **Dependencies.** Pin every first-party dependency to an exact version (`==`); `scripts/check_dependency_pins.py` enforces this, and its docstring explains the exceptions. Pin a new dependency to the latest stable release that the `exclude-newer` setting in `pyproject.toml` admits, and note version changes in the commit message.
- **Logging.** Use `structlog` (`logger = structlog.get_logger(__name__)`) with structured keyword context. Ruff blocks the standard-library `logging` import (`TID251`).
- **Randomness.** Pass explicit seeds or `numpy.random.Generator` instances, so runs can be compared with less Monte Carlo noise. Record the seed and simulation count, and do not rely on bit-for-bit reproducibility across platforms or dependency versions.
- **Assumptions.** State every assumption with its rationale, evidence status and sensitivity range in the [assumption register](assumptions.md), and label a placeholder as one.
- **Evidence flows one way.** No source-bundle file may derive its content from elicitation output ([governance](model-governance.md#authorities-and-ownership)).
- **Scenarios.** Change a taxonomy split as [Adding or changing a split](methodology.md#adding-or-changing-a-split) describes, and add or change a scenario as [Adding a scenario](docs/adding-a-scenario.md) describes. `uv run python scripts/validate_scenario_library.py` checks the scenario library, and `--fix` regenerates the catalog, the taxonomy explorer and the documented counts.
- **Documentation.** Committed documents describe the current system. Update the authoritative document in place; put change rationale, review findings and before/after comparisons in the pull request, not in committed files.

## Pull requests

Keep each pull request the smallest change that leaves the repository internally consistent, with a title and description a reader new to the repository can follow.

## License

Contributions are licensed under the [Apache License, Version 2.0](LICENSE), as section 5 of the license provides.
