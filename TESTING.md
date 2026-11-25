# Testing Policy

Run the repository checks before submitting a change; CI is authoritative for
merge gating.

## Repository checks

Set up as [CONTRIBUTING.md](CONTRIBUTING.md#set-up) describes, then run:

```sh
uv lock  # only when pyproject.toml dependencies change
uv run ruff check --fix
uv run ruff format
uv run lint-imports
uv run ty check
uv run python scripts/check_dependency_pins.py  # when dependency declarations change
uv run pytest -n auto -q
```

`scripts/check_dependency_pins.py` requires an exact `==` pin of every
project, dependency-group and build-system requirement in each `pyproject.toml`
in the repository, or in the files named on its command line; naming a file
that does not exist is an error.

CI installs with `uv sync --frozen`, which does not compare the lockfile with
`pyproject.toml`; the release workflow's `uv lock --check` does, so a stale
lockfile fails `ci-green`.

`uv run pre-commit install` adds the hooks in `.pre-commit-config.yaml`; they
do not replace the commands above (in particular `lint-imports` and the
tests).

## What tests establish

Tests protect the implementation, artifact contracts, numerical invariants and
release boundaries. A passing suite does not show that the model's outputs are
right ([limitations](docs/limitations.md)).

Tests should prefer:

- public behavior and contract assertions over private implementation details
- negative cases for schemas, validation and provenance
- conservation, monotonicity, finiteness and reconciliation invariants
- small end-to-end fixtures over mocks of code owned by this repository

Exact floating-point snapshots, log-string pins and tests of third-party library
behavior are discouraged.

Each test should identify a plausible repository regression and use an oracle
independent of the behavior under test. For numerical routines, use a small
hand-calculated example alongside relevant statistical invariants. For external
boundaries, replace the transport or remote service while retaining owned
validation and real local parsing or cryptography.

A test of a command that writes run evidence keeps that run, and any run it
reads, in temporary directories. Generated shared fixture trees are read-only
inputs to tests.

Keep one integration check that current authored inputs load successfully;
exercise validation rules with invalid inputs instead of repeatedly asserting
schema guarantees over those already-loaded fixtures. Keep independent semantic
pins, frozen reader fixtures and provenance checks. Assert copy, layout or
configuration only where it is a specific contract.

## CI jobs

`.github/workflows/ci.yml` contains the executable commands. Every job runs on
every pull request and on every push to `main`. The last job, `ci-green`, is
the single required status: it passes only when every other job has succeeded.

The other jobs are:

- `tests`: pytest in parallel on Python 3.13 and 3.14. A pull request splits
  the suite over four jobs for each version; a push to `main` runs it whole,
  with line and branch coverage on the Python 3.14 run
- `lint`: Ruff, formatting, import-boundary and dependency-pin checks
- `typecheck`: `ty` type checking
- `audit`: dependency vulnerability audit
- `release-artifact`: the reusable release workflow, which builds the wheel
  and source distribution, validates their contents and smoke-tests the
  installed CLI and packaged resources on Python 3.13

`tests` reports the 25 slowest test phases, including fixture setup. Reproduce
that report with `uv run pytest -n auto --durations=25 -q`.

`--shard I/N` runs part I of N of the collected tests. A test's part is fixed
by its id, so the N parts are disjoint and together run every test once.
Reproduce one job of a pull request with
`uv run pytest -n auto --shard 2/4 --durations=25 -q`.

## Test markers

The `stochastic` marker identifies tests that depend on simulation, sampling,
bootstrap or Monte Carlo diagnostics even when seeded. These tests run in the
default suite; run them alone with:

```sh
uv run pytest -q -m stochastic --durations=30
```

There is no `slow` marker: CI runs every test, and a slow test must justify the
risk it protects.

## Numerical tests

Numerical hot paths require tests for both ordinary and adverse inputs:

- distribution quantiles, tails, caps and support boundaries
- conditional Poisson, fixed-VMR negative-binomial and clustered initiating counts
- trajectory quadrature integration, conditional independence and probability-mixture invariants
- occurrence and aggregate policy terms and loss allocation
- EAL, AEP/OEP, VaR, TVaR and convergence diagnostics
- sensitivity direction, eligibility, unavailable alternatives and paired-seed behavior
- source capture, package binding, evidence review state and fail-closed execution

Use `pytest.approx` or `numpy.testing.assert_allclose` with a tolerance justified
by numerical behavior. A seeded Monte Carlo test should assert the intended
statistical property, not accidental bit-for-bit output. Derive its tolerance
from the estimator's standard error under the oracle law, normally five standard
errors, rather than tuning it to one seed. When the estimator has no standard
error (an infinite-variance mean), bound it one-sided through a truncated,
finite-variance statistic. Compute expected values independently of the code
under test: closed-form moments, a separate recursion or an exact law, not the
engine's own analytic path.

## Goldens and regression guards

- `tests/fixtures/llm/`: request goldens for each provider mechanism of the LLM transport, exercised by `tests/test_llm_litellm_provider.py`
- `tests/test_contextual_risk.py`: frozen contextual prompt identity and numerical-response schema
- [`config/setups/golden-poisson`](config/setups/golden-poisson/README.md) and `tests/test_golden_poisson.py`: a hand-derived end-to-end example run through `setup build`, `run`, `results replay` and `results export`
- [`docs/example-questions/`](docs/example-questions/README.md) and `tests/test_example_questions.py`: one scenario's direct, chain and parent questions as the committed hypothetical-lab study renders them
- `contracts/results/` and `tests/test_results_schema.py`: the results files' JSON Schema documents equal what the results contracts generate (`uv run python scripts/generate_results_schemas.py` writes them)
- `scripts/offline_demo.py` and `tests/test_offline_demo.py`: a one-scenario hypothetical-lab study taken from planning to the summary on synthetic answers from a fake respondent, without a provider

Refresh a golden or pinned fixture only for a reviewed, intentional behavior
change, never just to make a failure disappear. The PR states the numerical or
contract rationale, how the new fixture was produced and which downstream
conclusions move.

`test_support/capability_sources.py` supplies authored synthetic upstream
evidence. A test that needs a source bundle writes its own at the current
schema with real checksums. Do not restamp frozen historical evidence into a
current bundle or use it as a compatibility fixture.

## Installed-wheel acceptance

`uv run --frozen python scripts/check_incident_wheel.py --output <new-directory>`
builds a wheel, or reuses one passed with `--wheel <built-wheel>`, and installs
it with hash-locked dependencies in a separate environment. It then runs outside
the checkout with sockets blocked and provider keys omitted. It checks packaged
resources, the conditional example in `config/setups/stipulated-conditional` and
the campaign example in `config/setups/stipulated-campaign`; among other checks,
the reported EAL matches an independent analytic value.

## Source-distribution check

`uv run python scripts/check_sdist.py <built-sdist>` compares a built source
distribution with the files Git tracks in the checkout it runs from;
`uv build --sdist --out-dir <new-directory>` builds one. The source distribution
carries the tracked tree, less any file that `.gitattributes` marks
`export-ignore`. The check fails when the archive holds a file Git does not
track or one marked `export-ignore`, leaves out any other tracked file, or has
a page with a relative Markdown link whose target file or heading the archive
does not carry.

## Coverage

Coverage is diagnostic, not an acceptance threshold. Review missing branches in
numerical hot paths, schema gates and failure handling; do not add tests solely
to raise a percentage.

The main-branch Python 3.14 run uploads `coverage`; collect the same
diagnostics locally with:

```sh
uv run pytest -n auto --cov=catmodel --cov-branch --cov-report=term-missing --cov-report=json:coverage.json -q
```

When removing or replacing tests, inspect newly uncovered behavior and, where
practical, temporarily introduce a representative fault to confirm that the
replacement assertion fails. Keep audit findings and before/after measurements
in the PR.
