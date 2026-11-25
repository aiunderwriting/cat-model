"""Run a small study from questions to results, offline, on synthetic answers.

No network, API key or paid collection is needed. The script plans the
hypothetical-lab study for one scenario at three trajectory nodes, with one
respondent model and one sample, and takes it through every step of the main
pipeline:

  1. build the source bundle and plan the study;
  2. freeze every question, as a dry run of the collection does;
  3. answer the frozen questions from a fake respondent, where a collection
     would call the respondent models;
  4. validate each capture;
  5. import the answers into a setup;
  6. simulate the setup and write the readable summary.

Every answer is synthetic. The fake respondent answers by a fixed rule that
ignores what a question describes: the same incident count for every scenario,
and losses a fixed amount above each scenario's loss floor. The results
therefore say nothing about risk; they show what each step reads and writes.

The setup simulates a twentieth of the state-years that `catmodel setup
import` allows by default, so that the run takes seconds, and the summary
marks the tail statistics that so short a simulation cannot support.

Run it from a clean committed checkout, whose commit the source bundle
records, or pass --source-bundle. The output directory must be new and
outside the checkout.

    uv run python scripts/offline_demo.py

scripts/run_hypothetical_lab_study.py collects the same small study from
respondent models, with --live --scenario ID --nodes 3.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Only the standard library is imported here: the setup import starts worker
# processes, and each of them loads this file again.
REPO_ROOT = Path(__file__).resolve().parents[1]

PROFILE = "offline-demo"
MODEL_ID = "synthetic-respondent"
SETUP_ID = "offline-demo-synthetic-answers"
DEFAULT_SCENARIO = "output_reliance_software"
"""The scenario whose questions docs/example-questions/ shows."""
NODES = 3
"""Interior trajectory nodes, the fewest a study can have."""
STATE_YEAR_DIVISOR = 20
"""The setup's caps on simulated state-years are the defaults of ``catmodel
setup import`` (NUM-TAIL-GATE) divided by this."""

# The fake respondent reports reasoning effort max, so the profile states it.
PROFILES = f"""\
# Written by scripts/offline_demo.py. The one respondent is a fake that
# returns synthetic answers; no provider is called.
profiles:
  {PROFILE}:
    models:
      - provider: openai
        model_id: "{MODEL_ID}"
        reasoning_effort: max
        max_output_tokens: 100000
    settings:
      samples_per_question: 1
"""

SYNTHETIC = (
    "Every answer is synthetic: a fake respondent writes it and no provider is "
    "called. The results say nothing about risk."
)


class _Stopped(Exception):
    """A step failed; the message says why."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="new directory for every file the demo writes (default: a new "
        "temporary directory)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        dest="scenarios",
        metavar="ID",
        help=f"scenario to ask about (repeatable; default: {DEFAULT_SCENARIO}); "
        "each one adds 16 questions, before any follow-up question. `all`, "
        "alone, asks the complete catalog with its taxonomy parents, and "
        "takes a few minutes",
    )
    parser.add_argument(
        "--source-bundle",
        type=Path,
        help="use an existing source bundle instead of building one, for "
        "example when the checkout has uncommitted changes",
    )
    return parser


def _step(number: int, title: str) -> None:
    print(f"\n[{number}/6] {title}", flush=True)


def _cli(*arguments: str | Path) -> list[str]:
    return [sys.executable, "-m", "catmodel.cli", *map(str, arguments)]


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    result = subprocess.run(command, check=False, env=env)
    if result.returncode:
        raise _Stopped(f"the command above failed with exit code {result.returncode}")


def _build_bundle(output: Path) -> Path:
    """Build the study's source bundle from the committed recipe, as
    ``catmodel package build-source-bundle`` does."""
    from catmodel.evidence.source_inputs import build_source_bundle
    from scripts.run_hypothetical_lab_study import stage_authored_source

    recipe = stage_authored_source(output / "authored-source")
    bundle = output / "source-bundle"
    print(
        f"Building the source bundle from {recipe} "
        "(catmodel package build-source-bundle)",
        flush=True,
    )
    try:
        build_source_bundle(recipe, bundle)
    except ValueError as exc:
        raise _Stopped(
            f"cannot build the source bundle: {exc}. Commit or stash local "
            "changes, or pass --source-bundle with a bundle built earlier."
        ) from exc
    return bundle


def _answer_from_fake_respondent(plan: Path, profiles: Path) -> int:
    """Collect the planned study with the fake respondent in place of the
    profile's provider; return the number of questions it answered, follow-up
    questions included."""
    import asyncio
    import math
    from typing import cast

    from catmodel.collection.contextual_collection import collect_study
    from catmodel.config import load_model_profile, settings
    from catmodel.contracts.contextual_risk import ContextualRiskInstrument
    from catmodel.json_io import strict_json_loads
    from catmodel.llm.transport import LLMProvider
    from catmodel.setup.public_study import public_study_collections
    from test_support.collection_trace import (
        CollectionTrace,
        LatencyProvider,
        ModelLatency,
        StudyResponder,
    )

    contexts = [
        (
            ContextualRiskInstrument.from_saved(
                strict_json_loads(instrument.read_bytes())
            ),
            capture,
        )
        for instrument, capture in public_study_collections(plan)
    ]
    profile = load_model_profile(profile=PROFILE, path=profiles)
    # The settings a collection with this profile runs under.
    execution = profile.execution_settings(
        profile.models, max_output_tokens=settings.ELICITATION_MAX_TOKENS
    )
    responder = StudyResponder([instrument for instrument, _ in contexts])
    providers = cast(
        list[LLMProvider],
        [
            LatencyProvider(
                model_id=spec.model_id,
                provider_name=spec.provider,
                # The fake answers at once.
                latency=ModelLatency(
                    output_tokens=1.0,
                    tokens_per_second=math.inf,
                    first_token_seconds=0.0,
                ),
                responder=responder,
                trace=CollectionTrace(),
                max_concurrent=execution.concurrency_by_model[spec.model_id],
            )
            for spec in profile.models
        ],
    )
    results = asyncio.run(
        collect_study(contexts, providers=providers, config=execution)
    )
    failed = [str(result) for result in results if isinstance(result, Exception)]
    if failed:
        raise _Stopped(f"the collection from the fake respondent failed: {failed[0]}")
    return sum(
        1 for _, capture in contexts for _ in capture.glob("calls/*/request.json")
    )


def _validate(plan: Path, reports: Path, env: dict[str, str]) -> None:
    """Validate every capture of the study, several at a time, and keep the
    report ``catmodel capability-risk validate`` prints for each."""
    from catmodel.setup.public_study import public_study_collections
    from catmodel.worker_pool import available_cpus

    def validate(capture: Path) -> tuple[list[str], subprocess.CompletedProcess]:
        command = _cli("capability-risk", "validate", "--run-directory", capture)
        return command, subprocess.run(
            command, check=False, env=env, capture_output=True, text=True
        )

    reports.mkdir()
    captures = [capture for _, capture in public_study_collections(plan)]
    with ThreadPoolExecutor(max_workers=available_cpus()) as pool:
        for capture, (command, result) in zip(
            captures, pool.map(validate, captures), strict=True
        ):
            print("+ " + shlex.join(command), flush=True)
            if result.returncode:
                sys.stderr.write(result.stderr)
                raise _Stopped(
                    f"the command above failed with exit code {result.returncode}"
                )
            (reports / f"{capture.name}.json").write_text(
                result.stdout, encoding="utf-8"
            )


def _import_setup(plan: Path, setup: Path) -> None:
    """Import the study into a setup, as ``catmodel setup import`` does, with
    the demo's caps on simulated state-years."""
    from catmodel.contracts.incident import (
        DEFAULT_MAX_COMPARISON_STATE_YEARS,
        DEFAULT_MAX_REFERENCE_STATE_YEARS,
    )
    from catmodel.setup.public_study import import_public_frontier_study
    from catmodel.worker_pool import available_cpus

    reference = DEFAULT_MAX_REFERENCE_STATE_YEARS // STATE_YEAR_DIVISOR
    comparison = DEFAULT_MAX_COMPARISON_STATE_YEARS // STATE_YEAR_DIVISOR
    print(
        f"Importing {plan} as setup {SETUP_ID} (catmodel setup import), with at "
        f"most {reference:,} simulated state-years in the reference arm and "
        f"{comparison:,} in a comparison arm",
        flush=True,
    )
    try:
        import_public_frontier_study(
            source=plan,
            destination=setup,
            setup_id=SETUP_ID,
            max_reference_state_years=reference,
            max_comparison_state_years=comparison,
            workers=available_cpus(),
        )
    except ValueError as exc:
        raise _Stopped(f"cannot import the study: {exc}") from exc


def _demo(output: Path, scenarios: list[str], source_bundle: Path | None) -> None:
    from catmodel.config import PROFILES_ENV
    from catmodel.resources import data_path
    from scripts.run_hypothetical_lab_study import STUDY

    profiles = output / "profiles.yaml"
    profiles.write_text(PROFILES, encoding="utf-8")
    env = {**os.environ, PROFILES_ENV: str(profiles)}
    study = output / "public-study"
    plan = study / "public-study.json"
    validation = output / "validation"
    setup = output / "setup"
    model_run = output / "model-run"
    summary = output / "summary"

    _step(1, "Build the source bundle and plan the study")
    bundle = source_bundle or _build_bundle(output)
    _run(
        _cli(
            "--profile",
            PROFILE,
            "capability-forecast",
            "public-study",
            "--source-bundle",
            bundle,
            "--forecast",
            data_path("capability", "public-frontier-forecast-template.json"),
            "--context",
            STUDY / "context.json",
            *(arg for scenario in scenarios for arg in ("--scenario", scenario)),
            "--assumptions",
            STUDY / "public-assembly-assumptions.json",
            "--nodes",
            str(NODES),
            "--output",
            study,
        ),
        env,
    )

    _step(2, "Freeze every question (a dry run: no provider is called)")
    _run(
        _cli(
            "--profile",
            PROFILE,
            "capability-risk",
            "collect",
            "--study",
            plan,
            "--source-bundle",
            bundle,
            "--dry-run",
        ),
        env,
    )

    _step(3, "Answer the frozen questions from the fake respondent")
    print(
        "A collection runs the command above without --dry-run and calls the "
        "profile's models. Here a fake respondent answers in their place.",
        flush=True,
    )
    asked = _answer_from_fake_respondent(plan, profiles)
    print(f"The fake respondent answered {asked} questions.", flush=True)

    _step(4, "Validate each capture")
    _validate(plan, validation, env)

    _step(5, "Import the answers into a setup")
    _import_setup(plan, setup)

    _step(6, "Simulate the setup and write the summary")
    _run(_cli("run", "--setup", setup, "--output", model_run), env)
    _run(
        _cli(
            "results",
            "summary",
            "--results",
            model_run / "results.json",
            "--output",
            summary,
        ),
        env,
    )

    print(f"\n{SYNTHETIC}")
    collections = study / "collections" / "*"
    for label, path in (
        ("source bundle", bundle),
        ("study plan", plan),
        ("questions", collections / "calls" / "*" / "request.json"),
        ("synthetic answers", collections / "contextual-risk.json"),
        ("validation reports", validation / "*.json"),
        ("setup", setup),
        ("results", model_run / "results.json"),
        ("summary", summary / "summary.md"),
    ):
        print(f"  {label + ':':<20}{path}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # One linear-algebra thread, set before NumPy loads, as in every catmodel
    # command.
    from catmodel.linear_algebra_threads import use_one_linear_algebra_thread

    use_one_linear_algebra_thread()
    # For scripts.run_hypothetical_lab_study and test_support.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from catmodel.logging import configure_logging
    from catmodel.resources import source_checkout_root

    started = time.monotonic()
    if args.output is None:
        output = Path(tempfile.mkdtemp(prefix="catmodel-offline-demo-")).resolve()
    else:
        output = args.output.expanduser().resolve()
        if output.exists() or output.is_symlink():
            print(f"Error: output path already exists: {output}", file=sys.stderr)
            return 2
    checkout = source_checkout_root()
    if checkout is not None and output.is_relative_to(checkout.resolve()):
        if args.output is None:
            output.rmdir()
        print(
            f"Error: output directory is inside the source checkout: {output}. "
            "The source bundle is built from a clean committed checkout, so "
            "write outside the checkout.",
            file=sys.stderr,
        )
        return 2
    scenarios = args.scenarios or [DEFAULT_SCENARIO]
    print(
        f"Offline demo of the hypothetical-lab study: {', '.join(scenarios)} at "
        f"{NODES} trajectory nodes, with one fake respondent and one sample.\n"
        f"{SYNTHETIC}\nWriting to {output}",
        flush=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    configure_logging()
    source_bundle = args.source_bundle
    if source_bundle is not None:
        source_bundle = source_bundle.expanduser().resolve()
    try:
        _demo(output, scenarios, source_bundle)
    except _Stopped as stopped:
        print(f"Error: {stopped}", file=sys.stderr)
        return 1
    print(f"Finished in {time.monotonic() - started:.0f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
