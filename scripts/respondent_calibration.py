"""Collect known-answer count questions and fit the respondent spread.

The respondent spread calibration asks the respondent models count questions
whose answers are known, then fits a spread factor k and a common level bias
b to their answers (methodology.md, "Respondent spread calibration"). The fit
is a diagnostic of the respondents' answers to those questions. No setup
reads it, and it does not validate the risk model.

``collect`` freezes the seeds and a profile's respondents as an instrument,
with every prompt, in ``--run-directory``. It requires exactly one of
``--dry-run``, which stops there, and ``--live``, which asks the respondent
models; rerunning the same command resumes from the stored calls. It prints
the number of frozen requests; a dry run with ``--answers`` adds how many of
them the answer table already holds, and a collection adds the answered,
exhausted and unavailable cases.

``fit`` validates a collection offline, verifies each frozen resolution
script against its digest in this source checkout, and writes the fit as a
new JSON file. It uses 2,000 bootstrap replicates and 200 simulated reference
replicates (seed 42) in ``--workers`` processes; the fit does not depend on
the worker count.

    uv run --frozen python scripts/respondent_calibration.py collect --seeds data/calibration/respondent_spread/seeds.yaml --run-directory calibration-run --dry-run
    uv run --frozen python scripts/respondent_calibration.py collect --seeds data/calibration/respondent_spread/seeds.yaml --run-directory calibration-run --live
    uv run --frozen python scripts/respondent_calibration.py fit --run-directory calibration-run --resolutions data/calibration/respondent_spread/resolutions.yaml --output respondent-spread-fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from catmodel.linear_algebra_threads import use_one_linear_algebra_thread

# Before NumPy loads: the fit and its worker processes take the one
# linear-algebra thread every catmodel command runs with.
use_one_linear_algebra_thread()

from catmodel.calibration.respondent_questions import calibration_questions  # noqa: E402
from catmodel.collection.answer_table import AnswerTable  # noqa: E402
from catmodel.collection.respondent_calibration import (  # noqa: E402
    build_calibration_instrument,
    fit_calibration,
    run_calibration_collection,
)
from catmodel.config import load_model_profile  # noqa: E402
from catmodel.contracts.elicitation import Respondent  # noqa: E402
from catmodel.worker_pool import available_cpus  # noqa: E402


def collect(args: argparse.Namespace) -> dict[str, int]:
    """Freeze the seeds and the profile's respondents as an instrument, with
    every prompt, and unless ``--dry-run`` collect the answers. The
    respondents are every model of the profile, in roster order, at the
    profile's samples per question."""
    profile = load_model_profile(profile=args.profile, path=args.profiles)
    respondents = tuple(
        Respondent(
            respondent_id=f"{spec.model_id}:{sample}",
            model_id=spec.model_id,
            sample_index=sample,
        )
        for spec in profile.models
        for sample in range(profile.settings.samples_per_question)
    )
    instrument = build_calibration_instrument(args.seeds, respondents)
    result = run_calibration_collection(
        instrument=instrument,
        run_directory=args.run_directory,
        profile=args.profile,
        profiles=args.profiles,
        dry_run=args.dry_run,
        answers=args.answers,
    )
    summary = {"frozen_requests": len(calibration_questions(instrument))}
    if result is None:
        if args.answers is not None:
            held, _ = AnswerTable(args.answers).coverage(args.run_directory)
            summary["held_by_answer_table"] = held
        return summary
    answered = [row.answer for row in result.cases if row.answer is not None]
    return summary | {
        "cases": len(result.cases),
        "answered": len(answered),
        "exhausted": len(result.cases) - len(answered),
        "unavailable": sum(row.status == "unavailable" for row in answered),
    }


def fit(args: argparse.Namespace) -> dict[str, Any]:
    """Fit the joint spread and level factors from a calibration collection
    and write the fit to the new file ``--output``."""
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(
            f"{args.output} already exists; outputs are never overwritten, so "
            "name a new path."
        )
    result = fit_calibration(
        args.run_directory, args.resolutions, args.output, workers=args.workers
    )
    diagnostics = result.diagnostics
    return {
        "status": result.status,
        "k_hat": result.k_hat,
        "range": result.range.model_dump(mode="json"),
        "b_hat": result.b_hat,
        "b_range": result.b_range.model_dump(mode="json"),
        "k_b_correlation": result.k_b_correlation,
        "marginal": result.marginal.model_dump(mode="json"),
        "observation": result.observation.model_dump(mode="json"),
        "alternatives": result.alternatives.model_dump(mode="json"),
        "leave_one_block_out": {
            "joint_total_gain": diagnostics.leave_one_block_out.joint_total_gain,
            "conditional_k_total_gain": (
                diagnostics.leave_one_block_out.conditional_k_total_gain
            ),
        },
        "primary_pit_medians": {
            "at_k_1_b_1": diagnostics.primary_pit.at_k_1_b_1.median,
            "at_k_hat_b_1": diagnostics.primary_pit.at_k_hat_b_1.median,
            "at_k_1_b_hat": diagnostics.primary_pit.at_k_1_b_hat.median,
            "at_k_hat_b_hat": diagnostics.primary_pit.at_k_hat_b_hat.median,
        },
    }


def _workers(value: str) -> int:
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("at least one worker process is needed")
    return workers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collect_command = commands.add_parser(
        "collect", help="freeze or ask every calibration question"
    )
    collect_command.add_argument(
        "--seeds",
        type=Path,
        required=True,
        help="calibration seed definitions, such as data/calibration/respondent_spread/seeds.yaml",
    )
    collect_command.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="directory for the collection: a new one, or one to resume",
    )
    collect_command.add_argument(
        "--profile",
        default="example",
        help="profile name in the model profiles file (default: example)",
    )
    collect_command.add_argument(
        "--profiles",
        type=Path,
        help="model profiles file to read --profile from (default: the file CATMODEL_PROFILES names, else AIUC's example data/example_model_profiles.yaml)",
    )
    collect_command.add_argument(
        "--dry-run",
        action="store_true",
        help="freeze the instrument, plan and every prompt without calling a provider",
    )
    collect_command.add_argument(
        "--live",
        action="store_true",
        help="call the respondent models; required unless --dry-run is given",
    )
    collect_command.add_argument(
        "--answers",
        type=Path,
        help="answer table directory: a request the table already answers is not sent, and every response is added to it (a new path starts a table)",
    )
    fit_command = commands.add_parser("fit", help="fit a calibration collection")
    fit_command.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="calibration collection directory, which fitting validates offline first",
    )
    fit_command.add_argument(
        "--resolutions",
        type=Path,
        required=True,
        help="seed outcome resolutions, such as data/calibration/respondent_spread/resolutions.yaml",
    )
    fit_command.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new file for the respondent spread fit JSON; it must not exist",
    )
    fit_command.add_argument(
        "--workers",
        type=_workers,
        default=available_cpus(),
        help="worker processes (default: the CPUs available); the fit does not depend on it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "collect":
            if args.dry_run == args.live:
                print(
                    "Error: pass exactly one of --dry-run, which freezes every "
                    "question without calling a provider, or --live, which "
                    f"collects from the respondent models of profile "
                    f"{args.profile!r}; every call is billed.",
                    file=sys.stderr,
                )
                return 2
            summary = collect(args)
        else:
            summary = fit(args)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
