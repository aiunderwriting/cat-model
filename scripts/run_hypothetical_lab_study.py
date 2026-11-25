"""Plan, collect and run the hypothetical-frontier-lab study and, on request, its variant studies.

The output directory must be new and outside the source checkout, unless
``--resume`` continues an earlier run of the same request in it: each step
that resume can skip records what its output was built from in a request
record beside it, and an output whose record differs from the request stops
the script. Exactly one of ``--dry-run`` and ``--live`` is required.
``--dry-run`` builds the bundle, the study plans and the collection prompts
without calling a provider; ``--live`` also collects answers from the
respondent models, validates captures, imports a setup, simulates it and
writes summary and export outputs. A live collection of the complete catalog
at the default 15 nodes is tens of thousands of billed calls. ``--scenario``
and ``--nodes 3`` plan a small study: one scenario at three nodes is 16
questions for each sample of each respondent model, before any follow-up
question. ``scripts/offline_demo.py`` takes that small study through the same
steps on synthetic answers, without a provider.
Runbook section 2, "Running the hypothetical-lab study", documents the
steps, the full command and resumption; section 3 documents provider
failure handling.

    uv run --frozen python scripts/run_hypothetical_lab_study.py ../hfl-dry-run --dry-run --variant weaker --variant stronger
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from catmodel.linear_algebra_threads import use_one_linear_algebra_thread

# Before NumPy loads: this process loads setups and captures to check them,
# so it takes the one linear-algebra thread every catmodel command runs with.
use_one_linear_algebra_thread()

from catmodel.config import EXAMPLE_PROFILES, PROFILES_ENV, load_model_profile  # noqa: E402
from catmodel.evidence.source_bundle import (  # noqa: E402
    MANIFEST_PATH,
    capture_source_bundle_snapshot,
)
from catmodel.resources import config_path, data_path, source_checkout_root  # noqa: E402
from catmodel.results.execution import CAPTURE_MANIFEST, load_run_capture  # noqa: E402
from catmodel.results.results_export import load_results_export  # noqa: E402
from catmodel.results.results_file import diagnostics_path, load_results  # noqa: E402
from catmodel.results.results_summary import CSV_FIELDS  # noqa: E402
from catmodel.setup.public_study import public_study_collections  # noqa: E402
from catmodel.setup.sensitivity_catalog import (  # noqa: E402
    EXPOSURE_SCALE_VARIANTS,
    SAFEGUARD_LEVELS,
)
from catmodel.setup.store import MANIFEST as SETUP_MANIFEST  # noqa: E402
from catmodel.setup.store import load_setup  # noqa: E402

STUDY = config_path("studies", "hypothetical-frontier-lab")
AUTHORED_SOURCE_FILE_MAP = {
    "recipe.yaml": STUDY / "authored-source" / "recipe.yaml",
    "assumption_register.yaml": STUDY / "authored-source" / "assumption_register.yaml",
    "epoch_eci_scores.csv": data_path("capability", "epoch_eci_scores.csv"),
    "epoch_eci_scores.provenance.yaml": data_path(
        "capability", "epoch_eci_scores.provenance.yaml"
    ),
    "elicitation_source_cards.yaml": data_path("elicitation_source_cards.yaml"),
    "threat_models.yaml": data_path("threat_models.yaml"),
}


@dataclass(frozen=True)
class Variant:
    """One variant study: its directory and the public-study ``--variant``
    that plans it."""

    directory: str
    value: str


VARIANTS = (
    *(Variant(f"exposure-{value}", value) for value in EXPOSURE_SCALE_VARIANTS),
    *(Variant(f"safeguard-{value}", value) for value in SAFEGUARD_LEVELS),
)
"""The variants ``--variant`` can name, the ones public-study plans, in the
order they are planned, collected and imported; none is collected unless it
is named."""


@dataclass(frozen=True)
class Step:
    """A command; with ``output``, ``--resume`` skips it when that output
    exists, ``check`` accepts it and the request record beside it equals the
    step's request.

    The request is ``arguments``, the options and committed input digests
    known before any step runs, with the SHA-256 of each file ``upstream``
    names: the files earlier steps wrote that this one reads."""

    command: list[str]
    output: Path | None = None
    check: Callable[[Path], object] | None = None
    arguments: Mapping[str, object] = field(default_factory=dict)
    upstream: Callable[[], Mapping[str, Path]] = dict


def stage_authored_source(directory: Path) -> Path:
    """Copy the committed recipe and source bytes beside the staged recipe."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in AUTHORED_SOURCE_FILE_MAP.items():
        shutil.copyfile(source, directory / name)
    return directory / "recipe.yaml"


def _cli_command(*arguments: str | Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "catmodel.cli",
        *(str(argument) for argument in arguments),
    ]


def _profile_command(args: argparse.Namespace, *arguments: str | Path) -> list[str]:
    """A command that reads the model profile, which the global ``--profile``
    names; the profiles file reaches it through the environment (``_run_all``)."""
    return _cli_command("--profile", args.profile, *arguments)


def _environment(args: argparse.Namespace) -> dict[str, str] | None:
    """Every step's environment: the profiles file given with ``--profiles``
    as CATMODEL_PROFILES, else this process's own environment."""
    if args.profiles is None:
        return None
    return {**os.environ, PROFILES_ENV: str(args.profiles)}


def _check_study(path: Path) -> None:
    public_study_collections(path / "public-study.json")


def _check_model_run(path: Path) -> None:
    """``path`` is a run directory holding a capture and current results
    with their diagnostics file."""
    load_run_capture(path / "capture")
    load_results(path / "results.json", diagnostics=True)


def _check_summary(path: Path) -> None:
    if not (path / "summary.md").is_file():
        raise ValueError(f"{path} has no summary.md")
    with (path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        if tuple(next(csv.reader(handle), ())) != CSV_FIELDS:
            raise ValueError(f"{path / 'summary.csv'} has another header")


def _sha256(path: Path) -> str | None:
    """The SHA-256 of the file at ``path``, or None where there is no file."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _record_path(output: Path) -> Path:
    """The request record of the step that writes ``output``, beside it."""
    return output.with_name(output.name + ".request.json")


def _recorded_request(output: Path) -> dict[str, object] | None:
    """The request ``output`` was built from, or None without a readable
    record of it."""
    try:
        record = json.loads(_record_path(output).read_bytes())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _request(step: Step) -> dict[str, object]:
    return {
        **step.arguments,
        **{name: _sha256(path) for name, path in step.upstream().items()},
    }


def _record_request(output: Path, request: Mapping[str, object]) -> None:
    record = _record_path(output)
    partial = record.with_name(record.name + ".partial")
    partial.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(record)


def _other_request(
    output: Path, recorded: Mapping[str, object], request: Mapping[str, object]
) -> str | None:
    """Why ``output``, built from ``recorded``, is not the output of
    ``request``; None where the two agree."""
    differences = [
        f"{name} was {json.dumps(recorded.get(name))} and is now "
        f"{json.dumps(request.get(name))}"
        for name in sorted({*recorded, *request})
        if recorded.get(name) != request.get(name)
    ]
    if not differences:
        return None
    return (
        f"Error: {output} was built from another request: "
        f"{'; '.join(differences)}. --resume continues only the request that "
        "built it; rerun in a new output directory."
    )


def _changed_arguments(steps: list[Step]) -> str | None:
    """Why the recorded output of one of ``steps`` was built with other
    arguments than the step's, compared before any step runs; None where none
    was."""
    for step in steps:
        if step.output is None or not step.output.exists():
            continue
        recorded = _recorded_request(step.output)
        if recorded is None:
            continue
        refusal = _other_request(
            step.output,
            {name: recorded.get(name) for name in step.arguments},
            step.arguments,
        )
        if refusal is not None:
            return refusal
    return None


def _run_all(
    steps: list[Step], *, resume: bool, env: dict[str, str] | None = None
) -> int:
    for step in steps:
        output = step.output
        if output is not None:
            try:
                request = _request(step)
            except (ValueError, OSError) as exc:
                print(
                    f"Error: cannot read the inputs of {output}: {exc}",
                    file=sys.stderr,
                )
                return 2
            if resume and (output.exists() or output.is_symlink()):
                assert step.check is not None
                try:
                    step.check(output)
                except (ValueError, OSError) as exc:
                    print(
                        f"Error: {output} exists but does not validate ({exc}); "
                        "move it aside and rerun with --resume to rebuild it.",
                        file=sys.stderr,
                    )
                    return 2
                recorded = _recorded_request(output)
                refusal = (
                    f"Error: {output} exists without a readable request record "
                    f"({_record_path(output)}), so --resume cannot tell what it "
                    "was built from; rerun in a new output directory."
                    if recorded is None
                    else _other_request(output, recorded, request)
                )
                if refusal is not None:
                    print(refusal, file=sys.stderr)
                    return 2
                print(f"= complete, skipped: {shlex.join(step.command)}", flush=True)
                continue
            try:
                _record_request(output, request)
            except OSError as exc:
                print(
                    f"Error: cannot write the request record of {output}: {exc}",
                    file=sys.stderr,
                )
                return 2
        print("+ " + shlex.join(step.command), flush=True)
        result = subprocess.run(step.command, check=False, env=env)
        if result.returncode:
            print(
                f"Command failed with exit code {result.returncode}: "
                f"{shlex.join(step.command)}. Rerun the same command with --resume "
                "to continue; see runbook.md section 3 for collection resumption "
                "and provider-failure handling.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


def _interior_nodes(value: str) -> int:
    nodes = int(value)
    if nodes < 3 or not nodes % 2:
        raise argparse.ArgumentTypeError(
            "the trajectory rule needs an odd number of at least 3 interior nodes"
        )
    return nodes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument(
        "--profile",
        default="example",
        help="profile name in the model profiles file (default: example)",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        help="model profiles file to read --profile from, given to every step as CATMODEL_PROFILES (default: the file CATMODEL_PROFILES names, else AIUC's example data/example_model_profiles.yaml)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        dest="scenarios",
        metavar="ID",
        help="scenario to ask about, in the headline and every variant study (repeatable; default: all, the complete catalog); a selection short of the complete catalog asks no taxonomy parent question",
    )
    parser.add_argument(
        "--nodes",
        type=_interior_nodes,
        default=15,
        help="the headline's interior trajectory nodes, an odd number of at least 3; with --variant, 7, 11, 15, ..., so that each variant study takes the (nodes - 1) / 2 nodes of its embedded rule (default: 15)",
    )
    parser.add_argument(
        "--source-bundle",
        type=Path,
        help="use an existing bundle, such as an earlier study's; every instrument binds its manifest digest",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.value for variant in VARIANTS],
        default=[],
        dest="variants",
        help="also build, collect and import the named variant study (repeatable): half or double, the exposure-scale variants, or weaker or stronger, the safeguard-level variants; the import records a variant that is not named as unavailable",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        help="answer table directory given to every collection: a request the table already answers is not sent, and every response is added to it (a new path starts a table)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="freeze collection prompts without calling providers",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="call the respondent models: collect answers, then validate, import, simulate and report; required unless --dry-run is given",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an earlier run of the same request in an existing output directory: collections resume from their stored calls and every other step whose output exists, validates and was built from this request is skipped; --dry-run may become --live",
    )
    return parser


def _study(
    bundle: Path,
    variant: Variant | None,
    nodes: int,
    directory: Path,
    args: argparse.Namespace,
    profiles_sha256: str | None,
) -> Step:
    """Plan the headline study, or the variant study ``variant``, at ``nodes``
    interior nodes: the committed context's questions about the complete
    catalog, or about the scenarios ``--scenario`` names, on the committed
    forecast with its releases filled from the bundle's Epoch ECI table.
    ``profiles_sha256`` identifies the profiles file the profile is read
    from."""
    forecast = data_path("capability", "public-frontier-forecast-template.json")
    context = STUDY / "context.json"
    assumptions = STUDY / "public-assembly-assumptions.json"
    # The planner orders a selection by scenario ID, whatever order names it.
    selection = sorted(args.scenarios)
    scenarios = selection if selection and selection != ["all"] else ["all"]
    return Step(
        _profile_command(
            args,
            "capability-forecast",
            "public-study",
            "--source-bundle",
            bundle,
            "--forecast",
            forecast,
            "--context",
            context,
            *(
                argument
                for scenario in scenarios
                for argument in ("--scenario", scenario)
            ),
            *(() if variant is None else ("--variant", variant.value)),
            "--assumptions",
            assumptions,
            "--nodes",
            str(nodes),
            "--output",
            directory,
        ),
        directory,
        _check_study,
        {
            "nodes": nodes,
            "variant": None if variant is None else variant.value,
            "profile": args.profile,
            "profiles_file_sha256": profiles_sha256,
            "forecast_sha256": _sha256(forecast),
            "context_sha256": _sha256(context),
            "assumptions_sha256": _sha256(assumptions),
            # The complete catalog, the default, records no selection.
            **({} if scenarios == ["all"] else {"scenarios": scenarios}),
        },
        lambda: {"source_bundle_manifest_sha256": bundle / MANIFEST_PATH},
    )


def _collect(study_path: Path, bundle: Path, args: argparse.Namespace) -> Step:
    command = _profile_command(
        args,
        "capability-risk",
        "collect",
        "--study",
        study_path,
        "--source-bundle",
        bundle,
    )
    if args.dry_run:
        command.append("--dry-run")
    if args.answers is not None:
        command.extend(["--answers", str(args.answers)])
    return Step(command)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.variants and (args.nodes < 7 or args.nodes % 4 != 3):
        parser.error(
            "--variant needs a headline of 7, 11, 15, ... interior nodes, so that "
            "its embedded rule, the variant studies' nodes, has an odd number of "
            "at least 3"
        )
    if args.dry_run == args.live:
        print(
            "Error: pass exactly one of --dry-run, which freezes every prompt "
            "without calling a provider, or --live, which collects from the "
            f"respondent models of profile {args.profile!r}. A live collection "
            "of the complete catalog is tens of thousands of billed calls "
            '(runbook.md, "Running the hypothetical-lab study").',
            file=sys.stderr,
        )
        return 2
    requested_output = args.output.expanduser()
    output = requested_output.resolve()
    checkout = source_checkout_root()
    if checkout is not None and output.is_relative_to(checkout.resolve()):
        print(
            f"Error: output directory is inside the source checkout: {output}. "
            "The bundle build requires a clean committed checkout and setup import "
            "and the run record the checkout's state, so write outside the checkout.",
            file=sys.stderr,
        )
        return 2

    if not args.resume and (requested_output.exists() or requested_output.is_symlink()):
        print(
            f"Error: output path already exists: {output}. Pass --resume to "
            "continue an earlier run in it.",
            file=sys.stderr,
        )
        return 2

    try:
        profile = load_model_profile(profile=args.profile, path=args.profiles)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    profiles_sha256 = (
        _sha256(data_path(EXAMPLE_PROFILES))
        if profile.profiles_file is None
        else profile.profiles_file.sha256
    )

    try:
        output.mkdir(parents=True, exist_ok=args.resume)
    except OSError as exc:
        print(f"Error: cannot create output directory {output}: {exc}", file=sys.stderr)
        return 2

    bundle = (
        args.source_bundle.expanduser().resolve()
        if args.source_bundle
        else output / "source-bundle"
    )
    through_collection: list[Step] = []

    if args.source_bundle is None:
        recipe = output / "authored-source" / "recipe.yaml"
        if not bundle.exists():
            stage_authored_source(recipe.parent)
        through_collection.append(
            Step(
                _cli_command(
                    "package",
                    "build-source-bundle",
                    "--recipe",
                    recipe,
                    "--output",
                    bundle,
                ),
                bundle,
                capture_source_bundle_snapshot,
                {
                    name: _sha256(source)
                    for name, source in AUTHORED_SOURCE_FILE_MAP.items()
                },
            )
        )

    study_directory = output / "public-study"
    study_path = study_directory / "public-study.json"
    through_collection.append(
        _study(bundle, None, args.nodes, study_directory, args, profiles_sha256)
    )

    # Each variant is its own study on the headline's embedded rule, at the
    # three samples per model public-study gives a variant (runbook section
    # 2). Its plan holds the embedded rule's interior nodes and the no-progress
    # node, and every row is collected: the no-progress node gives each variant
    # current-frontier states.
    variant_nodes = (args.nodes - 1) // 2
    variants = tuple(variant for variant in VARIANTS if variant.value in args.variants)
    variant_studies: dict[Variant, Path] = {}
    for variant in variants:
        directory = output / variant.directory
        through_collection.append(
            _study(bundle, variant, variant_nodes, directory, args, profiles_sha256)
        )
        variant_studies[variant] = directory / "public-study.json"

    # One collection at a time, headline first: each takes every model's full
    # concurrency ceiling.
    through_collection.append(_collect(study_path, bundle, args))
    through_collection.extend(
        _collect(path, bundle, args) for path in variant_studies.values()
    )

    study_paths = (study_path, *variant_studies.values())

    def collected() -> dict[str, Path]:
        """Each study's plan and each of its captures' answers."""
        return {
            path.relative_to(output).as_posix(): path
            for study in study_paths
            for path in (
                study,
                *(
                    capture / "contextual-risk.json"
                    for _, capture in public_study_collections(study)
                ),
            )
        }

    setup = output / "setup"
    setup_manifest = {f"setup/{SETUP_MANIFEST}": setup / SETUP_MANIFEST}
    model_run = output / "model-run"
    results = model_run / "results.json"
    summary = output / "summary"
    export = output / "export"
    reporting = [
        Step(
            _cli_command(
                "setup",
                "import",
                "--study",
                study_path,
                "--setup-id",
                "hypothetical-frontier-lab",
                "--comparison-family",
                "all",
                *(
                    argument
                    for path in variant_studies.values()
                    for argument in ("--variant-study", path)
                ),
                "--output",
                setup,
            ),
            setup,
            load_setup,
            {"variants": [variant.value for variant in variants]},
            collected,
        ),
        Step(
            _cli_command("run", "--setup", setup, "--output", model_run),
            model_run,
            _check_model_run,
            upstream=lambda: setup_manifest,
        ),
        Step(
            _cli_command(
                "results", "summary", "--results", results, "--output", summary
            ),
            summary,
            _check_summary,
            upstream=lambda: {
                f"model-run/{path.name}": path
                for path in (results, diagnostics_path(results))
            },
        ),
        Step(
            _cli_command(
                "results",
                "export",
                "--setup",
                setup,
                "--capture",
                model_run / "capture",
                "--output",
                export,
            ),
            export,
            load_results_export,
            upstream=lambda: {
                **setup_manifest,
                f"model-run/capture/{CAPTURE_MANIFEST}": model_run
                / "capture"
                / CAPTURE_MANIFEST,
            },
        ),
    ]

    if args.resume:
        refusal = _changed_arguments([*through_collection, *reporting])
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 2

    environment = _environment(args)
    exit_code = _run_all(through_collection, resume=args.resume, env=environment)
    if exit_code:
        return exit_code
    if args.dry_run:
        print(
            "Dry run complete: every initial prompt is frozen under "
            f"{output}/*/collections/*/calls/*/request.json; no provider was called."
        )
        return 0

    validation = [
        Step(_cli_command("capability-risk", "validate", "--run-directory", capture))
        for path in study_paths
        for _, capture in public_study_collections(path)
    ]
    return _run_all([*validation, *reporting], resume=args.resume, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
