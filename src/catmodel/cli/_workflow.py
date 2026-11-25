"""Credential-free setup, simulation and results commands."""

import json
from collections.abc import Mapping
from pathlib import Path

import click
from click.core import ParameterSource

from catmodel.cli._helpers import (
    DIRECTORY,
    FILE,
    capture_option,
    fail,
    global_options,
    output_option,
    setup_option,
)
from catmodel.json_io import strict_json_loads

_PUBLIC_STUDY_ONLY = (
    "comparison_families",
    "variant_studies",
)


def register(cli: click.Group) -> None:
    cli.add_command(setup)
    cli.add_command(run)
    cli.add_command(results)


@click.group("setup")
def setup() -> None:
    """Freeze and inspect a run setup without new collection."""


@click.group("results")
def results() -> None:
    """Derive results from immutable run captures without simulation."""


@setup.command("build")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=FILE,
    help="Authored setup inputs in the inline layout, such as "
    "config/setups/golden-poisson/setup.json; not a saved setup's setup.json.",
)
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=FILE,
    help="Execution plan JSON: arms, views, simulations, seed and numerical controls.",
)
@click.option("--setup-id", required=True, help="Identifier of the new setup.")
@click.option(
    "--evidence",
    type=DIRECTORY,
    help="Directory of the evidence files the inputs cite; every file is "
    "frozen in the setup. Default: none.",
)
@output_option("directory", "the setup")
def build(
    spec_path: Path,
    plan_path: Path,
    setup_id: str,
    evidence: Path | None,
    output: Path,
) -> None:
    """Freeze authored scientific inputs, execution controls and evidence."""
    from catmodel.contracts._setup_codec import decode_scientific
    from catmodel.contracts.incident import ExecutionPlan
    from catmodel.setup.store import build_setup
    from catmodel.stored_files import DiskFile, StoredFiles

    try:
        selected = {}
        if evidence is not None:
            for path in sorted(evidence.rglob("*")):
                if path.is_symlink():
                    raise ValueError("selected evidence rejects symlinks")
                if path.is_file():
                    selected[path.relative_to(evidence).as_posix()] = DiskFile(path)
        package = build_setup(
            destination=output,
            setup=decode_scientific(
                strict_json_loads(spec_path.read_bytes()),
                ExecutionPlan.model_validate(strict_json_loads(plan_path.read_bytes())),
                setup_id=setup_id,
                evidence_files=StoredFiles(selected),
            ),
        )
        click.echo(str(package.root))
    except (ValueError, OSError) as exc:
        fail(exc)


@setup.command("import")
@click.option(
    "--study",
    type=FILE,
    help="A public study's public-study.json: import its paired trajectory "
    "captures, the headline route.",
)
@click.option(
    "--run-directory",
    type=DIRECTORY,
    help="A fixed-anchor collection's directory: import it for conditional "
    "what-if analysis at fixed capability.",
)
@click.option("--setup-id", required=True, help="Identifier of the new setup.")
@output_option("directory", "the setup")
@click.option(
    "--comparison-family",
    "comparison_families",
    multiple=True,
    help="Comparison family to execute (repeatable), replacing the default "
    "public_forecast, model_pooling_leave_one_out, tail_shape, "
    "uncapped_severity, exposure_scale and safeguard_level; `all`, alone, "
    "selects the whole catalog. Unselected families are recorded unavailable. "
    "Public studies only.",
)
@click.option(
    "--variant-study",
    "variant_studies",
    multiple=True,
    type=FILE,
    help="A variant study's public-study.json (repeatable): an exposure_scale or "
    "safeguard_level study of the same forecast on the headline's embedded "
    "rule. Public studies only.",
)
def import_setup(
    study: Path | None,
    run_directory: Path | None,
    setup_id: str,
    output: Path,
    comparison_families: tuple[str, ...],
    variant_studies: tuple[Path, ...],
) -> None:
    """Replay captured answers into a setup without new collection.

    Import a public study (--study), whose trajectory nodes the setup
    integrates with their quadrature weights beside its comparison arms, or
    one fixed-anchor collection (--run-directory), with a conditional kernel
    at every integer ECI score. The setup freezes the default numerical
    controls, among them plan seed 42, the tail gate's caps on simulated
    state-years (NUM-TAIL-GATE) and the work allowances (NUM-BUDGETS). A
    public study compiles in the global --workers processes; the setup does
    not depend on the count.
    """
    if (study is None) == (run_directory is None):
        raise click.UsageError("Give one of --study or --run-directory.")
    context = click.get_current_context()
    if run_directory is not None:
        flags = {param.name: param.opts[0] for param in context.command.params}
        given = [
            flags[name]
            for name in _PUBLIC_STUDY_ONLY
            if context.get_parameter_source(name) is ParameterSource.COMMANDLINE
        ]
        if given:
            raise click.UsageError(
                f"{given[0]} applies to public studies (--study) only."
            )
    try:
        if run_directory is not None:
            from catmodel.setup.contextual_import import import_contextual_setup

            package = import_contextual_setup(
                source=run_directory, destination=output, setup_id=setup_id
            )
            click.echo(str(package.root))
            return
        from catmodel.setup.public_study import import_public_frontier_study

        assert study is not None
        import_public_frontier_study(
            source=study,
            setup_id=setup_id,
            destination=output,
            comparison_families=_comparison_families(comparison_families),
            variant_studies=variant_studies,
            workers=global_options().workers,
        )
        click.echo(str(output))
    except (ValueError, OSError) as exc:
        fail(exc)


def _comparison_families(named: tuple[str, ...]) -> tuple[str, ...] | None:
    """Named families, the whole catalog for `all`, or None for the default."""
    if "all" not in named:
        return named or None
    if len(named) > 1:
        raise ValueError("--comparison-family all selects every family; use it alone")
    from catmodel.setup.sensitivity_catalog import SENSITIVITY_FAMILIES

    return tuple(row.family_id for row in SENSITIVITY_FAMILIES)


@setup.command("inspect")
@setup_option
def inspect(setup_path: Path) -> None:
    """Validate a setup and print its scope, evidence, arms, controls and workload."""
    from catmodel.results.execution import load_with_workload

    try:
        package, workload = load_with_workload(setup_path)
        click.echo(
            json.dumps(
                {
                    "setup_id": package.manifest.setup_id,
                    "setup_manifest_sha256": package.manifest_sha256,
                    "scope_id": package.scope.scope_id,
                    "scope": package.scope.model_dump(mode="json"),
                    "evidence": [
                        row.model_dump(mode="json") for row in package.evidence.records
                    ],
                    "assumptions": package.assumptions.model_dump(mode="json"),
                    "joint_state_ids": [
                        state.state_id for state in package.joint_states
                    ],
                    "execution": package.execution.model_dump(mode="json"),
                    "workload": workload,
                    "excluded_samples": next(
                        (
                            row.acquisition["excluded_samples"]
                            for row in package.evidence.records
                            if row.acquisition and "excluded_samples" in row.acquisition
                        ),
                        None,
                    ),
                },
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
    except (ValueError, OSError) as exc:
        fail(exc)


@click.command("run")
@setup_option
@output_option(
    "directory",
    "the run capture, results.json, its diagnostics file "
    "results.diagnostics.json and tornado.svg",
)
def run(setup_path: Path, output: Path) -> None:
    """Simulate every required arm and state with the setup's frozen controls.

    The only command that simulates. Each tail gate round's states, and then
    the arms' views, are processed in the global --workers processes, no more
    than available memory holds. The numbers and identities in the results do
    not depend on the count, which results.json records under
    execution_context.
    """
    from catmodel.results.execution import IncompleteRun, run_setup

    try:
        run_setup(
            setup_path=setup_path, output=output, workers=global_options().workers
        )
        click.echo(str(output / "results.json"))
    except (ValueError, OSError, IncompleteRun) as exc:
        fail(exc)


def _read_weights(path: Path | None) -> tuple[Mapping | None, Mapping | None]:
    """A --weights file as replay's ``state_weights`` and
    ``state_weights_by_arm``: an object of state probabilities is the first,
    an object of such objects the second."""
    if path is None:
        return None, None
    weights = strict_json_loads(path.read_bytes())
    if isinstance(weights, Mapping) and weights:
        nested = [isinstance(vector, Mapping) for vector in weights.values()]
        if all(nested):
            return None, weights
        if not any(nested):
            return weights, None
    raise ValueError(
        "target weights must be a JSON object in one of two shapes: every "
        "declared state ID with its target probability, or every required arm "
        "ID with such an object of state probabilities"
    )


def _reduction_spec(levels: tuple[float, ...]):
    """The default tail levels with any requested ones added."""
    if not levels:
        return None
    from catmodel.results.reduction import DEFAULT_TAIL_LEVELS, ReductionSpec

    return ReductionSpec(levels=tuple(sorted({*DEFAULT_TAIL_LEVELS, *levels})))


@results.command("replay")
@setup_option
@capture_option
@click.option(
    "--weights",
    type=FILE,
    help="JSON object of target weights for the captured states: every "
    "declared state ID with its target probability or, to give each arm its "
    "own, every required arm ID with such an object. Default: the setup's "
    "weights, with the state weights an arm authors for itself.",
)
@click.option(
    "--level",
    "levels",
    multiple=True,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    help="Tail level whose VaR and TVaR to report besides the defaults 0.9, "
    "0.99, 0.995, 0.996 and 0.999 (repeatable).",
)
@output_option("directory", "results.json and its results.diagnostics.json")
def replay(
    setup_path: Path,
    capture_directory: Path,
    weights: Path | None,
    levels: tuple[float, ...],
    output: Path,
) -> None:
    """Derive results from a run capture with the current reducer, without
    simulating.

    --weights changes the joint states' weights without changing their
    conditional laws, and refuses weight on a state outside an arm's captured
    support. A file of state probabilities reweights a setup in which no arm
    authors its own state weights. Where an arm does, as in a public study's
    setup, the file gives every required arm its own state probabilities,
    keyed by arm ID. The captured laws are checked and the views reduced in
    the global --workers processes.
    """
    from catmodel.results.execution import replay_capture

    try:
        state_weights, state_weights_by_arm = _read_weights(weights)
        # The directory is created only once the results are derived.
        replay_capture(
            setup_path=setup_path,
            capture_directory=capture_directory,
            output=output / "results.json",
            state_weights=state_weights,
            state_weights_by_arm=state_weights_by_arm,
            reduction_spec=_reduction_spec(levels),
            workers=global_options().workers,
        )
        click.echo(str(output / "results.json"))
    except (ValueError, OSError) as exc:
        fail(exc)


@results.command("summary")
@click.option(
    "--results",
    "results_path",
    required=True,
    type=FILE,
    help="results.json from run or results replay, with its diagnostics file "
    "beside it.",
)
@output_option("directory", "summary.md and summary.csv")
def summary(results_path: Path, output: Path) -> None:
    """Write a readable summary of one results file without recomputing it.

    summary.md and the long-format summary.csv hold EAL by scenario and by
    reporting domain, each domain's expected numbers of large incidents where
    the results hold them, the sensitivity exhibits and every unavailable item
    with its reason. They also hold aggregate and occurrence losses at
    1-in-10, 1-in-100, 1-in-200 and 1-in-250, VaR and TVaR, which summary.md
    labels illustrative.
    """
    from catmodel.results.results_summary import write_results_summary

    try:
        click.echo(str(write_results_summary(results_path=results_path, output=output)))
    except (ValueError, OSError) as exc:
        fail(exc)


@results.command("export")
@setup_option
@capture_option
@click.option(
    "--results",
    "results_path",
    type=FILE,
    help="results.json that run or results replay derived from this setup and "
    "capture, with its diagnostics file beside it, to export in place of the "
    "results the capture reduces to under the setup's plan. Refused unless it "
    "reproduces from the capture. Default: reduce the capture.",
)
@click.option(
    "--include-answers",
    is_flag=True,
    help="Copy the setup whole, with the captured prompts, model responses "
    "and parsed answers under evidence/captured/. Default: leave those files "
    "out and list each by path and SHA-256 in the copy's "
    "omitted-evidence.json.",
)
@output_option(
    "directory",
    "export.json, the capability-scenario files, copies of the setup and "
    "capture, and the results",
)
def export(
    setup_path: Path,
    capture_directory: Path,
    results_path: Path | None,
    include_answers: bool,
    output: Path,
) -> None:
    """Write a portable, checksummed export of a run's results.

    The results are those the capture reduces to under the setup's plan or,
    with --results, a results file derived from the same setup and capture,
    such as a replay at other weights or with added tail levels, which must
    reproduce from the capture. Writes export.json and the hashed
    capability_scenarios.json, capability_scenarios.csv and
    capability_nodes.csv beside copies of the native setup and capture and
    the results (results.json, and a diagnostics file with the exhibits),
    reducing in the global --workers processes. Each available series gives
    P(loss > x) at x = 0 and at each percentile of its simulated annual
    losses, at most 102 points.

    The setup's copy leaves out the captured prompts, model responses and
    parsed answers of an imported setup and records each file's path and
    SHA-256, so the export still verifies; --include-answers copies them.
    """
    from catmodel.results.results_export import export_results

    try:
        export_results(
            setup_path=setup_path,
            capture_directory=capture_directory,
            results_path=results_path,
            output=output,
            workers=global_options().workers,
            include_answers=include_answers,
        )
        click.echo(str(output / "export.json"))
    except (ValueError, OSError) as exc:
        fail(exc)
