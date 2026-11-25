"""Commands for public-frontier ECI forecasting and context-bound risk judgments."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import click

from catmodel.cli._answers import ANSWERS, ANSWERS_HELP
from catmodel.cli._helpers import DIRECTORY, FILE, global_options, output_option
from catmodel.json_io import strict_json_loads
from catmodel.setup.sensitivity_catalog import (
    EXPOSURE_SCALE_VARIANTS,
    SAFEGUARD_LEVELS,
)

VARIANT_SAMPLES = 3
"""Samples per model in a variant study (``public-study --variant``), whatever
the profile's samples_per_question: each variant is its own public study on the
headline's embedded rule, at three samples per model (runbook section 2)."""

_SCENARIO = click.option(
    "--scenario",
    "scenario_ids",
    multiple=True,
    help="Scenario to ask about (repeatable, in any order); `all`, alone, "
    "selects the complete catalog. A selection of the complete catalog, by "
    "`all` or by naming every scenario, also asks its taxonomy parents. Omit "
    "it when the context lists its own nodes.",
)


def _trajectory_nodes(
    _context: click.Context, _parameter: click.Parameter, value: int | None
) -> int | None:
    """Refuse a --nodes count the trajectory rule cannot take."""
    from catmodel.forecast.public_frontier_quadrature import require_trajectory_nodes

    if value is None:
        return None
    try:
        return require_trajectory_nodes(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def register(cli: click.Group) -> None:
    for command in (capability_forecast, capability_risk):
        cli.add_command(command)


def _write_json(payload: object, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if output is None:
        click.echo(rendered, nl=False)
    else:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
        click.echo(str(output))


def profile_respondents(profile: str, samples: int | None = None):
    """The respondents of a model profile: every model of the profile, in
    roster order, with ``samples`` samples, or the profile's samples per
    question."""
    from catmodel.config import load_model_profile
    from catmodel.contracts.elicitation import Respondent

    model_profile = load_model_profile(profile=profile)
    count = samples or model_profile.settings.samples_per_question
    return tuple(
        Respondent(
            respondent_id=f"{spec.model_id}:{sample}",
            model_id=spec.model_id,
            sample_index=sample,
        )
        for spec in model_profile.models
        for sample in range(count)
    )


@click.group("capability-forecast")
def capability_forecast() -> None:
    """Forecast the public ECI frontier and plan trajectory studies."""


@click.group("capability-risk")
def capability_risk() -> None:
    """Build, collect and validate context-bound risk judgments."""


@capability_risk.command("build-instrument")
@click.option(
    "--source-bundle",
    required=True,
    type=DIRECTORY,
    help="Source bundle whose manifest the instrument binds and whose source "
    "cards its questions show.",
)
@click.option(
    "--context",
    required=True,
    type=FILE,
    help="Context JSON: the lab, AI model and deployment, capability scale, "
    "period and assumptions, with every instrument field except the respondents "
    "and the source bundle's digest.",
)
@_SCENARIO
@click.option(
    "--source-status",
    type=click.Choice(["public", "not_yet_public"]),
    help="Source status whose instrument to build from a context whose "
    "source_statuses describe both.",
)
@output_option("file", "the instrument JSON")
def build_instrument(
    source_bundle: Path,
    context: Path,
    scenario_ids: tuple[str, ...],
    source_status: str | None,
    output: Path,
) -> None:
    """Render questions about one named lab, AI model and deployment.

    Writes one instrument, such as a fixed-anchor instrument for collect
    --instrument; public-study renders a trajectory study's instruments
    itself. The respondents are every model of the global --profile, at the
    profile's samples_per_question.
    """
    from catmodel.setup.contextual_inputs import build_contextual_instrument

    try:
        payload = strict_json_loads(context.read_bytes())
        if not isinstance(payload, dict):
            raise ValueError("context must contain a JSON object")
        instrument = build_contextual_instrument(
            payload=payload,
            source_bundle=source_bundle,
            respondents=profile_respondents(global_options().profile),
            scenario_ids=scenario_ids,
            source_status=source_status,
        )
        _write_json(instrument.model_dump(mode="json"), output)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@capability_risk.command("collect")
@click.option(
    "--study",
    type=FILE,
    help="A public study's public-study.json: collect every context it plans, "
    "into the capture directories it names.",
)
@click.option(
    "--instrument",
    type=FILE,
    help="One instrument to collect, such as a fixed-anchor instrument, into "
    "--run-directory.",
)
@click.option(
    "--run-directory",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for --instrument's collection: a new one, or one to resume.",
)
@click.option(
    "--source-bundle",
    required=True,
    type=DIRECTORY,
    help="Source bundle the instruments bind; each collection keeps a copy.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Freeze every collection plan and initial prompt, then stop without "
    "calling a provider.",
)
@click.option("--answers", type=ANSWERS, help=ANSWERS_HELP)
def collect(
    study: Path | None,
    instrument: Path | None,
    run_directory: Path | None,
    source_bundle: Path,
    dry_run: bool,
    answers: Path | None,
) -> None:
    """Freeze plans and prompts, then collect answers from the respondent models.

    Collect a public study (--study) or one instrument (--instrument and
    --run-directory) with the global --profile. Every context sends its
    requests through one lane per model, bounded by the profile's limits.
    A study is collected in up to the global --workers collector processes,
    which divide its contexts and each model's limit among them: one process
    serves its requests on one CPU core. The frozen files do not depend on
    --workers. Rerunning the same command resumes from the stored calls. A
    dry run with --answers ends by saying how many of the frozen requests the
    table already holds, which a collection with it would not send.
    """
    from catmodel.collection.contextual_collection import (
        run_contextual_collection,
        run_study_collection,
    )
    from catmodel.contracts.contextual_risk import ContextualRiskInstrument

    if (study is None) == (instrument is None):
        raise click.UsageError("Give --study, or --instrument with --run-directory.")
    if (instrument is None) != (run_directory is None):
        raise click.UsageError(
            "--run-directory goes with --instrument; a study's plan names its "
            "capture directories."
        )
    profile = global_options().profile
    try:
        if study is not None:
            artifacts = run_study_collection(
                study=study,
                source_bundle=source_bundle,
                profile=profile,
                dry_run=dry_run,
                answers=answers,
                workers=global_options().workers,
            )
            if artifacts is not None:
                click.echo(f"Collected {len(artifacts)} contexts")
                return
            validated = "Validated frozen collection plans"
            if answers is not None:
                from catmodel.setup.public_study import public_study_collections

                validated += _held_by_table(
                    answers, (capture for _, capture in public_study_collections(study))
                )
            click.echo(validated)
            return
        assert instrument is not None and run_directory is not None
        run_contextual_collection(
            instrument=ContextualRiskInstrument.from_saved(
                strict_json_loads(instrument.read_bytes())
            ),
            source_bundle=source_bundle,
            run_directory=run_directory,
            profile=profile,
            dry_run=dry_run,
            answers=answers,
        )
        if not dry_run:
            click.echo(run_directory / "contextual-risk.json")
            return
        validated = "Validated frozen collection plan"
        if answers is not None:
            validated += _held_by_table(answers, [run_directory])
        click.echo(validated)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


def _held_by_table(answers: Path, run_directories: Iterable[Path]) -> str:
    """What a dry run with an answer table adds to its report: how many of
    the requests frozen in ``run_directories`` the table already answers."""
    from catmodel.collection.answer_table import AnswerTable

    table = AnswerTable(answers)
    held = total = 0
    for run_directory in run_directories:
        found, requests = table.coverage(run_directory)
        held += found
        total += requests
    return f"; the answer table holds {held} of {total} frozen requests"


@capability_risk.command("validate")
@click.option(
    "--run-directory",
    required=True,
    type=DIRECTORY,
    help="A collection's directory, such as one under a public study's collections/.",
)
def validate_capability_risk(run_directory: Path) -> None:
    """Replay every raw response against its exact prompt, offline.

    Prints the complete and unanswered cases, the exhausted calls, the GPD tail
    shapes each respondent's p999 implies, each model's call-limit use and, by
    scenario and model, the methods the answers state and the excerpts and
    threat model they cite. As advance notice of what setup import decides
    over a whole study, it also lists each scenario that a model left
    unanswered in more than the excluded-sample control's share, 10%, of
    this collection's samples (unanswered_scenarios): import leaves such a
    model out of that scenario if the same holds over the whole study, and
    out of a variant study's scenario also where it holds over the headline
    study.
    """
    from catmodel.elicitation.contextual_risk import (
        contextual_tail_diagnostics,
        reasoning_use,
    )
    from catmodel.frozen_calls import call_limit_usage
    from catmodel.setup.contextual_inputs import load_contextual_collection
    from catmodel.setup.excluded_samples import unanswered_pairs

    try:
        contextual, artifact, files = load_contextual_collection(run_directory)
        _write_json(
            {
                "evidence_status": artifact.evidence_status,
                "complete_cases": len(artifact.cases),
                "unanswered_cases": [
                    list(row.key()) for row in artifact.unanswered_cases
                ],
                "unanswered_scenarios": [
                    {
                        "node_id": pair.leaf,
                        "model_id": pair.model,
                        "samples": pair.samples,
                        "lacking_samples": pair.lacking,
                        "lacking_share": pair.share,
                    }
                    for pair in unanswered_pairs(
                        {contextual.context_id: ((contextual, artifact),)}
                    )
                ],
                "exhausted_calls": [asdict(row) for row in artifact.exhausted_calls()],
                "p999_implied_gpd_shapes": [
                    asdict(row)
                    for row in contextual_tail_diagnostics(artifact, contextual)
                ],
                "call_limits": call_limit_usage(files),
                "reasoning_use": [
                    asdict(row) for row in reasoning_use(artifact, contextual)
                ],
            },
            None,
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@capability_forecast.command("public-study")
@click.option(
    "--source-bundle",
    required=True,
    type=DIRECTORY,
    help="Source bundle whose manifest every instrument binds and whose source "
    "cards the questions show; it captures Epoch's ECI model-score table for a "
    "forecast without releases.",
)
@click.option(
    "--forecast",
    "forecast_path",
    required=True,
    type=FILE,
    help="Public-frontier forecast JSON. One without releases, such as "
    "data/capability/public-frontier-forecast-template.json, takes them from "
    "the bundle's Epoch ECI table; one that states its releases is used as it "
    "is.",
)
@click.option(
    "--context",
    required=True,
    type=FILE,
    help="Context JSON whose source_statuses describe the lab's public and "
    "not-yet-public models and deployments, with every instrument field except "
    "the respondents, the source bundle's digest and the capability scale and "
    "anchors, which the study takes from the forecast.",
)
@_SCENARIO
@click.option(
    "--variant",
    type=click.Choice([*EXPOSURE_SCALE_VARIANTS, *SAFEGUARD_LEVELS]),
    help="Plan a variant study, at three samples per model: half or double "
    "multiplies every path-linked deployment scale (exposure_scale); weaker or "
    "stronger replaces the stipulated safeguards with that level, authored in "
    "the context's safeguard_levels (safeguard_level). A variant study is "
    "planned on its headline's embedded rule, the (n - 1) / 2 nodes of an "
    "n-node headline: three by default, as the default seven-node headline "
    "needs, or --nodes. Only a headline of 7, 11, 15, ... nodes can have "
    "variant studies.",
)
@click.option(
    "--assumptions",
    "assumptions_path",
    required=True,
    type=FILE,
    help="JSON holding the study's exposure_rationale, "
    "residual_independence_rationale and parameter_coupling_rationale.",
)
@click.option(
    "--nodes",
    type=int,
    callback=_trajectory_nodes,
    help="Interior trajectory nodes, an odd number of at least 3. By default "
    "seven for a headline study and three for a variant study. The variant "
    "studies of an n-node headline take (n - 1) / 2 nodes, itself an odd "
    "number of at least 3, so a headline that will have variant studies needs "
    "7, 11, 15, ... nodes.",
)
@output_option(
    "directory",
    "the study plan public-study.json, the fitted forecast and the paired instruments",
)
def public_frontier_study(
    source_bundle: Path,
    forecast_path: Path,
    context: Path,
    scenario_ids: tuple[str, ...],
    variant: str | None,
    assumptions_path: Path,
    nodes: int | None,
    output: Path,
) -> None:
    """Plan a public-frontier trajectory study without calling a provider.

    Renders the context's questions about incidents from publicly deployed
    and from not-yet-public source models for every model of the global
    --profile, at the profile's samples_per_question, or at three samples per
    model for a variant study (--variant). Fits the forecast
    (trend-posterior.json), places trajectory quadrature nodes over the
    context's policy period on the public frontier, the maximum of every
    forecast lab's path, and writes a pair of instruments per node. By default
    a headline study has seven interior nodes and a variant study the embedded
    three-node rule's; --nodes plans another rule, such as the 15-node
    headline of the hypothetical-lab study. Each adds a no-progress node.
    """
    from catmodel.setup.public_study import plan_public_frontier_study

    try:
        result = plan_public_frontier_study(
            source_bundle=source_bundle,
            forecast_path=forecast_path,
            context_path=context,
            respondents=profile_respondents(
                global_options().profile, VARIANT_SAMPLES if variant else None
            ),
            scenario_ids=scenario_ids,
            variant=variant,
            assumptions_path=assumptions_path,
            nodes=nodes,
            output=output,
        )
        click.echo(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
