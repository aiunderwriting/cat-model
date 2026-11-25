"""The committed hypothetical-lab study builds offline on the headline route."""

import hashlib
import json
import math
import re
import shutil
from itertools import pairwise

import numpy as np
import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.cli._capability import profile_respondents
from catmodel.collection.contextual_collection import run_contextual_collection
from catmodel.contracts.contextual_risk import (
    ContextualRiskInstrument,
    ContextualStageAnswer,
    PathLinkedExposure,
    ProbabilityQuantiles,
    path_growth_multiple,
    render_exposure,
    scale_text,
)
from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.contracts.public_frontier import (
    PublicFrontierForecast,
    PublicFrontierQuadrature,
    progress_elapsed_years,
    quadrature_node_ids,
)
from catmodel.elicitation.contextual_frequency import (
    ContextualChainPrompt,
    chain_count_law_for_answer,
)
from catmodel.evidence.source_bundle import SOURCE_CARDS_PATH
from catmodel.forecast.public_frontier_epoch import EPOCH_ECI_SCORES_PATH
from catmodel.llm.transport import ProviderResponse
from catmodel.resources import config_path, data_path, scenarios_path
from catmodel.results.execution import setup_workload
from catmodel.scenarios.hierarchy_contract import (
    cached_hierarchy_contract,
    children_by_id,
)
from catmodel.scenarios.loader import load_scenarios
from catmodel.setup.contextual_inputs import (
    CAPABILITY_ANCHOR_FIELDS,
    build_contextual_instrument,
)
from catmodel.setup.public_study import (
    import_public_frontier_study,
    plan_public_frontier_study,
    prepare_public_forecast_file,
    public_study_templates,
    write_public_frontier_study,
)
from catmodel.yaml_io import safe_load
from test_support.capability_sources import authored_source_files, write_source_bundle
from test_support.contextual_risk import (
    chain_answer,
    contextual_instrument,
    fixture_response,
    governed_contextual_instrument,
    hierarchy_instrument,
    path_linked_payload,
    scenario_node_answer,
)
from test_support.public_frontier import synthetic_forecast
from tests.test_public_frontier_study import _study

STUDY = config_path("studies", "hypothetical-frontier-lab")
SOFTWARE = "output_reliance_software"
HYPOTHETICAL = "The lab, model and deployment described below are hypothetical"


def _run(*args):
    result = CliRunner().invoke(cli, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output
    return result.output


def _shared_context():
    """The synthetic lab context with both source statuses authored once."""
    payload = contextual_instrument().model_dump(
        mode="json", exclude={"nodes", "respondents", "source_bundle_manifest_sha256"}
    )
    statuses = {}
    for status in ("public", "not_yet_public"):
        model, deployment = f"model_{status}", f"deployment_{status}"
        statuses[status] = {
            "model": {**payload["model"], "entity_id": model},
            "deployment": {**payload["deployment"], "entity_id": deployment},
            "deployment_binding": {
                **payload["deployment_binding"],
                "model_id": model,
                "deployment_id": deployment,
            },
            "capability_access": "public" if status == "public" else "internal",
        }
    for field in ("model", "deployment", "deployment_binding", "capability_access"):
        payload.pop(field)
    return {**payload, "source_statuses": statuses}


def test_source_status_builds_each_template_from_one_shared_context(tmp_path):
    bundle = write_source_bundle(tmp_path / "source")
    respondents = contextual_instrument().respondents

    def build(context, status=None):
        return build_contextual_instrument(
            context,
            source_bundle=bundle,
            respondents=respondents,
            scenario_ids=("output_reliance_legal",),
            source_status=status,
        )

    context = _shared_context()
    public, unreleased = build(context, "public"), build(context, "not_yet_public")
    assert (public.model.entity_id, public.capability_access) == (
        "model_public",
        "public",
    )
    assert unreleased.nodes[0].population.deployment_ids == (
        "deployment_not_yet_public",
    )
    assert public.lab == unreleased.lab and public.period == unreleased.period
    with pytest.raises(ValueError, match="requires a source status"):
        build(context)
    # The fixed-anchor builder needs the scale and anchors a public study's
    # context may leave to its forecast.
    for field in CAPABILITY_ANCHOR_FIELDS:
        with pytest.raises(ValueError, match=f"does not state {field}, which"):
            build({k: v for k, v in context.items() if k != field}, "public")
    with pytest.raises(ValueError, match="must author fields"):
        build({**context, "source_statuses": {}}, "public")
    with pytest.raises(ValueError, match="must author fields"):
        build(
            {**context, "source_statuses": {"public": {}, "not_yet_public": {}}},
            "public",
        )
    plain = contextual_instrument().model_dump(
        mode="json", exclude={"nodes", "respondents", "source_bundle_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="requires a context with source_statuses"):
        build(plain, "public")
    with pytest.raises(ValueError, match="authored once"):
        build({**context, "capability_access": "public"}, "public")
    uneven = json.loads(json.dumps(context))
    uneven["source_statuses"]["public"].pop("capability_access")
    with pytest.raises(ValueError, match="same fields"):
        build({**uneven, "capability_access": "public"}, "public")


def test_questions_name_exactly_the_hypothetical_entities():
    from catmodel.elicitation.contextual_risk import contextual_prompts

    def condition(statuses):
        payload = contextual_instrument().model_dump(mode="json")
        for role, status in zip(("lab", "model", "deployment"), statuses, strict=True):
            payload[role]["status"] = status
        instrument = ContextualRiskInstrument.model_validate(payload)
        return contextual_prompts(instrument)[0].render().split("\n\n", 1)[0]

    assert "hypothetical" not in condition(("actual", "actual", "actual"))
    assert condition(("hypothetical", "actual", "actual")).startswith(
        "The lab described below is hypothetical: stipulated for this study"
    )
    assert condition(("hypothetical",) * 3).startswith(HYPOTHETICAL)


def _run_prompt(instrument):
    from catmodel.elicitation.contextual_risk import contextual_prompts

    return contextual_prompts(instrument)[0].render()


def _hypothetical_path_exposure(progress_ratio):
    context = json.loads(
        config_path("studies", "hypothetical-frontier-lab", "context.json").read_text()
    )
    return PathLinkedExposure.model_validate(
        {
            **context["path_linked_exposure"],
            "progress_ratio": progress_ratio,
        }
    )


def test_path_linked_scales_render_two_significant_figures_and_refuse_leftovers():
    from catmodel.contracts.contextual_risk import (
        scale_text,
    )

    exposure = PathLinkedExposure.model_validate(
        {
            "max_growth_multiple": 3,
            "scales": [
                {
                    "scale_id": "users",
                    "unit": "users",
                    "as_of": 5e8,
                    "study_year_median": 6e8,
                },
                {
                    "scale_id": "staff",
                    "unit": "staff",
                    "as_of": 6000,
                    "study_year_median": 8000,
                },
            ],
            "progress_ratio": 2.144,
        }
    )
    assert (
        render_exposure("{users} users and {staff} staff", exposure)
        == "700 million users and 10,000 staff"
    )
    payload = contextual_instrument().model_dump(mode="json")
    del payload["fixed_exposure"]
    payload["path_linked_exposure"] = exposure.model_dump(mode="json")
    with pytest.raises(ValueError, match="exactly on a public trajectory"):
        ContextualRiskInstrument.model_validate(payload)
    del payload["path_linked_exposure"]["progress_ratio"]
    with pytest.raises(ValueError, match="needs a scale placeholder"):
        ContextualRiskInstrument.model_validate(payload)
    assert [scale_text(value) for value in (1.2, 45, 6500, 2.5e6)] == [
        "1.2",
        "45",
        "6,500",
        "2.5 million",
    ]
    with pytest.raises(ValueError, match="cannot be below its as-of level"):
        PathLinkedExposure.model_validate(
            {
                "max_growth_multiple": 3,
                "scales": [
                    {
                        "scale_id": "users",
                        "unit": "users",
                        "as_of": 6,
                        "study_year_median": 5,
                    }
                ],
            }
        )


def test_path_growth_multiple_has_linear_endpoints_for_several_ceilings():
    for maximum in (1.5, 2, 3, 5):
        assert path_growth_multiple(0, maximum) == 0
        assert path_growth_multiple(1, maximum) == 1


def test_path_growth_multiple_is_monotone_bounded_and_linear_below_median():
    maximum = 5.0
    ratios = np.linspace(0, 50, 1001)
    values = [path_growth_multiple(float(ratio), maximum) for ratio in ratios]
    assert all(left < right for left, right in pairwise(values))
    assert all(value < maximum for value in values)
    assert maximum - path_growth_multiple(1e6, maximum) < 1e-6
    for ratio in np.linspace(0, 1, 101):
        assert path_growth_multiple(float(ratio), maximum) == ratio


def test_path_growth_multiple_is_concave_above_median_with_slope_at_most_one():
    ratios = np.linspace(0, 50, 1001)
    values = [path_growth_multiple(float(ratio), 3) for ratio in ratios]
    assert all(value <= ratio for value, ratio in zip(values, ratios, strict=True))
    assert any(value < ratio for value, ratio in zip(values, ratios, strict=True))

    above_median = np.linspace(1, 20, 381)
    increments = np.diff(
        [path_growth_multiple(float(ratio), 3) for ratio in above_median]
    )
    assert all(left >= right for left, right in pairwise(increments))
    assert increments[0] <= above_median[1] - above_median[0]


def test_hypothetical_lab_fast_trajectory_scales_respect_the_growth_ceiling():
    expected = {
        5.105: ["790 million", "2.8 million", "220 trillion", "12,000"],
        13.15: ["800 million", "2.8 million", "220 trillion", "12,000"],
    }
    for ratio, rendered in expected.items():
        exposure = _hypothetical_path_exposure(ratio)
        assert exposure.max_growth_multiple == 3
        assert [
            scale_text(exposure.value(scale)) for scale in exposure.scales
        ] == rendered
        for scale in exposure.scales:
            ceiling = scale.as_of + exposure.max_growth_multiple * (
                scale.study_year_median - scale.as_of
            )
            assert exposure.value(scale) <= ceiling


def test_hypothetical_lab_median_and_below_median_scales_keep_linear_rendering():
    medians = ["600 million", "1.6 million", "100 trillion", "8,000"]
    for ratio in (0.9999965, 1):
        exposure = _hypothetical_path_exposure(ratio)
        assert [
            scale_text(exposure.value(scale)) for scale in exposure.scales
        ] == medians

    ratio = 0.4665
    exposure = _hypothetical_path_exposure(ratio)
    for scale in exposure.scales:
        linear = scale.as_of + (scale.study_year_median - scale.as_of) * ratio
        assert exposure.value(scale) == float(f"{linear:.2g}")


def test_path_linked_exposure_requires_a_finite_ceiling_above_one():
    data = {
        "scales": [
            {
                "scale_id": "users",
                "unit": "users",
                "as_of": 100,
                "study_year_median": 200,
            }
        ]
    }
    for maximum in (1, 0.5, float("inf")):
        with pytest.raises(ValueError, match="max_growth_multiple"):
            PathLinkedExposure.model_validate({**data, "max_growth_multiple": maximum})
    with pytest.raises(ValueError, match="max_growth_multiple"):
        PathLinkedExposure.model_validate(data)


def _epoch_bundle(root):
    """A bundle capturing the repository's source cards and Epoch ECI table."""
    table = data_path("capability", "epoch_eci_scores.csv").read_bytes()
    provenance = safe_load(
        data_path("capability", "epoch_eci_scores.provenance.yaml").read_text()
    )
    files = {
        **authored_source_files(),
        SOURCE_CARDS_PATH: data_path("elicitation_source_cards.yaml").read_bytes(),
        EPOCH_ECI_SCORES_PATH: table,
    }
    source = {
        **provenance,
        "content_hash": hashlib.sha256(table).hexdigest(),
        "captured_paths": [EPOCH_ECI_SCORES_PATH],
    }
    return write_source_bundle(root, files=files, extra_sources=[source])


def test_committed_hypothetical_lab_builds_a_complete_catalog_public_study(tmp_path):
    bundle = _epoch_bundle(tmp_path / "source")
    template = data_path("capability", "public-frontier-forecast-template.json")
    summary = json.loads(
        _run(
            "--profile",
            "test",
            "capability-forecast",
            "public-study",
            "--source-bundle",
            bundle,
            "--forecast",
            template,
            "--context",
            STUDY / "context.json",
            "--scenario",
            "all",
            "--assumptions",
            STUDY / "public-assembly-assumptions.json",
            "--output",
            tmp_path / "study",
        )
    )
    assert (summary["trajectory_nodes"], summary["conditional_collections"]) == (8, 16)
    assert summary["target_lab"] == "public_frontier"
    # The committed forecast states no releases; the study's forecast is it
    # with the roster labs' releases from the bundle's Epoch ECI table.
    forecast = json.loads((tmp_path / "study" / "forecast.json").read_bytes())
    assert "releases" not in json.loads(template.read_bytes())
    assert {row["lab_id"] for row in forecast["releases"]} == {
        row["lab_id"] for row in forecast["coordinates"]
    }
    catalog = {row.id for row in load_scenarios(scenarios_path())}
    plan = json.loads((tmp_path / "study" / "public-study.json").read_text())
    rendered, chains = {}, {}
    for row in plan["contexts"]:
        instrument = ContextualRiskInstrument.model_validate_json(
            (tmp_path / "study" / row["instrument"]).read_text()
        )
        assert {node.node_id for node in instrument.nodes} == catalog
        assert all(node.frequency_chain is not None for node in instrument.nodes)
        # Every split of the taxonomy holds an estimated scenario.
        assert {row.parent_id for row in instrument.parents} == {
            node
            for node, children in children_by_id(cached_hierarchy_contract()).items()
            if children
        }
        assert instrument.lab.entity_id == "hypothetical_frontier_lab"
        assert {
            entity.status
            for entity in (instrument.lab, instrument.model, instrument.deployment)
        } == {"hypothetical"}
        assert instrument.public_trajectory is not None
        assert instrument.public_trajectory.target_lab == "public_frontier"
        exposure = instrument.path_linked_exposure
        assert exposure is not None and exposure.progress_ratio is not None
        rendered[row["draw_id"], row["source_model_status"]] = (
            exposure.progress_ratio,
            instrument.deployment_binding.use_population,
        )
        # Every derived chain base reads a scale its deployment's
        # use_population states, at the level rendered for this trajectory
        # node; a fixed base is the same stipulated count on every node.
        for node in instrument.nodes:
            assert node.frequency_chain is not None
            stipulated = instrument.chain_exposure(node.frequency_chain)
            if stipulated.scale_id is None:
                assert stipulated.count > 0 and stipulated.scale_level is None
                continue
            scale = next(
                scale
                for scale in exposure.scales
                if scale.scale_id == stipulated.scale_id
            )
            assert stipulated.scale_level == exposure.value(scale)
            assert (
                f"{scale_text(exposure.value(scale))} {scale.unit}"
                in instrument.deployment_binding.use_population
            ), (node.node_id, row["draw_id"])
        software = next(node for node in instrument.nodes if node.node_id == SOFTWARE)
        assert software.frequency_chain is not None
        chains[row["draw_id"], row["source_model_status"]] = (
            instrument,
            software,
            instrument.chain_exposure(software.frequency_chain),
        )
    # The no-progress node keeps the as-of scales; faster nodes carry more use.
    assert rendered["node-00", "public"][0] == 0
    for text in ("500 million weekly", "1 million paying", "40 trillion tokens"):
        assert text in rendered["node-00", "public"][1]
    assert "6,000 lab staff" in rendered["node-00", "not_yet_public"][1]
    ratios = [rendered[draw_id, "public"][0] for draw_id in quadrature_node_ids(7)]
    assert ratios == sorted(ratios) and "{" not in rendered["node-7-8", "public"][1]
    # A node renders each scale's as-of level plus its median-path growth times
    # the saturating link of its progress ratio, its frontier progress over the
    # base law's median, to two significant figures.
    quadrature = PublicFrontierQuadrature.model_validate_json(
        (tmp_path / "study" / "public-frontier.json").read_text()
    )
    center = rendered["node-1-2", "public"][0]
    assert center == pytest.approx(
        quadrature.nodes[4].progress / quadrature.median_progress, rel=1e-12
    )
    growth = center if center <= 1 else 1 + 2 * math.tanh((center - 1) / 2)

    def level(as_of, median):
        return float(f"{as_of + (median - as_of) * growth:.2g}")

    for as_of, median, unit in (
        (500e6, 600e6, "weekly"),
        (1e6, 1.6e6, "paying"),
        (40e12, 100e12, "tokens"),
    ):
        assert (
            f"{scale_text(level(as_of, median))} {unit}"
            in (rendered["node-1-2", "public"][1])
        )
    assert (
        f"{scale_text(level(6000, 8000))} lab staff"
        in (rendered["node-1-2", "not_yet_public"][1])
    )
    # Software reliance states its rate per business customer shipping the
    # lab's code publicly and per staff member shipping code internally; the
    # base count is the node's rendered scale times the authored factor, so
    # faster trajectories carry more.
    levels = {
        key: (exposure.scale_id, exposure.scale_level, exposure.count)
        for key, (_, _, exposure) in chains.items()
    }
    assert levels["node-00", "public"] == ("paying_business_customers", 1e6, 400_000)
    customers = level(1e6, 1.6e6)
    assert levels["node-1-2", "public"] == (
        "paying_business_customers",
        customers,
        float(f"{0.4 * customers:.3g}"),
    )
    assert levels["node-00", "not_yet_public"] == ("lab_staff", 6000, 3_600)
    staff = level(6000, 8000)
    assert levels["node-1-2", "not_yet_public"] == (
        "lab_staff",
        staff,
        float(f"{0.6 * staff:.3g}"),
    )
    assert levels["node-7-8", "public"][1] > 1.6e6
    instrument, software, exposure = chains["node-1-2", "public"]
    rate = chain_answer().model_copy(
        update={
            "opportunity_rate": Quantiles(**dict.fromkeys(QUANTILES, 0.5)),
            "stages": tuple(
                ContextualStageAnswer(
                    stage_id=stage.stage_id,
                    probability=ProbabilityQuantiles(**dict.fromkeys(QUANTILES, 1)),
                )
                for stage in software.frequency_chain.stages
            ),
        }
    )
    assert chain_count_law_for_answer(rate, exposure.count) == Quantiles(
        **dict.fromkeys(QUANTILES, 0.5 * exposure.count)
    )
    question = ContextualChainPrompt(
        instrument, software, instrument.respondents[0], instrument.anchor_scores[0]
    ).render()
    for text in (
        'one base unit is "business customer that ships code written by the '
        f"lab's models to production\"; the stipulated count is {exposure.count:,.0f} base units.",
        "per base unit",
        f"by the stipulated count, {exposure.count:,.0f},",
    ):
        assert text in question
    from catmodel.forecast.public_frontier_quadrature import base_progress_cdf

    elapsed = progress_elapsed_years(quadrature.as_of, quadrature.times)
    assert quadrature.base.pace_shape == 5 and quadrature.base.pace_cap == 4
    assert base_progress_cdf(
        quadrature.base.pace_shape,
        quadrature.posterior,
        elapsed,
        quadrature.median_progress,
        pace_cap=quadrature.base.pace_cap,
    ) == pytest.approx(0.5, abs=1e-9)
    # The context leaves its capability scale and anchors to the study's
    # forecast, so the fixed-anchor builder refuses it and names the study.
    refused = CliRunner().invoke(
        cli,
        [
            *("--profile", "test", "capability-risk", "build-instrument"),
            *("--source-bundle", str(bundle)),
            *("--context", str(STUDY / "context.json")),
            *("--source-status", "public", "--scenario", "all"),
            *("--output", str(tmp_path / "public.json")),
        ],
    )
    assert refused.exit_code != 0 and "build the public study" in refused.output
    assert not (tmp_path / "public.json").exists()
    # A template has no trajectory to set its path-linked scales, so its
    # questions cannot be rendered.
    unconditioned = public_study_templates(
        json.loads((STUDY / "context.json").read_bytes()),
        forecast=PublicFrontierForecast.model_validate(forecast),
        source_bundle=bundle,
        respondents=profile_respondents("test"),
        scenario_ids=(SOFTWARE,),
    )["public"]
    with pytest.raises(ValueError, match="build the public study"):
        _run_prompt(unconditioned)
    row = plan["contexts"][2]
    capture = tmp_path / "study" / row["capture"]
    _run(
        "--profile",
        "test",
        "capability-risk",
        "collect",
        "--instrument",
        tmp_path / "study" / row["instrument"],
        "--source-bundle",
        bundle,
        "--run-directory",
        capture,
        "--dry-run",
    )
    requests = [
        json.loads(path.read_text()) for path in capture.glob("calls/*/request.json")
    ]
    respondents = len(instrument.respondents)
    scenarios = len(load_scenarios(scenarios_path()))
    assert len(requests) == (2 * scenarios + len(instrument.parents)) * respondents
    assert all(HYPOTHETICAL in request["user_prompt"] for request in requests)


def _files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_study_command_plans_what_its_steps_plan_one_by_one(tmp_path):
    """public-study composes three steps: filling the forecast's releases from
    the bundle, building the context's two templates for the profile's
    respondents and planning the study from them. Its plan is theirs, byte for
    byte."""
    bundle = _epoch_bundle(tmp_path / "source")
    template = data_path("capability", "public-frontier-forecast-template.json")
    scenarios = ("fraud_individuals", SOFTWARE)
    steps = tmp_path / "steps"
    steps.mkdir()
    forecast = prepare_public_forecast_file(
        source_bundle=bundle, template=template, output=steps / "forecast.json"
    )
    for status, instrument in public_study_templates(
        json.loads((STUDY / "context.json").read_bytes()),
        forecast=forecast,
        source_bundle=bundle,
        respondents=profile_respondents("test"),
        scenario_ids=scenarios,
    ).items():
        (steps / f"{status}.json").write_text(instrument.model_dump_json())
    write_public_frontier_study(
        forecast_path=steps / "forecast.json",
        public_instrument=steps / "public.json",
        not_yet_public_instrument=steps / "not_yet_public.json",
        assumptions_path=STUDY / "public-assembly-assumptions.json",
        output=tmp_path / "expected",
        nodes=3,
    )
    _run(
        "--profile",
        "test",
        "capability-forecast",
        "public-study",
        "--source-bundle",
        bundle,
        "--forecast",
        template,
        "--context",
        STUDY / "context.json",
        *(option for scenario in scenarios for option in ("--scenario", scenario)),
        "--assumptions",
        STUDY / "public-assembly-assumptions.json",
        "--nodes",
        3,
        "--output",
        tmp_path / "study",
    )
    assert _files(tmp_path / "study") == _files(tmp_path / "expected")
    # Nothing staged for the plan is left beside it.
    assert {path.name for path in tmp_path.iterdir()} == {
        *("source", "steps", "expected", "study")
    }


@pytest.mark.parametrize(
    ("variant", "exposure", "safeguard"),
    [(None, None, None), ("double", "double", None), ("weaker", None, "weaker")],
)
def test_the_planner_plans_the_filled_forecast_and_the_variants_templates(
    tmp_path, variant, exposure, safeguard
):
    """``plan_public_frontier_study`` plans from the forecast with its releases
    filled from the bundle and the context's two templates, built for the
    variant's family; a forecast that states its releases is planned from as
    it is."""
    bundle = _epoch_bundle(tmp_path / "source")
    template = data_path("capability", "public-frontier-forecast-template.json")
    assumptions = STUDY / "public-assembly-assumptions.json"
    respondents = contextual_instrument().respondents
    filled = prepare_public_forecast_file(
        source_bundle=bundle, template=template, output=tmp_path / "forecast.json"
    )
    assert filled.releases
    templates = public_study_templates(
        json.loads((STUDY / "context.json").read_bytes()),
        forecast=filled,
        source_bundle=bundle,
        respondents=respondents,
        scenario_ids=(SOFTWARE,),
        variant=variant,
    )
    scales = templates["public"].path_linked_exposure
    assert scales is not None
    assert (None if scales.variant is None else scales.variant.variant_id) == exposure
    assert templates["public"].safeguard_level == safeguard
    for status, instrument in templates.items():
        (tmp_path / f"{status}.json").write_text(instrument.model_dump_json())
    write_public_frontier_study(
        forecast_path=tmp_path / "forecast.json",
        public_instrument=tmp_path / "public.json",
        not_yet_public_instrument=tmp_path / "not_yet_public.json",
        assumptions_path=assumptions,
        output=tmp_path / "expected",
        nodes=5,
    )
    for name, forecast in (
        ("filled", template),
        ("as-it-is", tmp_path / "forecast.json"),
    ):
        summary = plan_public_frontier_study(
            source_bundle=bundle,
            forecast_path=forecast,
            context_path=STUDY / "context.json",
            respondents=respondents,
            scenario_ids=(SOFTWARE,),
            variant=variant,
            assumptions_path=assumptions,
            nodes=5,
            output=tmp_path / name,
        )
        assert summary["trajectory_nodes"] == 6
        assert _files(tmp_path / name) == _files(tmp_path / "expected")
    assert {path.name for path in tmp_path.iterdir()} == {
        *("source", "forecast.json", "public.json", "not_yet_public.json"),
        *("expected", "filled", "as-it-is"),
    }


def test_the_planner_refuses_a_one_status_context_and_an_unknown_variant(tmp_path):
    bundle = write_source_bundle(tmp_path / "source")
    forecast = tmp_path / "forecast.json"
    forecast.write_text(synthetic_forecast().model_dump_json())
    context = tmp_path / "context.json"

    def plan(payload, variant=None):
        context.write_text(json.dumps(payload))
        return plan_public_frontier_study(
            source_bundle=bundle,
            forecast_path=forecast,
            context_path=context,
            respondents=contextual_instrument().respondents,
            scenario_ids=("output_reliance_legal",),
            variant=variant,
            assumptions_path=STUDY / "public-assembly-assumptions.json",
            output=tmp_path / "study",
        )

    one_status = contextual_instrument().model_dump(
        mode="json", exclude={"nodes", "respondents", "source_bundle_manifest_sha256"}
    )
    for payload in (one_status, [_shared_context()]):
        with pytest.raises(ValueError, match="whose source_statuses describe"):
            plan(payload)
    with pytest.raises(ValueError, match="variant must be one of"):
        plan(_shared_context(), "triple")
    # The synthetic context authors no safeguard levels.
    with pytest.raises(ValueError, match="requires a context with safeguard_levels"):
        plan(_shared_context(), "weaker")
    assert plan(_shared_context())["trajectory_nodes"] == 8
    assert {path.name for path in tmp_path.iterdir()} == {
        *("source", "forecast.json", "context.json", "study")
    }
    # The study takes its capability scale and anchors from the forecast,
    # whether or not the context states them.
    stated = _files(tmp_path / "study")
    shutil.rmtree(tmp_path / "study")
    plan(
        {
            name: value
            for name, value in _shared_context().items()
            if name not in CAPABILITY_ANCHOR_FIELDS
        }
    )
    assert _files(tmp_path / "study") == stated


class _Provider:
    """Synthetic answers for any scenario-derived node; chains never trigger."""

    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}

    async def complete(self, *, user_prompt, **_kwargs):
        found = re.search(r"^## Scenario ([a-z0-9_]+)$", user_prompt, re.MULTILINE)
        assert found is not None
        answer = scenario_node_answer(self.nodes[found.group(1)])
        return ProviderResponse(
            text=fixture_response(user_prompt, answer),
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=5,
            output_tokens=10,
        )


def test_public_import_keeps_fine_nodes_and_checks_path_linked_scales(
    tmp_path, monkeypatch
):
    fine = "enforcement_matter"
    # Both scenarios' chains read weekly users publicly and staff internally.
    payload = path_linked_payload(
        governed_contextual_instrument(("output_reliance_legal", fine)).model_dump(
            mode="json"
        ),
        "consumer_weekly_users",
        "lab_staff",
    )
    template = ContextualRiskInstrument.model_validate(payload)
    provider = _Provider(template.nodes)
    folder, bundle, _ = _study(
        tmp_path, monkeypatch, template=template, providers=[provider]
    )
    plan = json.loads((folder / "public-study.json").read_text())
    row = next(
        row
        for row in plan["contexts"]
        if (row["draw_id"], row["source_model_status"]) == ("node-1-2", "public")
    )
    planned, capture = folder / row["instrument"], folder / row["capture"]
    original = json.loads(planned.read_text())
    template_text = payload["deployment_binding"]["use_population"]
    rendered = (
        "About 200 synthetic users, 200 synthetic consumer_weekly_users, "
        "200 synthetic lab_staff"
    )
    assert original["deployment_binding"]["use_population"] == rendered
    # Edited scale text no longer matches its template at the node's ratio.
    tampered = json.loads(json.dumps(original))
    tampered["deployment_binding"]["use_population"] = (
        "About 900 synthetic users, all of them government agencies"
    )
    with pytest.raises(ValueError, match="template rendered at the progress ratio"):
        ContextualRiskInstrument.model_validate(tampered)
    # Leftover braces in the rendered text are refused.
    tampered["path_linked_exposure"]["use_population_template"] = (
        "About {users} synthetic {kind}"
    )
    tampered["deployment_binding"]["use_population"] = "About 200 synthetic {kind}"
    with pytest.raises(ValueError, match="keeps no placeholder"):
        ContextualRiskInstrument.model_validate(tampered)
    # A consistent edit of one node's template is collected but refused at intake.
    tampered["path_linked_exposure"]["use_population_template"] = (
        template_text + ", all of them government agencies"
    )
    tampered["deployment_binding"]["use_population"] = (
        rendered + ", all of them government agencies"
    )
    planned.write_text(
        ContextualRiskInstrument.model_validate(tampered).model_dump_json()
    )
    for row in plan["contexts"]:
        run_contextual_collection(
            instrument=ContextualRiskInstrument.model_validate_json(
                (folder / row["instrument"]).read_text()
            ),
            source_bundle=bundle,
            run_directory=folder / row["capture"],
            profile="test",
        )

    def imported(name):
        return import_public_frontier_study(
            source=folder / "public-study.json",
            destination=tmp_path / name,
            setup_id="fines",
            years_per_state=4,
        )

    with pytest.raises(ValueError, match="cannot change across public paths"):
        imported("edited-template")
    planned.write_text(json.dumps(original))
    shutil.rmtree(capture)
    run_contextual_collection(
        instrument=ContextualRiskInstrument.model_validate(original),
        source_bundle=bundle,
        run_directory=capture,
        profile="test",
    )
    setup = imported("setup")
    # The import freezes the default tolerance and state-year caps.
    assert setup.execution.numerical_controls["tail_gate"] == {
        "tolerance": 0.1,
        "batches": 16,
        "max_reference_state_years": 2_000_000,
        "max_comparison_state_years": 500_000,
    }
    # And every work allowance per simulated year at its default (NUM-BUDGETS).
    assert {
        key: value
        for key, value in setup.execution.numerical_controls.items()
        if key != "tail_gate"
    } == {
        "max_vectorized_events_per_year": 50_000,
        "max_campaign_members_per_year": 50_000,
        "max_members_per_campaign": 2_000_000,
    }
    # At 4 planned years the reference arm's 3 states may double to 524,288
    # years each within 2,000,000 state-years, its followers with it; a
    # comparison arm's 500,000 allow 131,072 for 3 states and 262,144 for 1.
    workload = setup_workload(setup)["by_arm"]
    assert {
        arm: (row["conditional_states"], row["max_years_per_state"])
        for arm, row in workload.items()
    } == {
        "reference": (3, 524_288),
        "public_forecast::lower_pace_shape": (3, 524_288),
        "public_forecast::higher_pace_shape": (3, 524_288),
        "public_forecast::lower_pace_cap": (3, 524_288),
        "public_forecast::no_progress": (1, 262_144),
        "tail_shape_lower": (3, 131_072),
        "tail_shape_higher": (3, 131_072),
    }
    # The scope shows the template; each trajectory's laws carry its own scales.
    assert {row.use_population for row in setup.scope.deployments} == {template_text}
    assert {row.scenario_id for row in setup.scope.populations} == {
        "output_reliance_legal",
        fine,
    }
    assert setup.limitations[0].startswith(
        "Hypothetical entities, stipulated rather than observed: Synthetic lab (lab `lab`)"
    )
    assert setup.limitations[1] == (
        "Respondent spread is uncalibrated: stated parameter quantiles are read at face value (k = 1)."
    )
    assert any(
        "users: 100 as of, 200 on the median path" in row for row in setup.limitations
    )
    assert any("and M = 3 is the ceiling on growth" in row for row in setup.limitations)
    # Intake recomputes the quadrature, median progress included, from the
    # frozen forecast, and every node's progress ratio from that quadrature.
    import catmodel.setup.public_study as service

    frozen = folder / "public-frontier.json"
    payload = json.loads(frozen.read_text())
    payload["median_progress"] *= 2
    tampered = PublicFrontierQuadrature.model_validate(payload)
    frozen.unlink()
    frozen.write_text(tampered.model_dump_json())
    with pytest.raises(ValueError, match="trajectory quadrature differs"):
        imported("tampered")
    monkeypatch.setattr(
        service, "public_frontier_quadrature", lambda *_args, **_kwargs: tampered
    )
    with pytest.raises(ValueError, match="path-linked exposure differs"):
        imported("tampered-rule")


def test_parent_questions_name_the_hypothetical_entities():
    from catmodel.elicitation.contextual_hierarchy import ContextualParentPrompt

    instrument = hierarchy_instrument(("output_reliance_legal",))
    node, respondent = instrument.nodes[0], instrument.respondents[0]
    parent = next(row for row in instrument.parents if node.node_id in row.children)
    question = ContextualParentPrompt(instrument, parent, respondent, 0.5)
    assert question.render().startswith(HYPOTHETICAL)
