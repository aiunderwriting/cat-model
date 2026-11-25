"""Exposure-scale variants: separately collected studies measured against the embedded rule."""

import json
import re
from typing import cast

import pytest

from catmodel.collection.contextual_collection import run_study_collection
from catmodel.contracts.contextual_risk import (
    ContextualRiskInstrument,
    PathLinkedExposure,
    render_exposure,
)
from catmodel.contracts.public_frontier import PublicFrontierQuadrature
from catmodel.llm.transport import ProviderResponse
from catmodel.results.execution import run_setup
from catmodel.setup.contextual_inputs import build_contextual_instrument
from catmodel.setup.public_frontier_comparisons import (
    CONTEXT_VARIANT_FAMILIES,
    EXPOSURE_CONTROL,
)
from catmodel.setup.public_study import (
    import_public_frontier_study,
    write_public_frontier_study,
)
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
    path_linked_payload,
)
from test_support.public_frontier import collect_study_nodes
from tests.test_hypothetical_frontier_lab import _run_prompt, _shared_context
from tests.test_public_frontier_study import (
    _collect,
    _study,
)

USERS = re.compile(r"About ([0-9,]+) synthetic users")
BLIND_WORDS = (
    "0.5",
    "half",
    "halved",
    "double",
    "doubled",
    "factor",
    "times",
    "variant",
    "variants",
    "arm",
    "exposure_scale",
    "sensitivity",
    "scaled",
    "headline",
    "control",
)
QUANTILES = ("p5", "p25", "p50", "p75", "p95")
exposure_variant_state_id = CONTEXT_VARIANT_FAMILIES["exposure_scale"].state_id


class _ScaledProvider:
    """A synthetic annual rate proportional to the question's rendered users."""

    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    async def complete(self, *, user_prompt, **_kwargs):
        found = USERS.search(user_prompt)
        assert found is not None
        rate = float(found.group(1).replace(",", "")) / 100
        answer = contextual_answer()
        assert answer.rate is not None
        answer = answer.model_copy(
            update={
                "rate": answer.rate.model_copy(update=dict.fromkeys(QUANTILES, rate))
            }
        )
        return ProviderResponse(
            text=fixture_response(user_prompt, answer),
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=5,
            output_tokens=10,
        )


def _template(sample_count=2):
    payload = path_linked_payload(contextual_instrument().model_dump(mode="json"))
    payload["respondents"] = [
        {
            "respondent_id": f"fixture-model:{sample}",
            "model_id": "fixture-model",
            "sample_index": sample,
        }
        for sample in range(sample_count)
    ]
    return ContextualRiskInstrument.model_validate(payload)


def _variant_study(
    tmp_path,
    bundle,
    variant_id,
    factor,
    *,
    nodes=3,
    no_progress=True,
    name=None,
    **edits,
):
    """A variant on the headline's forecast with one sample.

    Its no-progress node-00 is collected unless ``no_progress`` is unset;
    ``edits`` replace instrument fields.
    """
    name = name or f"{variant_id}-{nodes}"
    templates = {}
    for status in ("public", "not_yet_public"):
        payload = json.loads((tmp_path / f"{status}.json").read_text())
        payload["path_linked_exposure"]["variant"] = {
            "variant_id": variant_id,
            "factor": factor,
        }
        payload["respondents"] = payload["respondents"][:1]
        payload.update(edits)
        templates[status] = tmp_path / f"{name}-{status}.json"
        templates[status].write_text(
            ContextualRiskInstrument.model_validate(payload).model_dump_json()
        )
    folder = tmp_path / f"exposure-{name}"
    write_public_frontier_study(
        forecast_path=tmp_path / "forecast-input.json",
        public_instrument=templates["public"],
        not_yet_public_instrument=templates["not_yet_public"],
        assumptions_path=tmp_path / "assumptions.json",
        output=folder,
        nodes=nodes,
        seed=42,
    )
    study = folder / "public-study.json"
    if no_progress:
        run_study_collection(study=study, source_bundle=bundle, profile="test")
    else:
        plan = json.loads(study.read_text())
        draws = {row["draw_id"] for row in plan["contexts"]}
        collect_study_nodes(
            study, draws - {"node-00"}, source_bundle=bundle, profile="test"
        )
        assert not (folder / "collections" / "node-00-public").exists()
    return study


def _instrument(study, draw_id):
    plan = json.loads(study.read_text())
    row = next(
        row
        for row in plan["contexts"]
        if row["draw_id"] == draw_id and row["source_model_status"] == "public"
    )
    return ContextualRiskInstrument.model_validate_json(
        (study.parent / row["instrument"]).read_text()
    )


def _rendered(study, draw_id):
    return _instrument(study, draw_id).deployment_binding.use_population


def _headline(tmp_path, monkeypatch, *, samples=2):
    folder, bundle, _ = _study(
        tmp_path,
        monkeypatch,
        nodes=7,
        template=_template(samples),
        providers=[_ScaledProvider()],
    )
    _collect(folder, bundle)
    return folder, bundle


def _importer(tmp_path, headline):
    def imported(name, studies, **options):
        return import_public_frontier_study(
            source=headline,
            destination=tmp_path / name,
            setup_id="variants",
            years_per_state=8,
            # The import's 4:1 ratio of reference to comparison state-years.
            max_reference_state_years=22 * 4096,
            max_comparison_state_years=22 * 1024,
            seed=5,
            variant_studies=studies,
            code_commit="fixture",
            **options,
        )

    return imported


def test_exposure_variants_run_against_the_embedded_rule_control(tmp_path, monkeypatch):
    folder, bundle = _headline(tmp_path, monkeypatch)
    half = _variant_study(tmp_path, bundle, "half", 0.5)
    double = _variant_study(tmp_path, bundle, "double", 2.0)
    headline = folder / "public-study.json"
    # A variant node is the headline node of the same ID, its scales multiplied
    # before rounding.
    assert _rendered(half, "node-1-2") == "About 100 synthetic users"
    assert _rendered(headline, "node-1-2") == "About 200 synthetic users"
    assert _rendered(double, "node-1-2") == "About 400 synthetic users"
    # Variants are blind: the question shows the scaled numbers as the study's own
    # and adds no factor or arm language to the headline question at that node.
    variant = _instrument(half, "node-1-2")
    assert variant.path_linked_exposure is not None
    assert variant.path_linked_exposure.variant is not None
    assert variant.path_linked_exposure.variant.factor == 0.5
    variant_prompt = _run_prompt(variant)
    headline_prompt = _run_prompt(_instrument(headline, "node-1-2"))
    assert "About 100 synthetic users" in variant_prompt
    for word in BLIND_WORDS:
        pattern = rf"(?<![\w.]){re.escape(word)}(?!\w)"
        assert len(re.findall(pattern, variant_prompt, re.IGNORECASE)) == len(
            re.findall(pattern, headline_prompt, re.IGNORECASE)
        ), word

    imported = _importer(tmp_path, headline)

    package = imported("package", (half, double))
    arms = {row.alternative_id: row for row in package.alternatives}
    variants = ("exposure_scale::half", "exposure_scale::double")
    assert {EXPOSURE_CONTROL, *variants} <= set(package.execution.required_arms)
    assert {arms[arm].matched_control_id for arm in variants} == {EXPOSURE_CONTROL}
    assert {
        row.status
        for row in package.execution.comparison_requirements
        if row.family_id == "exposure_scale"
    } == {"required"}
    # Three embedded-rule nodes and the no-progress node by one sample, at zero
    # reference weight; the no-progress node enters no arm.
    weights = {state.state_id: state.weight for state in package.joint_states}
    variant_states = {
        key for key in weights if key.startswith(exposure_variant_state_id("half", ""))
    }
    assert len(variant_states) == 4 and {weights[key] for key in variant_states} == {
        0.0
    }
    no_progress = {key for key in variant_states if "-node-00:" in key}
    assert len(no_progress) == 1
    half_weights = arms["exposure_scale::half"].state_weights
    control_weights = arms[EXPOSURE_CONTROL].state_weights
    assert half_weights is not None and control_weights is not None
    assert {key for key, value in half_weights.items() if value} == (
        variant_states - no_progress
    )
    # The control is the headline on the embedded rule's three nodes.
    quadrature = PublicFrontierQuadrature.model_validate_json(
        (folder / "public-frontier.json").read_text()
    )
    embedded = {
        node.draw_id: weight
        for node, weight in zip(
            quadrature.nodes, quadrature.base.embedded_weights, strict=True
        )
    }
    for key, value in control_weights.items():
        draw = key.split(":", 1)[0]
        assert value == pytest.approx(embedded.get(draw, 0.0) / 2)
    assert any("Exposure-scale variants" in row for row in package.limitations)

    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    summary = report["sensitivity_summary"]
    # Variant states cost only the variants, and each variant's effect pairs it with its
    # control on their common years.
    gate = report["tail_gate"]["by_arm"]
    assert gate["reference"]["states"] == sum(
        state.weight > 0 for state in package.joint_states
    )
    # The control reweights headline states the reference arm simulates.
    assert gate[EXPOSURE_CONTROL]["follows_arm"] == "reference"
    for arm in variants:
        # Its three weighted states and, at weight zero, its no-progress state.
        assert gate[arm]["states"] == 4
        effect = cast(dict, report["comparisons"])[arm]["welfare"]["var::0.99"]
        assert effect["years_per_state"] == min(
            gate[arm]["years_per_state"], gate[EXPOSURE_CONTROL]["years_per_state"]
        )
    ranked = {row["arm"]: row for row in summary["tornado"]["rows"]}
    assert EXPOSURE_CONTROL not in ranked
    assert {ranked[arm]["control_arm"] for arm in variants} == {EXPOSURE_CONTROL}
    # The synthetic rate is proportional to the rendered users, which each variant
    # rounds to two significant figures after scaling.
    assert ranked["exposure_scale::half"]["relative_change"] == pytest.approx(
        -0.5, abs=0.02
    )
    assert ranked["exposure_scale::double"]["relative_change"] == pytest.approx(
        1.0, abs=0.04
    )
    control_eal = ranked["exposure_scale::half"]["control_eal"]
    quadrature = summary["trajectory_quadrature"]
    assert quadrature["status"] == "available"
    diagnostic = quadrature["by_arm"]["reference"]
    assert diagnostic["embedded_rule_expected_annual_loss"] == pytest.approx(
        control_eal, rel=1e-9
    )
    family = next(
        row
        for row in report["coverage"]["sensitivity_families"]
        if row["family_id"] == "exposure_scale"
    )
    assert family["status"] == "complete"

    without = imported("without", ())
    # Adding variants leaves every headline state and every other arm unchanged.
    headline_states = {state.state_id: state for state in without.joint_states}
    assert {
        state.state_id: state
        for state in package.joint_states
        if state.state_id in headline_states
    } == headline_states
    # Two variants of three nodes and the no-progress node at one sample each.
    assert len(package.joint_states) == len(headline_states) + 8
    for arm in without.alternatives:
        other = arms[arm.alternative_id]
        assert package.overrides.of(other.alternative_id) == (
            without.overrides.of(arm.alternative_id)
        )
        if arm.state_weights is not None:
            assert other.state_weights is not None
            assert {
                key: value
                for key, value in other.state_weights.items()
                if key in headline_states
            } == arm.state_weights
    assert [
        row
        for row in package.execution.comparison_requirements
        if row.family_id != "exposure_scale"
    ] == [
        row
        for row in without.execution.comparison_requirements
        if row.family_id != "exposure_scale"
    ]
    assert {
        row.reason
        for row in without.execution.comparison_requirements
        if row.family_id == "exposure_scale"
    } == {
        "No exposure-scale variant studies were imported; collect each variant as a public study on the embedded rule's nodes and the no-progress node and import it with --variant-study."
    }
    with pytest.raises(ValueError, match="require the exposure_scale comparison"):
        imported("unselected", (half,), comparison_families=("tail_shape",))
    with pytest.raises(ValueError, match="imported twice"):
        imported("twice", (half, half))
    wrong = _variant_study(tmp_path, bundle, "half", 0.5, nodes=7)
    with pytest.raises(ValueError, match="embedded rule's node count"):
        imported("wrong", (wrong,))
    with pytest.raises(ValueError, match="cannot carry an exposure variant"):
        import_public_frontier_study(
            source=wrong,
            destination=tmp_path / "variant-as-headline",
            setup_id="variants",
            code_commit="fixture",
        )


def test_import_refuses_an_exposure_variant_without_its_no_progress_node(
    tmp_path, monkeypatch
):
    folder, bundle = _headline(tmp_path, monkeypatch, samples=1)
    variant = _variant_study(tmp_path, bundle, "half", 0.5, no_progress=False)
    destination = tmp_path / "no-zero-node"
    with pytest.raises(ValueError, match="must collect its no-progress node node-00"):
        _importer(tmp_path, folder / "public-study.json")(
            "no-zero-node", (variant,), comparison_families=("exposure_scale",)
        )
    assert not destination.exists()


def test_exposure_variant_intake_guards(tmp_path, monkeypatch):
    folder, bundle = _headline(tmp_path, monkeypatch)
    imported = _importer(tmp_path, folder / "public-study.json")
    off_catalog = _variant_study(tmp_path, bundle, "half", 0.6, name="off-catalog")
    with pytest.raises(ValueError, match="must use the catalog scale factor"):
        imported("off-catalog", (off_catalog,))
    other_controls = _variant_study(
        tmp_path,
        bundle,
        "half",
        0.5,
        name="other-controls",
        fixed_controls="Synthetic controls that differ from the headline's.",
    )
    with pytest.raises(ValueError, match="only in its scale factor"):
        imported("other-controls", (other_controls,))
    # A variant node must carry the embedded rule's weight for its node.
    import catmodel.setup.public_study as service

    half = _variant_study(tmp_path, bundle, "half", 0.5)
    base = service._base_weights

    def shifted(quadrature):
        weights = base(quadrature)
        if quadrature.interior_nodes == 3:  # the variant's rule, not the headline's
            moved = weights["node-1-4"] / 2
            weights["node-1-4"] -= moved
            weights["node-1-2"] += moved
        return weights

    monkeypatch.setattr(service, "_base_weights", shifted)
    with pytest.raises(ValueError, match="embedded-rule node at its weight"):
        imported("shifted", (half,))


def test_variant_factors_multiply_path_linked_scales_before_rounding():
    exposure = PathLinkedExposure.model_validate(
        {
            "max_growth_multiple": 3,
            "scales": [
                {
                    "scale_id": "tokens",
                    "unit": "tokens per day",
                    "as_of": 4e13,
                    "study_year_median": 1e14,
                },
            ],
            "progress_ratio": 2.144,
        }
    )
    # About 162 trillion before rounding: rounding before the half factor
    # would give 80 trillion.
    text = "{tokens} tokens"
    assert render_exposure(text, exposure) == "160 trillion tokens"
    for factor, expected in (
        (0.5, "81 trillion tokens"),
        (2.0, "320 trillion tokens"),
    ):
        variant = PathLinkedExposure.model_validate(
            {
                **exposure.model_dump(mode="json"),
                "variant": {"variant_id": "variant", "factor": factor},
            }
        )
        assert render_exposure(text, variant) == expected


def test_build_instrument_records_a_catalog_exposure_variant(tmp_path):
    bundle = write_source_bundle(tmp_path / "source")
    context = _shared_context()
    # The scenario's public chain reads weekly users.
    for block in context["source_statuses"].values():
        block["deployment_binding"]["use_population"] = (
            "About {users} users and {consumer_weekly_users} weekly users"
        )
    linked = {
        **{key: value for key, value in context.items() if key != "fixed_exposure"},
        "path_linked_exposure": {
            "max_growth_multiple": 3,
            "scales": [
                {
                    "scale_id": scale_id,
                    "unit": scale_id,
                    "as_of": 100,
                    "study_year_median": 200,
                }
                for scale_id in ("users", "consumer_weekly_users")
            ],
        },
    }

    def build(payload, variant):
        return build_contextual_instrument(
            payload,
            source_bundle=bundle,
            respondents=contextual_instrument().respondents,
            scenario_ids=("output_reliance_legal",),
            source_status="public",
            exposure_variant=variant,
        )

    exposure = build(linked, "double").path_linked_exposure
    assert exposure is not None and exposure.variant is not None
    assert (exposure.variant.variant_id, exposure.variant.factor) == ("double", 2.0)
    plain = build(linked, None).path_linked_exposure
    assert plain is not None and plain.variant is None
    with pytest.raises(ValueError, match="exposure variant must be one of"):
        build(linked, "triple")
    with pytest.raises(ValueError, match="requires path-linked exposure"):
        build(context, "half")
