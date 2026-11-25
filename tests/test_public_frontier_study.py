"""Offline forecast-to-collection-to-native-loss integration."""

import inspect
import json
import math
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from typing import Any, cast

import numpy as np
import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.collection.contextual_collection import run_study_collection
from catmodel.config import ModelProfile
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.public_frontier import YEAR_SECONDS, PublicFrontierQuadrature
from catmodel.elicitation.question_text import question_preamble
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.forecast.public_frontier import fit_public_trend
from catmodel.forecast.public_frontier_quadrature import (
    public_frontier_quadrature,
)
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import ProviderResponse
from catmodel.results.execution import (
    load_run_capture,
    run_setup,
    setup_workload,
)
from catmodel.results.results_file import load_results
from catmodel.setup import contextual_node
from catmodel.setup.public_frontier import public_state_node
from catmodel.setup.public_study import (
    _same_quadrature,
    import_public_frontier_study,
    write_public_frontier_study,
)
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from catmodel.setup.sensitivity_catalog import PUBLIC_STUDY_UNAVAILABLE_CHALLENGES
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_moments import analytic_linear_eals
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
    governed_contextual_answer,
    governed_contextual_instrument,
    with_access_bases,
)
from test_support.count_law_oracle import CountLawOracle
from test_support.public_frontier import synthetic_forecast
from test_support.results import (
    available_metric,
    present,
    reduced_arm,
    reported_exhibit,
)
from test_support.version_markers import unstated_markers

START = datetime(2027, 1, 1, tzinfo=UTC)
END = datetime(2028, 1, 1, tzinfo=UTC)


def test_quadrature_replay_accepts_roundoff_but_rejects_material_changes():
    original = public_frontier_quadrature(
        synthetic_forecast(), period_start=START, period_end=END, nodes=3
    )
    for difference, accepted in ((1e-13, True), (1e-3, False)):
        payload = original.model_dump(mode="json")
        weights = payload["base"]["weights"]
        weights[1] += difference
        weights[2] -= difference
        replay = PublicFrontierQuadrature.model_validate(payload)
        assert _same_quadrature(original, replay) is accepted
    for factor, accepted in ((1 + 1e-10, True), (10, False)):
        payload = original.model_dump(mode="json")
        payload["posterior"]["rate_variance"] *= factor
        replay = PublicFrontierQuadrature.model_validate(payload)
        assert _same_quadrature(original, replay) is accepted
    # A node's trend rate is a root found to a tolerance, so a replay may
    # differ from it by roundoff: every lab of the node then moves with it.
    anchors = np.asarray([100.0, 90.0])
    elapsed = np.asarray(
        [(at - original.as_of).total_seconds() / YEAR_SECONDS for at in original.times]
    )
    for factor, accepted in ((1 + 1e-11, True), (1 + 1e-4, False)):
        payload = original.model_dump(mode="json")
        node = payload["nodes"][2]
        node["baseline_rate"] *= factor
        rates = node["baseline_rate"] * np.asarray(node["pace_multipliers"])
        node["rates"] = rates.tolist()
        node["values"] = (anchors + elapsed[:, None] * rates).tolist()
        node["progress"] *= factor
        replay = PublicFrontierQuadrature.model_validate(payload)
        assert _same_quadrature(original, replay) is accepted
    # A recomputed rule is the same rule; one under another pace cap is not,
    # and neither is one whose lower-cap alternative carries another cap.
    recomputed = public_frontier_quadrature(
        synthetic_forecast(), period_start=START, period_end=END, nodes=3
    )
    assert _same_quadrature(original, recomputed)
    other = public_frontier_quadrature(
        synthetic_forecast().model_copy(update={"pace_cap": 3.0}),
        period_start=START,
        period_end=END,
        nodes=3,
    )
    assert not _same_quadrature(original, other)
    payload = original.model_dump(mode="json")
    lower = next(
        law for law in payload["alternatives"] if law["label"] == "lower_pace_cap"
    )
    assert lower["pace_cap"] == 2
    lower["pace_cap"] = 2.5
    replay = PublicFrontierQuadrature.model_validate(payload)
    assert not _same_quadrature(original, replay)


def _estimated_discrepancy(quadrature):
    """The quadrature with an estimated trend discrepancy and its interval."""
    payload = quadrature.model_dump(mode="json")
    payload["posterior"].update(
        trend_discrepancy_basis="pooled_reml_estimate",
        trend_discrepancy_interval=[1.5, 3.0],
    )
    return PublicFrontierQuadrature.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["posterior_sd", "posterior_interval", "law_sd"],
)
def test_quadrature_replay_accepts_fitted_discrepancy_roundoff(field):
    original = _estimated_discrepancy(
        public_frontier_quadrature(
            synthetic_forecast(), period_start=START, period_end=END, nodes=3
        )
    )
    for factor, accepted in ((1 + 1e-9, True), (1 - 1e-9, True), (1 + 1e-3, False)):
        payload = original.model_dump(mode="json")
        posterior = payload["posterior"]
        if field == "posterior_sd":
            posterior["trend_discrepancy_sd"] *= factor
        elif field == "posterior_interval":
            posterior["trend_discrepancy_interval"] = [
                value * factor for value in posterior["trend_discrepancy_interval"]
            ]
        else:
            for law in (payload["base"], *payload["alternatives"]):
                law["trend_discrepancy_sd"] *= factor
        replay = PublicFrontierQuadrature.model_validate(payload)
        assert _same_quadrature(original, replay) is accepted


class _Provider:
    provider_name = "openai"
    reasoning_effort = "low"

    def __init__(self, model_id="fixture-model", answer=None):
        self.model_id = model_id
        self.answer = answer or (lambda _call: contextual_answer())
        self.calls = 0
        self.prompts = []

    async def complete(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["user_prompt"])
        return ProviderResponse(
            text=fixture_response(kwargs["user_prompt"], self.answer(self.calls)),
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=5,
            output_tokens=10,
        )


def _study(
    tmp_path,
    monkeypatch,
    *,
    nodes=3,
    period=None,
    template=None,
    providers=None,
    forecast_input=None,
):
    bundle = write_source_bundle(tmp_path / "source")
    digest = capture_source_bundle_snapshot(bundle).manifest_sha256
    payload = (template or contextual_instrument()).model_dump(mode="json")
    payload["source_bundle_manifest_sha256"] = digest
    # The two templates represent disjoint source populations; usage location is
    # deliberately internal for the PUBLIC population as well.
    payload["deployment"]["definition"] = "Synthetic internal use location"
    templates = {}
    for status in ("public", "not_yet_public"):
        candidate = json.loads(json.dumps(payload))
        access = "public" if status == "public" else "internal"
        with_access_bases(candidate["nodes"], access)
        candidate["context_id"] = status
        candidate["model"]["entity_id"] = f"model_{status}"
        candidate["lab"]["related_entity_ids"] = [f"model_{status}"]
        candidate["deployment"]["entity_id"] = f"deployment_{status}"
        candidate["deployment_binding"]["model_id"] = f"model_{status}"
        candidate["deployment_binding"]["deployment_id"] = f"deployment_{status}"
        for node in candidate["nodes"]:
            node["population"]["deployment_ids"] = [f"deployment_{status}"]
        candidate["capability_access"] = access
        candidate["period"].update(period or {})
        instrument = ContextualRiskInstrument.model_validate(candidate)
        path = tmp_path / f"{status}.json"
        path.write_text(instrument.model_dump_json())
        templates[status] = path
    forecast = tmp_path / "forecast-input.json"
    forecast.write_text((forecast_input or synthetic_forecast()).model_dump_json())
    assumptions = tmp_path / "assumptions.json"
    assumptions.write_text(
        json.dumps(
            {
                "exposure_rationale": "Disjoint synthetic incidents by source model status at origin.",
                "residual_independence_rationale": "Conditional independence given the common public path, respondent and rank.",
                "parameter_coupling_rationale": "Matched respondent identity across the synthetic conditional judgments.",
            }
        )
    )
    folder = tmp_path / "study"
    summary = write_public_frontier_study(
        forecast_path=forecast,
        public_instrument=templates["public"],
        not_yet_public_instrument=templates["not_yet_public"],
        assumptions_path=assumptions,
        output=folder,
        nodes=nodes,
        seed=42,
    )
    providers = providers or [_Provider()]
    profile = ModelProfile(
        models=tuple(
            ModelSpec(
                provider="openai",
                model_id=provider.model_id,
                reasoning_effort="low",
                max_output_tokens=100_000,
            )
            for provider in providers
        )
    )
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: providers
    )
    return folder, bundle, summary


def _collect(folder, bundle):
    artifacts = run_study_collection(
        study=folder / "public-study.json",
        source_bundle=bundle,
        profile="test",
    )
    plan = json.loads((folder / "public-study.json").read_text())
    assert artifacts is not None and len(artifacts) == len(plan["contexts"])
    for row in plan["contexts"]:
        frozen = json.loads(
            (folder / row["capture"] / "collection-plan.json").read_text()
        )
        # Caps schedule requests; they are no part of a run's identity.
        assert "concurrency_by_model" not in frozen["execution"]["settings"]
    return plan


SPREAD_TEST_RATE = Quantiles(p5=1, p25=4.5, p50=12.6, p75=35.8, p95=160)


def _spread_test_provider():
    """A non-degenerate count law so spread and tail-width alternatives are observable."""
    answer = contextual_answer().model_copy(update={"rate": SPREAD_TEST_RATE})
    return _Provider(answer=lambda _call: answer)


def test_public_study_replays_quadrature_nodes_with_status_specific_lab_relations(
    tmp_path, monkeypatch
):
    folder, bundle, summary = _study(tmp_path, monkeypatch)
    assert summary["trajectory_nodes"] == 4
    assert summary["conditional_collections"] == 8
    plan = _collect(folder, bundle)
    paired = [
        ContextualRiskInstrument.model_validate_json(
            (folder / row["instrument"]).read_text()
        ).public_trajectory
        for row in plan["contexts"][:2]
    ]
    assert paired[0] is not None and paired[1] is not None
    assert paired[0].interpolation == "linear"
    assert paired[0].target_lab == "public_frontier"
    assert plan["target_lab"] == "public_frontier"
    assert paired[0].values == paired[1].values
    assert paired[0].source_model_status == "public"
    assert paired[1].source_model_status == "not_yet_public"
    package = import_public_frontier_study(
        source=folder / "public-study.json",
        destination=tmp_path / "package",
        setup_id="synthetic-public",
        years_per_state=4,
        # 4,096 years for each of the 4 states, whatever the arm.
        max_reference_state_years=4 * 4096,
        max_comparison_state_years=4 * 4096,
        seed=3,
        code_commit="fixture",
    )
    quadrature = PublicFrontierQuadrature.model_validate_json(
        (folder / "public-frontier.json").read_text()
    )
    assert sum(state.weight for state in package.joint_states) == pytest.approx(1)
    assert len(package.joint_states) == 4
    # The no-progress node carries the no-progress paths and no base weight.
    assert sorted(state.weight for state in package.joint_states)[0] == 0
    assert {state.weight for state in package.joint_states} == set(
        quadrature.base.weights
    )
    assert "evidence/captured/public-frontier.json" in package.files
    evidence = package.evidence.records[0]
    assert evidence.origin == "fit"
    acquisition = evidence.acquisition
    assert acquisition is not None
    assert acquisition["as_of_frontier"] == {
        "lab_a": 100,
        "lab_b": 90,
    }
    record: Any = acquisition["trajectory_quadrature"]
    assert record["interior_nodes"] == 3 and record["embedded_interior_nodes"] == 1
    assert set(record["state_draws"]) == {
        state.state_id for state in package.joint_states
    }
    # Each state's entity paths follow its node's frontier: the labs' maximum.
    conditions = {}
    for row in plan["contexts"]:
        condition = ContextualRiskInstrument.model_validate_json(
            (folder / row["instrument"]).read_text()
        ).public_trajectory
        assert condition is not None
        conditions[condition.draw_id] = condition
    for state in package.joint_states:
        condition = conditions[public_state_node(state.state_id)]
        for path in state.entity_paths:
            assert len(path.knots) == len(condition.target_path()[0])
            assert path.knots[0].value == pytest.approx(max(condition.values[0]))
            assert path.knots[-1].value == pytest.approx(max(condition.values[-1]))
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")


def test_public_study_collections_select_trajectory_nodes_and_refuse_unknown_ones(
    tmp_path, monkeypatch
):
    from catmodel.setup.public_study import public_study_collections

    folder, _, _ = _study(tmp_path, monkeypatch)
    plan = json.loads((folder / "public-study.json").read_text())
    every = public_study_collections(folder / "public-study.json")
    assert every == [
        (folder / row["instrument"], folder / row["capture"])
        for row in plan["contexts"]
    ]
    first = plan["contexts"][0]["draw_id"]
    assert public_study_collections(folder / "public-study.json", [first]) == [
        (folder / row["instrument"], folder / row["capture"])
        for row in plan["contexts"]
        if row["draw_id"] == first
    ]
    with pytest.raises(ValueError, match="no trajectory node node-99"):
        public_study_collections(folder / "public-study.json", ["node-99"])


def test_public_study_rejects_missing_status_or_tampered_path(tmp_path, monkeypatch):
    folder, bundle, _ = _study(tmp_path, monkeypatch)
    plan = _collect(folder, bundle)
    plan["contexts"].pop()
    (folder / "public-study.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="every node and both"):
        import_public_frontier_study(
            source=folder / "public-study.json",
            destination=tmp_path / "missing",
            setup_id="missing",
            code_commit="fixture",
        )
    assert not (tmp_path / "missing").exists()


def test_public_study_import_rechecks_recaptured_instrument_cutoff(
    tmp_path, monkeypatch
):
    folder, bundle, _ = _study(tmp_path, monkeypatch)
    plan = json.loads((folder / "public-study.json").read_text())
    for row in plan["contexts"]:
        path = folder / row["instrument"]
        payload = json.loads(path.read_text())
        payload["period"]["forecast_as_of"] = "2026-09-30T00:00:00Z"
        instrument = ContextualRiskInstrument.model_validate(payload)
        path.write_text(instrument.model_dump_json())
    _collect(folder, bundle)
    with pytest.raises(ValueError, match="exceeds the instrument information cutoff"):
        import_public_frontier_study(
            source=folder / "public-study.json",
            destination=tmp_path / "earlier-cutoff",
            setup_id="earlier-cutoff",
            code_commit="fixture",
        )
    assert not (tmp_path / "earlier-cutoff").exists()


def test_a_variant_study_defaults_to_the_default_rules_embedded_nodes(
    tmp_path, monkeypatch
):
    """Without a node count, a headline study plans the default seven interior
    nodes and a variant study, whose templates carry a context variant, the
    embedded three-node rule's, as import requires of the variants of such a
    headline."""
    _study(tmp_path, monkeypatch)

    def plan(name, levels):
        templates = {}
        for status, level in zip(("public", "not_yet_public"), levels, strict=True):
            payload = json.loads((tmp_path / f"{status}.json").read_text())
            payload["safeguard_level"] = level
            templates[status] = tmp_path / f"{name}-{status}.json"
            templates[status].write_text(
                ContextualRiskInstrument.model_validate(payload).model_dump_json()
            )
        return write_public_frontier_study(
            forecast_path=tmp_path / "forecast-input.json",
            public_instrument=templates["public"],
            not_yet_public_instrument=templates["not_yet_public"],
            assumptions_path=tmp_path / "assumptions.json",
            output=tmp_path / name,
        )

    # Each count adds the no-progress node.
    assert plan("headline", (None, None))["trajectory_nodes"] == 8
    assert plan("weaker", ("weaker", "weaker"))["trajectory_nodes"] == 4
    with pytest.raises(ValueError, match="must carry the same context variant"):
        plan("mixed", ("weaker", None))
    assert not (tmp_path / "mixed").exists()


def test_the_public_study_command_plans_the_named_node_count(tmp_path, monkeypatch):
    """``--nodes`` plans a rule other than the default seven, such as the
    hypothetical-lab study's 15, and refuses a count the rule cannot take
    before it writes anything."""
    from tests.test_hypothetical_frontier_lab import _shared_context

    # The forecast with its releases, the assumptions and the source bundle.
    _study(tmp_path, monkeypatch)
    context = tmp_path / "context.json"
    context.write_text(json.dumps(_shared_context()))

    def plan(name, nodes):
        return CliRunner().invoke(
            cli,
            [
                "--profile",
                "test",
                "capability-forecast",
                "public-study",
                "--source-bundle",
                str(tmp_path / "source"),
                "--forecast",
                str(tmp_path / "forecast-input.json"),
                "--context",
                str(context),
                "--scenario",
                "output_reliance_legal",
                "--assumptions",
                str(tmp_path / "assumptions.json"),
                "--nodes",
                nodes,
                "--output",
                str(tmp_path / name),
            ],
        )

    result = plan("three", "3")
    assert result.exit_code == 0, result.output
    # The count adds the no-progress node.
    assert json.loads(result.output)["trajectory_nodes"] == 4
    # A forecast that states its releases is the study's forecast as it is.
    assert json.loads((tmp_path / "three" / "forecast.json").read_bytes()) == (
        json.loads((tmp_path / "forecast-input.json").read_bytes())
    )
    planned = sorted(tmp_path.iterdir())
    refused = plan("even", "4")
    assert refused.exit_code != 0
    assert "odd number of at least 3 nodes" in refused.output
    # Neither the study nor anything staged for it is left behind.
    assert sorted(tmp_path.iterdir()) == planned


def test_public_study_cli_available_and_never_accepts_unknown_variant():
    result = CliRunner().invoke(cli, ["capability-forecast", "public-study", "--help"])
    assert result.exit_code == 0
    for option in ("--source-bundle", "--forecast", "--context", "--variant"):
        assert option in result.output
    result = CliRunner().invoke(cli, ["setup", "import", "--help"])
    assert result.exit_code == 0
    for option in ("--study", "--comparison-family", "--variant-study"):
        assert option in result.output


def test_a_public_study_reads_count_laws_at_face_value_beside_its_fixed_alternatives(
    tmp_path, monkeypatch
):
    folder, _, _ = _study(tmp_path, monkeypatch, providers=[_spread_test_provider()])
    _collect(folder, tmp_path / "source")
    package = import_public_frontier_study(
        source=folder / "public-study.json",
        destination=tmp_path / "package",
        setup_id="face-value-public",
        years_per_state=4,
        max_reference_state_years=16,
        max_comparison_state_years=16,
        seed=3,
        comparison_families=(
            "epistemic_spread_calibration",
            "epistemic_tail_width",
            "common_frequency_bias",
        ),
        code_commit="fixture",
    )

    # Every respondent states SPREAD_TEST_RATE, so every count law is that law
    # read at the arm's spread and scaled by its factor; CountLawOracle
    # integrates its mean from the methodology's formulas.
    def reads(law, rank_spread, count_factor, tail_scale=1.0):
        rank = law.parameter_rank
        assert rank is not None
        assert (rank.rank_spread, rank.tail_form) == (
            rank_spread,
            REFERENCE_SPREAD_TAIL_FORM,
        )
        assert rank.epistemic_tail_scale == tail_scale
        assert law.conditioning_context["count_factor"] == pytest.approx(
            count_factor, rel=1e-15
        )
        for name in ("p5", "p25", "p50", "p75", "p95"):
            assert getattr(rank.count, name) == pytest.approx(
                count_factor * getattr(SPREAD_TEST_RATE, name), rel=1e-12
            )
        oracle = CountLawOracle(
            SPREAD_TEST_RATE,
            tail_scale=tail_scale,
            spread=rank_spread,
            tail_form=REFERENCE_SPREAD_TAIL_FORM,
        )
        assert law.reference_annual_rate == pytest.approx(
            count_factor * oracle.mean(), rel=1e-10
        )

    # The reference is the stated law at face value: no spread and no factor.
    reference_laws = [law for state in package.joint_states for law in state.laws]
    assert reference_laws
    for law in reference_laws:
        reads(law, 1.0, 1.0)
        assert law.parameter_rank is not None
        assert law.parameter_rank.count == SPREAD_TEST_RATE
    # Each arm's spread, count factor and tail-width factor.
    expected = {
        "epistemic_spread_widened": (1.5, 1.0, 1.0),
        "epistemic_spread_narrowed": (0.75, 1.0, 1.0),
        "epistemic_tail_narrowed": (1.0, 1.0, 0.75),
        "epistemic_tail_widened": (1.0, 1.0, 1.5),
        "common_frequency_bias_lower": (1.0, 0.5, 1.0),
        "common_frequency_bias_higher": (1.0, 2.0, 1.0),
    }
    alternatives = {row.alternative_id: row for row in package.alternatives}
    assert set(alternatives) == set(expected)
    assert set(package.execution.required_arms) == {"reference", *expected}
    for arm, (rank_spread, count_factor, tail_scale) in expected.items():
        alternative = alternatives[arm]
        assert (
            f"Count laws use k = {rank_spread:g} with "
            f"tail_form = {REFERENCE_SPREAD_TAIL_FORM}." in alternative.description
        )
        if count_factor == 1:
            assert "Count factor = " not in alternative.description
        else:
            assert f"Count factor = {count_factor:g}." in alternative.description
        overrides = package.overrides.of(alternative.alternative_id)
        assert overrides
        for override in overrides:
            reads(override.law, rank_spread, count_factor, tail_scale)
    # No arm is marked mandatory: each holds fixed only the common pair.
    arm_requirements = [
        row
        for row in package.execution.comparison_requirements
        if row.arm_id in expected
    ]
    assert len(arm_requirements) == len(expected)
    assert all(len(row.held_fixed) == 2 for row in arm_requirements)

    # No setup carries a respondent fit: no evidence record, no stored fit
    # and no limitation that names one.
    assert not any("calibration" in row.evidence_id for row in package.evidence.records)
    assert not any("calibration" in name for name in package.files)
    assert package.limitations[0].startswith("Hypothetical entities")
    assert package.limitations[1] == (
        "Respondent spread is uncalibrated: stated parameter quantiles are read "
        "at face value (k = 1)."
    )
    assert not any(
        "exploratory" in limitation or "respondent-spread fit" in limitation
        for limitation in package.limitations
    )

    report = run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="fixture",
    )
    summary = report["sensitivity_summary"]
    estimate = reported_exhibit(summary["frequency_estimate"])
    assert estimate["views"]["welfare"]["by_arm"]["final"][
        "expected_annual_loss"
    ] == pytest.approx(
        present(summary["tornado"]["eal"])["expected_annual_loss"], rel=1e-10
    )
    # Results report what the count laws use and disclose it.
    scenarios = report["capability_scenarios"]
    assert scenarios["respondent_spread"] == {
        "k": 1.0,
        "basis": "face_value",
        "level_bias": 1.0,
    }
    assert (
        "EALs are conditional on respondent spread k = 1 (face value)."
        in scenarios["disclosures"]
    )
    assert not any("fitted" in text for text in scenarios["disclosures"])


def test_public_study_import_takes_no_respondent_fit(tmp_path):
    """``setup import`` refuses ``--spread-calibration`` as an unknown option,
    before it reads or writes anything, and the import function takes no
    such argument."""
    fit = tmp_path / "respondent-spread-fit.json"
    fit.write_text("{}")
    study = tmp_path / "public-study.json"
    study.write_text("{}")
    result = CliRunner().invoke(
        cli,
        [
            "setup",
            "import",
            "--study",
            str(study),
            "--setup-id",
            "study",
            "--output",
            str(tmp_path / "setup"),
            "--spread-calibration",
            str(fit),
        ],
    )
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert "--spread-calibration" in result.output
    assert not (tmp_path / "setup").exists()
    assert (
        "spread_calibration"
        not in inspect.signature(import_public_frontier_study).parameters
    )


def test_fit_public_retains_prior_uncertainty_without_window_records():
    posterior = fit_public_trend(
        synthetic_forecast().model_copy(
            update={"trend_window_start": synthetic_forecast().as_of}
        )
    )
    assert (posterior.rate_location, posterior.rate_variance) == (10, 4)
    assert all(lab.likelihood_release_ids == () for lab in posterior.labs.values())
    assert posterior.labs["lab_a"].as_of_value == 100
    assert posterior.labs["lab_a"].as_of_release_id == "a-2"


def test_public_study_rejects_changed_posterior_before_loading_captures(
    tmp_path, monkeypatch
):
    folder, _, _ = _study(tmp_path, monkeypatch)
    posterior_path = folder / "trend-posterior.json"
    payload = json.loads(posterior_path.read_text())
    payload["rate_location"] += 1
    posterior_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="retained posterior differs"):
        import_public_frontier_study(
            source=folder / "public-study.json",
            destination=tmp_path / "tampered",
            setup_id="tampered",
            code_commit="fixture",
        )
    assert not (tmp_path / "tampered").exists()


def test_public_study_rejects_changed_rule_weights_before_loading_captures(
    tmp_path, monkeypatch
):
    # The weights reach no question, so replay alone checks them.
    folder, _, _ = _study(tmp_path, monkeypatch)
    path = folder / "public-frontier.json"
    payload = json.loads(path.read_text())
    payload["base"]["weights"][1] += 1e-3
    payload["base"]["weights"][2] -= 1e-3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="trajectory quadrature differs"):
        import_public_frontier_study(
            source=folder / "public-study.json",
            destination=tmp_path / "reweighted",
            setup_id="reweighted",
            code_commit="fixture",
        )
    assert not (tmp_path / "reweighted").exists()


def test_a_study_whose_files_do_not_state_their_schema_is_refused(
    tmp_path, monkeypatch
):
    folder, bundle, _ = _study(tmp_path, monkeypatch)
    quadrature = folder / "public-frontier.json"
    written = quadrature.read_bytes()
    for payload in unstated_markers(json.loads(written)).values():
        quadrature.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="quadrature schema version"):
            import_public_frontier_study(
                source=folder / "public-study.json",
                destination=tmp_path / "unstated",
                setup_id="unstated",
                code_commit="fixture",
            )
    quadrature.write_bytes(written)
    plan = json.loads((folder / "public-study.json").read_text())
    instrument = folder / plan["contexts"][0]["instrument"]
    written = instrument.read_bytes()
    for payload in unstated_markers(json.loads(written)).values():
        instrument.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="contextual instrument schema"):
            _collect(folder, bundle)
    instrument.write_bytes(written)
    _collect(folder, bundle)


def test_a_refined_study_replays_the_collections_of_the_rule_it_contains(
    tmp_path, monkeypatch
):
    first = _Provider()
    coarse, bundle, _ = _study(tmp_path, monkeypatch, nodes=7, providers=[first])
    _collect(coarse, bundle)
    fine = tmp_path / "refined"
    write_public_frontier_study(
        forecast_path=tmp_path / "forecast-input.json",
        public_instrument=tmp_path / "public.json",
        not_yet_public_instrument=tmp_path / "not_yet_public.json",
        assumptions_path=tmp_path / "assumptions.json",
        output=fine,
        nodes=15,
        seed=42,
    )
    assert (
        run_study_collection(
            study=fine / "public-study.json",
            source_bundle=bundle,
            profile="test",
            dry_run=True,
        )
        is None
    )
    rows = {
        (row["draw_id"], row["source_model_status"]): row
        for row in json.loads((fine / "public-study.json").read_text())["contexts"]
    }
    shared = [
        (
            coarse / row["capture"],
            fine / rows[row["draw_id"], row["source_model_status"]]["capture"],
        )
        for row in json.loads((coarse / "public-study.json").read_text())["contexts"]
    ]
    assert len(rows) == 2 * len(shared) == 32
    # Every shared context freezes the same instrument, request keys and prompts.
    for old, new in shared:
        for name in ("contextual-instrument.json", "collection-plan.json"):
            assert (old / name).read_bytes() == (new / name).read_bytes()
        requests = sorted(new.glob("calls/*/request.json"))
        assert requests
        for path in requests:
            assert (old / path.relative_to(new)).read_bytes() == path.read_bytes()
        for name in ("calls", "acquisition", "parser-protocols"):
            shutil.copytree(old / name, new / name, dirs_exist_ok=True)
    second = _Provider()
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: [second]
    )
    _collect(fine, bundle)
    # The refined collection asks only the new nodes' questions, as many as
    # the whole coarse study asked.
    # A question names no context; each context's questions begin with its
    # own preamble, whose trajectory node and source status differ.
    preambles = {}
    for collection in (fine / "collections").iterdir():
        instrument = ContextualRiskInstrument.model_validate_json(
            (collection / "contextual-instrument.json").read_text()
        )
        preambles[collection.name] = question_preamble(instrument, 0) + "\n\n"
    asked = set()
    for prompt in second.prompts:
        (context,) = (
            name for name, preamble in preambles.items() if prompt.startswith(preamble)
        )
        asked.add(context)
    reused = {new.name for _, new in shared}
    assert asked == {path.name for path in (fine / "collections").iterdir()} - reused
    assert second.calls == first.calls
    for old, new in shared:
        assert (old / "contextual-risk.json").read_bytes() == (
            new / "contextual-risk.json"
        ).read_bytes()
    package = import_public_frontier_study(
        source=fine / "public-study.json",
        destination=tmp_path / "refined-package",
        setup_id="refined",
        years_per_state=4,
        code_commit="fixture",
    )
    assert {public_state_node(state.state_id) for state in package.joint_states} == {
        draw_id for draw_id, _ in rows
    }
    acquisition = package.evidence.records[0].acquisition
    assert acquisition is not None
    record: Any = acquisition["trajectory_quadrature"]
    assert record["interior_nodes"] == 15


def test_public_study_refuses_a_node_that_departs_from_its_scenario(
    tmp_path, monkeypatch
):
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios

    folder, _, _ = _study(tmp_path, monkeypatch)
    template = tmp_path / "public.json"
    scenario_id = json.loads(template.read_text())["nodes"][0]["population"][
        "scenario_id"
    ]
    # The template's hand-authored node now names a repository scenario whose
    # authored population it does not reproduce.
    other_scenario = next(
        row
        for row in load_scenarios(scenarios_path())
        if row.id == "service_outage_degradation"
    )
    monkeypatch.setattr(
        "catmodel.setup.public_study.load_scenarios",
        lambda *_args, **_kwargs: [
            other_scenario.model_copy(update={"id": scenario_id})
        ],
    )
    with pytest.raises(
        ValueError,
        match=f"node for scenario {scenario_id!r} differs from its authored "
        "scenario definition in population",
    ):
        write_public_frontier_study(
            forecast_path=tmp_path / "forecast-input.json",
            public_instrument=template,
            not_yet_public_instrument=tmp_path / "not_yet_public.json",
            assumptions_path=tmp_path / "assumptions.json",
            output=tmp_path / "unresolved-study",
            nodes=3,
            seed=42,
        )
    assert folder.exists()
    assert not (tmp_path / "unresolved-study").exists()


def test_public_study_rejects_different_lab_loss_scopes_before_collection(
    tmp_path, monkeypatch
):
    _study(tmp_path, monkeypatch)
    path = tmp_path / "not_yet_public.json"
    payload = json.loads(path.read_text())
    payload["lab_annual_business_loss_definition"] = (
        "Different synthetic business-loss boundary"
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="same lab-loss scope"):
        write_public_frontier_study(
            forecast_path=tmp_path / "forecast-input.json",
            public_instrument=tmp_path / "public.json",
            not_yet_public_instrument=path,
            assumptions_path=tmp_path / "assumptions.json",
            output=tmp_path / "different-scope",
            nodes=3,
            seed=42,
        )
    assert not (tmp_path / "different-scope").exists()


def test_study_target_cannot_be_relabelled_after_collection(tmp_path, monkeypatch):
    folder, bundle, _ = _study(tmp_path, monkeypatch)
    plan = _collect(folder, bundle)
    plan["target_lab"] = "lab_a"
    (folder / "public-study.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="target lab must be public_frontier"):
        import_public_frontier_study(
            source=folder / "public-study.json",
            destination=tmp_path / "changed-target",
            setup_id="changed-target",
            code_commit="fixture",
        )
    assert not (tmp_path / "changed-target").exists()


def test_public_study_takes_its_forecast_period_from_the_templates(
    tmp_path, monkeypatch
):
    period = {
        "policy_period_start_utc": "2027-04-01T00:00:00Z",
        "policy_period_end_utc": "2028-04-01T00:00:00Z",
    }
    folder, _, _ = _study(tmp_path, monkeypatch, period=period)
    ensemble = json.loads((folder / "public-frontier.json").read_text())
    assert ensemble["times"][0] == "2027-04-01T00:00:00Z"
    assert ensemble["times"][-1] == "2028-04-01T00:00:00Z"
    assert len(ensemble["times"]) == 13
    values = np.asarray(ensemble["nodes"][1]["values"])
    assert np.all(values >= [100, 90])
    comparison = json.loads((folder / "forecast-comparison.json").read_text())
    assert list(comparison["variants"]["baseline"]["checkpoints"]) == [
        "2027-04-01T00:00:00+00:00",
        "2027-10-01T00:00:00+00:00",
        "2028-04-01T00:00:00+00:00",
    ]


def test_public_study_rejects_templates_with_different_periods(tmp_path, monkeypatch):
    _study(tmp_path, monkeypatch)
    path = tmp_path / "not_yet_public.json"
    payload = json.loads(path.read_text())
    payload["period"]["policy_period_start_utc"] = "2027-02-01T00:00:00Z"
    payload["period"]["policy_period_end_utc"] = "2028-02-01T00:00:00Z"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="share one policy period"):
        write_public_frontier_study(
            forecast_path=tmp_path / "forecast-input.json",
            public_instrument=tmp_path / "public.json",
            not_yet_public_instrument=path,
            assumptions_path=tmp_path / "assumptions.json",
            output=tmp_path / "different-period",
            nodes=3,
            seed=42,
        )
    assert not (tmp_path / "different-period").exists()


def test_default_comparisons_run_end_to_end_into_the_sensitivity_summary(
    tmp_path, monkeypatch
):
    models = ("model-a", "model-b", "model-c")
    template = governed_contextual_instrument(
        ("compromise_critical_infrastructure",), models
    )
    # One call per provider and collection, in node order with both source
    # statuses per node, so the synthetic rate level rises with the node's rate.
    providers = [
        _Provider(
            model,
            partial(
                lambda scale, call: governed_contextual_answer(
                    template.nodes[0], scale=scale * (1 + (call - 1) // 2)
                ),
                index + 1,
            ),
        )
        for index, model in enumerate(models)
    ]
    folder, bundle, _ = _study(
        tmp_path,
        monkeypatch,
        template=template,
        providers=providers,
        forecast_input=synthetic_forecast().model_copy(update={"pace_shape": 5.0}),
    )
    _collect(folder, bundle)
    package = import_public_frontier_study(
        source=folder / "public-study.json",
        destination=tmp_path / "package",
        setup_id="synthetic-public",
        years_per_state=16,
        # The import's 4:1 ratio of reference to comparison state-years.
        max_reference_state_years=12 * 4096,
        max_comparison_state_years=12 * 1024,
        seed=3,
        code_commit="fixture",
    )
    plan = package.execution
    quadrature = PublicFrontierQuadrature.model_validate_json(
        (folder / "public-frontier.json").read_text()
    )
    forecast_arms = tuple(
        f"public_forecast::{law.label}" for law in quadrature.alternatives
    )
    assert forecast_arms == (
        "public_forecast::lower_pace_shape",
        "public_forecast::higher_pace_shape",
        "public_forecast::lower_pace_cap",
        "public_forecast::no_progress",
    )
    assert plan.required_arms == (
        "reference",
        *forecast_arms,
        *(f"model_pooling_leave_one_out::{model}" for model in models),
        "tail_shape_lower",
        "tail_shape_higher",
        "uncapped_severity",
    )
    assert {
        row.reason
        for row in plan.comparison_requirements
        if row.family_id == "exposure_scale"
    } == {
        "The study's deployment scales are authored text, not path-linked values, so no variant can rescale them."
    }
    for law in quadrature.alternatives:
        arm = next(
            row
            for row in package.alternatives
            if row.alternative_id == f"public_forecast::{law.label}"
        )
        assert arm.selection_provenance["pace_shape"] == law.pace_shape
        assert arm.selection_provenance["pace_cap"] == law.pace_cap
        assert arm.state_weights is not None
    unavailable = {
        row.reason.split(":")[0]
        for row in plan.comparison_requirements
        if row.family_id == "public_forecast"
        and row.status == "unavailable"
        and row.reason is not None
    }
    # A cap above the forecast's has paths beyond every collected node.
    assert unavailable == {"trend_discrepancy", "higher_pace_cap"}
    # Each arm simulates only the states it weights: the reference arm leaves
    # out the no-progress node, which only the no-progress arm weights.
    weights = {state.state_id: state.weight for state in package.joint_states}
    for arm in plan.required_arms:
        alternative = next(
            (row for row in package.alternatives if row.alternative_id == arm), None
        )
        arm_weights = (
            weights
            if alternative is None or alternative.state_weights is None
            else alternative.state_weights
        )
        assert set(plan.required_states(arm, tuple(weights))) == {
            state for state, weight in arm_weights.items() if weight > 0
        }
    zero_rate = len(plan.required_states("public_forecast::no_progress", ()))
    positive = sum(weight > 0 for weight in weights.values())
    workload = setup_workload(package)
    # Reweighting arms reuse the reference draws; three arms change laws and
    # the no-progress arm simulates the no-progress node's states.
    assert workload["reference_work_multiple"] == pytest.approx(
        4 + zero_rate / positive
    )
    assert workload["by_arm"]["public_forecast::lower_pace_shape"]["follows_arm"] == (
        "reference"
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    summary = report["sensitivity_summary"]
    assert summary["tornado"]["view"] == "welfare"
    # The comparison arms that change laws stop at a quarter of the reference
    # arm's state-years, and their effects pair the arms' common years.
    gate = report["tail_gate"]["by_arm"]
    shorter = [
        arm
        for arm in ("tail_shape_lower", "tail_shape_higher", "uncapped_severity")
        if gate[arm]["years_per_state"] < gate["reference"]["years_per_state"]
    ]
    assert shorter
    comparisons = cast(dict, report["comparisons"])
    for arm in shorter:
        assert (
            comparisons[arm]["welfare"]["var::0.99"]["years_per_state"]
            == gate[arm]["years_per_state"]
        )
    assert (
        gate["public_forecast::lower_pace_shape"]["years_per_state"]
        == gate["reference"]["years_per_state"]
    )
    # A reweighting arm's EAL is the captured states' analytic means under
    # its own weights, so its effect is their reweighted difference and has
    # no simulation error: no half-width, and nothing for the gate to
    # withhold. That holds for gross and the lab's loss, whose means carry
    # the certified error of the annual business loss term, as for welfare,
    # whose mean is closed-form.
    draws, capture = load_run_capture(tmp_path / "run" / "capture")
    reweighting = [
        arm
        for arm in plan.required_arms
        if arm.startswith(("public_forecast::", "model_pooling_leave_one_out::"))
    ]
    assert len(reweighting) == 7

    def reweighted_eal(arm, view):
        return math.fsum(
            weight * draws[arm][state].views[view].native_mean.value
            for state, weight in capture.state_weights_by_arm[arm].items()
            if weight > 0
        )

    reference_metrics = reduced_arm(report["arms"]["reference"])["metrics"]
    for view in ("welfare", "gross", "lab_loss::lab"):
        certified = view != "welfare"
        precision = available_metric(reference_metrics[view])["eal_precision"]
        assert (precision["integration_error_bound"] > 0) is certified
        for arm in reweighting:
            effect = comparisons[arm][view]["expected_annual_loss"]
            assert "half_width" not in effect and "status" not in effect, (arm, view)
            assert effect["absolute_difference"] == pytest.approx(
                reweighted_eal(arm, view) - reweighted_eal("reference", view),
                rel=1e-9,
                abs=1e-9 * effect["reference"],
            )
    # Unexecutable register challenges stay disclosed and keep coverage partial.
    assert report["status"] == "partial"
    assert report["coverage"]["comparisons"]["status"] == "partial"
    assert set(PUBLIC_STUDY_UNAVAILABLE_CHALLENGES) <= set(
        summary["tornado"]["not_assessed_families"]
    )
    assert {
        row["family_id"]
        for row in report["coverage"]["sensitivity_families"]
        if row["kind"] == "register_challenge" and row["status"] == "unavailable"
    } == set(PUBLIC_STUDY_UNAVAILABLE_CHALLENGES)
    ranked = {row["arm"]: row for row in summary["tornado"]["rows"]}
    assert set(ranked) == set(plan.required_arms) - {"reference"}
    assert ranked["uncapped_severity"]["absolute_difference"] > 0
    assert ranked["tail_shape_higher"]["absolute_difference"] > 0
    assert ranked["tail_shape_lower"]["absolute_difference"] < 0
    # Synthetic risk rises with the trajectory's rate, so holding the frontier
    # lowers the EAL, the wider pace alternative raises it and the lower pace
    # cap lowers it.
    assert ranked["public_forecast::no_progress"]["absolute_difference"] < 0
    assert ranked["public_forecast::lower_pace_shape"]["absolute_difference"] > 0
    assert ranked["public_forecast::lower_pace_cap"]["absolute_difference"] < 0
    diagnostic = summary["trajectory_quadrature"]
    assert diagnostic["status"] == "available"
    assert set(diagnostic["by_arm"]) == {"reference", *forecast_arms}
    reference = diagnostic["by_arm"]["reference"]
    assert reference["expected_annual_loss"] == pytest.approx(
        present(summary["tornado"]["eal"])["expected_annual_loss"], rel=1e-9
    )
    assert diagnostic["by_arm"]["public_forecast::no_progress"]["indicator"] == "exact"
    # Three interior nodes leave a one-node embedded rule, which cannot reveal
    # the full rule's error for any spread law.
    assert {
        row["indicator"]
        for arm, row in diagnostic["by_arm"].items()
        if arm != "public_forecast::no_progress"
    } == {"uninformative"}
    assert diagnostic["by_arm"]["public_forecast::no_progress"][
        "absolute_difference"
    ] == pytest.approx(0)
    # Recompute the embedded-rule EAL from the frozen node weights and the
    # exact rank-integrated conditional EAL of every joint state.
    rules = {
        node.draw_id: (full, embedded)
        for node, full, embedded in zip(
            quadrature.nodes,
            quadrature.base.weights,
            quadrature.base.embedded_weights,
            strict=True,
        )
    }
    recomputed = math.fsum(
        state.weight
        * rules[public_state_node(state.state_id)][1]
        / rules[public_state_node(state.state_id)][0]
        * analytic_linear_eals(
            compile_incident_state(package, state.state_id), ("welfare",)
        ).values["welfare"]
        for state in package.joint_states
        if rules[public_state_node(state.state_id)][0] > 0
    )
    assert reference["embedded_rule_expected_annual_loss"] == pytest.approx(
        recomputed, rel=1e-9
    )
    assert summary["dependence_bracket"]["status"] == "available"
    assert (tmp_path / "run" / "tornado.svg").exists()
    assert reported_exhibit(summary["frequency_estimate"])["views"]
    # Reweighting the reference target leaves the setup-weight exhibits behind:
    # they are unavailable rather than reported at the setup's weights.
    positive = [state for state in package.joint_states if state.weight > 0]
    # Shift weight between trajectory nodes, whose states have different EALs,
    # so the reweighted EAL must move.
    nodes = sorted({public_state_node(state.state_id) for state in positive})
    shifted = {
        state.state_id: state.weight
        * (1 + nodes.index(public_state_node(state.state_id)) % 2)
        if state.weight > 0
        else 0.0
        for state in package.joint_states
    }
    total = math.fsum(shifted.values())
    shifted = {key: value / total for key, value in shifted.items()}
    assert len(positive) > 1
    by_arm = {
        arm: next(
            (
                dict(row.state_weights)
                for row in package.alternatives
                if row.alternative_id == arm and row.state_weights is not None
            ),
            shifted,
        )
        for arm in plan.required_arms
    }

    def cli_replay(label, weights):
        (tmp_path / f"{label}.json").write_text(json.dumps(weights))
        result = CliRunner().invoke(
            cli,
            [
                "--workers",
                "1",
                "results",
                "replay",
                "--setup",
                str(package.root),
                "--capture",
                str(tmp_path / "run" / "capture"),
                "--weights",
                str(tmp_path / f"{label}.json"),
                "--output",
                str(tmp_path / label),
            ],
        )
        return result, tmp_path / label / "results.json"

    # One vector cannot say what the reweighting arms' own weights become:
    # the command refuses it, naming them, and takes a vector for each arm.
    refused, unwritten = cli_replay("one-vector", shifted)
    assert refused.exit_code == 1
    assert all(arm in refused.output for arm in reweighting)
    assert not unwritten.parent.exists()
    accepted, written = cli_replay("reweighted", by_arm)
    assert accepted.exit_code == 0, accepted.output
    replay = load_results(written, diagnostics=True)
    reweighted = replay["sensitivity_summary"]
    assert present(reweighted["tornado"]["eal"])[
        "expected_annual_loss"
    ] != pytest.approx(present(summary["tornado"]["eal"])["expected_annual_loss"])
    for key in ("trajectory_quadrature", "frequency_estimate"):
        record = cast(Mapping[str, object], reweighted[key])
        assert record["status"] == "unavailable"
        assert "reweight" in str(record["reason"])


def test_worker_processes_compile_the_same_setup(tmp_path, monkeypatch):
    # The captures are read, and the draws of the reference and every
    # comparison arm compile, in two worker processes; the saved setup is the
    # serial import's, byte for byte.
    import catmodel.setup.public_study as service

    folder, bundle, _ = _study(tmp_path, monkeypatch)
    _collect(folder, bundle)
    concurrency = []
    pooled = service.WorkerPool.map

    def recorded(self, function, tasks, **kwargs):
        concurrency.append((function.__name__, kwargs.get("concurrency")))
        return pooled(self, function, tasks, **kwargs)

    # However little memory this machine has, the parallel import uses two.
    monkeypatch.setattr(service, "memory_bounded_workers", lambda workers, _: workers)
    monkeypatch.setattr(service.WorkerPool, "map", recorded)
    unit_laws: list[object] = []
    unit_law = contextual_node._unit_law

    def counted(query):
        unit_laws.append(query)
        return unit_law(query)

    monkeypatch.setattr(contextual_node, "_unit_law", counted)
    packages, compiled_here = {}, {}
    for workers in (1, 2):
        # Each import starts without a unit law worked out.
        monkeypatch.setattr(contextual_node, "_UNIT_LAWS", {})
        unit_laws.clear()
        packages[workers] = import_public_frontier_study(
            source=folder / "public-study.json",
            destination=tmp_path / f"package-{workers}",
            setup_id="synthetic-public",
            years_per_state=4,
            max_reference_state_years=4 * 4096,
            max_comparison_state_years=4 * 4096,
            seed=3,
            code_commit="fixture",
            workers=workers,
        )
        compiled_here[workers] = len(unit_laws)
    assert {("_read_collection", 2), ("_compile_draw", 2)} <= set(concurrency)
    # Checking a collection compiles its cases' unit laws. With workers the
    # checks run there and the unit laws come back with each collection, so
    # this process compiles none.
    assert compiled_here[1] > 0
    assert compiled_here[2] == 0
    serial, parallel = packages[1], packages[2]
    assert parallel.scientific_sha256 == serial.scientific_sha256
    assert parallel.manifest_sha256 == serial.manifest_sha256
    for name in ("setup-manifest.json", "setup.json", "execution-plan.json"):
        assert (tmp_path / "package-2" / name).read_bytes() == (
            tmp_path / "package-1" / name
        ).read_bytes()
