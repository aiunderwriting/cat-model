"""Lab losses are capped at a stipulated frontier-lab value (SEV-LAB-CAP)."""

import copy
import json
import math
import re

import pytest

from catmodel.collection.contextual_collection import run_contextual_collection
from catmodel.config import ModelProfile
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskInstrument,
)
from catmodel.contracts.incident import IncidentLaw
from catmodel.contracts.native_severity import native_severity_sampler
from catmodel.elicitation.contextual_risk import ContextualPrompt
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import ProviderResponse
from catmodel.results.execution import run_setup
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.public_study import (
    import_public_frontier_study,
    write_public_frontier_study,
)
from catmodel.setup.sensitivity_catalog import (
    FRONTIER_LAB_VALUE_SENSITIVITY_USD,
    FRONTIER_LAB_VALUE_USD,
    SENSITIVITY_FAMILIES,
)
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_artifact,
    contextual_instrument,
    fixture_response,
)
from test_support.public_frontier import synthetic_forecast
from test_support.results import available_metric, reduced_arm

LAB_LOSS_QUANTILES = {
    "p5": 1e10,
    "p25": 2e10,
    "p50": 4e10,
    "p75": 8e10,
    "p95": 2e11,
    "p99": 6e11,
    "p999": 2e12,
}


def test_one_stipulated_value_and_its_alternatives():
    assert FRONTIER_LAB_VALUE_USD == 1e12
    assert FRONTIER_LAB_VALUE_SENSITIVITY_USD == (3e11, 3e12)
    (family,) = (row for row in SENSITIVITY_FAMILIES if row.family_id == "lab_loss_cap")
    assert family.assumption_ids == ("SEV-LAB-CAP", "ACC-LAB-ANNUAL-BUSINESS-LOSS")
    assert "USD 300 billion and USD 3 trillion instead of USD 1 trillion" in (
        family.description
    )


def _instrument(
    status: str = "hypothetical", endpoint: float | None = None
) -> ContextualRiskInstrument:
    payload = contextual_instrument().model_dump(mode="json")
    payload["lab"]["status"] = status
    payload["nodes"][0]["lab_loss_severity_cap"] = endpoint
    return ContextualRiskInstrument.model_validate(payload)


def _named_lab(payload: dict) -> dict:
    payload["lab"].update({"entity_id": "lab_b", "status": "actual"})
    payload["lab"]["related_entity_ids"] = ["model"]
    payload["model"]["related_entity_ids"] = []
    return payload


def _answer(business_loss: dict, share: float = 0) -> ContextualRiskAnswer:
    payload = contextual_answer().model_dump(mode="json")
    payload["lab_annual_business_loss"] = business_loss
    payload["lab_compensation"] = {"status": "available", "share": share}
    return ContextualRiskAnswer.model_validate(payload)


def _lab_law(instrument, answer, **options):
    law = compile_contextual_node(
        instrument,
        instrument.nodes[0],
        {0: answer, 2: answer},
        1,
        options=CompileOptions(**options),
    )
    assert isinstance(law, IncidentLaw)
    # The same cap bounds each episode and, separately, the annual business loss.
    (episode,) = law.segments[0].kernel.campaign_lab_losses
    (lab,) = law.annual_lab_losses
    assert episode.amount == 0
    return law, lab


@pytest.mark.parametrize("status", ["actual", "hypothetical"])
@pytest.mark.parametrize(
    ("endpoint", "options", "expected"),
    [
        (None, {}, FRONTIER_LAB_VALUE_USD),
        (1e14, {}, FRONTIER_LAB_VALUE_USD),
        (5e11, {}, 5e11),
        (None, {"frontier_lab_value": 3e11}, 3e11),
        (1e14, {"frontier_lab_value": None}, 1e14),
        (None, {"frontier_lab_value": None}, None),
    ],
)
def test_every_lab_is_capped_at_the_lower_of_value_and_endpoint(
    status, endpoint, options, expected
):
    instrument = _instrument(status, endpoint)
    stochastic = _answer(
        {"status": "available", "positive_probability": 1, "loss": LAB_LOSS_QUANTILES}
    )
    law, lab = _lab_law(instrument, stochastic, **options)
    assert law.segments[0].kernel.campaign_lab_losses[0].episode_cap == expected
    assert lab.severity is not None
    top = native_severity_sampler(lab.severity).quantile(1)
    assert top == (math.inf if expected is None else pytest.approx(expected))
    diagnostic = law.conditioning_context["lab_annual_business_loss_tail_diagnostic"]
    assert diagnostic["governed_severity_cap"] == expected
    assert diagnostic["class_endpoint_cap"] == endpoint
    value = options.get("frontier_lab_value", FRONTIER_LAB_VALUE_USD)
    assert diagnostic["frontier_lab_value_cap"] == value
    assert law.conditioning_context["lab_loss_frontier_lab_value"] == value
    fixed = _answer({"status": "available", "positive_probability": 1, "amount": 2e14})
    _, lab = _lab_law(instrument, fixed, **options)
    assert lab.amount == (2e14 if expected is None else expected)


def test_named_lab_needs_no_value_of_its_own():
    instrument = ContextualRiskInstrument.model_validate(
        _named_lab(contextual_instrument().model_dump(mode="json"))
    )
    fixed = _answer({"status": "available", "positive_probability": 1, "amount": 2e12})
    _, lab = _lab_law(instrument, fixed)
    assert lab.amount == FRONTIER_LAB_VALUE_USD


@pytest.mark.parametrize("endpoint", [None, 1e14])
@pytest.mark.parametrize("status", ["actual", "hypothetical"])
def test_prompt_states_a_stipulated_value_without_showing_it(status, endpoint):
    instrument = _instrument(status, endpoint)
    rendered = ContextualPrompt(
        instrument, instrument.nodes[0], instrument.respondents[0], 0
    ).render()
    # Quantiles are uncapped; the question neither shows nor describes the cap.
    assert "with no cap" in rendered
    assert "not shown" not in rendered and "frontier-lab value" not in rendered
    assert not re.search(
        r"(?<![\d.])1(?:000000000000(?:\.0)?|e\+?12)(?![\d.])", rendered
    )
    assert "trillion" not in rendered
    assert "enterprise value" not in rendered


def test_setup_limitations_disclose_the_value_and_unexecuted_alternatives():
    instrument = contextual_instrument()
    setup = compile_contextual_setup(
        instrument,
        contextual_artifact(instrument),
        setup_id="lab-cap",
        years_per_state=1,
    )
    (disclosure,) = (text for text in setup.limitations if "SEV-LAB-CAP," in text)
    assert "capped at USD 1 trillion" in disclosure
    assert "a year's total lab loss can exceed the value" in disclosure
    assert (
        "alternatives (USD 300 billion, USD 3 trillion and no lab cap) are not executed"
        in disclosure
    )


class _Provider:
    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    async def complete(self, **kwargs):
        answer = _answer(
            {"status": "available", "positive_probability": 1, "amount": 1.5e12},
            share=0.5,
        )
        return ProviderResponse(
            text=fixture_response(kwargs["user_prompt"], answer),
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=5,
            output_tokens=10,
        )


def test_public_study_import_and_run_cap_lab_loss_quantiles(tmp_path, monkeypatch):
    """Offline forecast, collection, import and run for named lab ``lab_b``.

    Every fixture answer has a Poisson count with mean one per year per source
    status, a USD 20 provider-borne fine, half of its USD 100 third-party
    component as compensation and a fixed annual business loss of USD 1.5
    trillion, which the stipulated value caps at USD 1 trillion. The selected
    lab_loss_cap alternatives recompile the same captures at USD 300 billion, USD 3
    trillion and without the cap.
    """
    bundle = write_source_bundle(tmp_path / "source")
    payload = _named_lab(contextual_instrument().model_dump(mode="json"))
    payload["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        bundle
    ).manifest_sha256
    templates = {}
    for status in ("public", "not_yet_public"):
        candidate = copy.deepcopy(payload)
        candidate["context_id"] = status
        candidate["model"]["entity_id"] = f"model_{status}"
        candidate["lab"]["related_entity_ids"] = [f"model_{status}"]
        candidate["deployment"]["entity_id"] = f"deployment_{status}"
        candidate["deployment_binding"]["model_id"] = f"model_{status}"
        candidate["deployment_binding"]["deployment_id"] = f"deployment_{status}"
        candidate["nodes"][0]["population"]["deployment_ids"] = [f"deployment_{status}"]
        candidate["capability_access"] = "public" if status == "public" else "internal"
        path = tmp_path / f"{status}.json"
        path.write_text(
            ContextualRiskInstrument.model_validate(candidate).model_dump_json()
        )
        templates[status] = path
    forecast = tmp_path / "forecast.json"
    forecast.write_text(synthetic_forecast().model_dump_json())
    assumptions = tmp_path / "assumptions.json"
    assumptions.write_text(
        json.dumps(
            {
                "exposure_rationale": "Disjoint synthetic incidents by source status.",
                "residual_independence_rationale": "Independent given the public path.",
                "parameter_coupling_rationale": "Matched synthetic respondent identity.",
            }
        )
    )
    study = tmp_path / "study"
    write_public_frontier_study(
        forecast_path=forecast,
        public_instrument=templates["public"],
        not_yet_public_instrument=templates["not_yet_public"],
        assumptions_path=assumptions,
        output=study,
        nodes=3,
        seed=42,
    )
    profile = ModelProfile(
        models=(
            ModelSpec(
                provider="openai",
                model_id="fixture-model",
                reasoning_effort="low",
                max_output_tokens=100_000,
            ),
        )
    )
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: [_Provider()]
    )
    plan = json.loads((study / "public-study.json").read_text())
    for row in plan["contexts"]:
        run_contextual_collection(
            instrument=ContextualRiskInstrument.model_validate_json(
                (study / row["instrument"]).read_text()
            ),
            source_bundle=bundle,
            run_directory=study / row["capture"],
            profile="test",
        )
    package = import_public_frontier_study(
        source=study / "public-study.json",
        destination=tmp_path / "package",
        setup_id="lab-cap",
        years_per_state=64,
        # 4,096 years for each of the 4 states, whatever the arm.
        max_reference_state_years=4 * 4096,
        max_comparison_state_years=4 * 4096,
        seed=3,
        comparison_families=("lab_loss_cap", "lab_loss_provider_only"),
        code_commit="fixture",
    )
    assert any(
        "capped at USD 1 trillion" in line and "are executed" in line
        for line in package.limitations
    )
    amounts = {
        lab.amount
        for state in package.joint_states
        for law in state.laws
        for lab in law.annual_lab_losses
    }
    assert amounts == {FRONTIER_LAB_VALUE_USD}
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    # Without the no-progress arm no arm weights or simulates the no-progress
    # node, and the trajectory-quadrature exhibit needs no draw of it.
    assert report["sensitivity_summary"]["trajectory_quadrature"]["status"] == (
        "available"
    )
    # Two source statuses at a constant mean of one episode a year each. Each
    # episode costs the lab its USD 20 fine, a provider-borne component, and
    # USD 50 of compensation, capped once per episode; a year with an episode
    # adds USD 1.5 trillion of business loss once, capped on its own. Every
    # alternative's mean is exact, including where the cap binds.
    occurrence = -math.expm1(-1)
    for arm, annual in (
        ("reference", 1e12),
        ("lab_loss_cap_lower", 3e11),
        ("lab_loss_cap_higher", 1.5e12),
        ("lab_loss_cap_removed", 1.5e12),
    ):
        metric = available_metric(
            reduced_arm(report["arms"][arm])["metrics"]["lab_loss::lab_b"]
        )
        mean = metric["expected_annual_loss"]
        assert mean == pytest.approx(2 * (70 + occurrence * annual)), arm
        # The EAL is the exact analytic mean over the parameter rank.
        assert metric["eal_estimator"] == "native_analytic_mean", arm
        assert metric["eal_precision"]["precision_basis"] == (
            "deterministic_integration"
        ), arm
        quantiles = [metric["annual_loss_median"]] + [
            row["var"]
            for row in metric["tail_metrics"].values()
            if row["var"] is not None
        ]
        assert max(quantiles) > 0, arm
        # Every annual quantile is whole episodes plus at most one business
        # loss per source status.
        assert all(
            any(
                (value - years * annual) / 70
                == pytest.approx(round((value - years * annual) / 70), abs=1e-3)
                and value >= years * annual
                for years in (0, 1, 2)
            )
            for value in quantiles
        ), arm
    # The provider-borne lower bound keeps only the USD 20 fine per episode:
    # neither the compensation nor the business loss.
    bound = available_metric(
        reduced_arm(report["arms"]["lab_loss_provider_only"])["metrics"][
            "lab_loss::lab_b"
        ]
    )
    assert bound["expected_annual_loss"] == pytest.approx(2 * 20)
