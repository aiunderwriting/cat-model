"""A node's lab loss is its provider-borne components and a compensation share
of the same draw's other components, capped once per episode without changing
any other view, plus its own annual business loss, which gross also counts,
capped separately by the same value (``ACC-LAB-BEARER``,
``ACC-LAB-ANNUAL-BUSINESS-LOSS``, ``SEV-LAB-CAP``).

The tests guard against a designation whose welfare books the lab's own
resource costs while its lab loss is zero, a fine the lab pays outside its lab
loss, a lab view below a fraud campaign's provider-borne slice, and a
compensation that exceeds the harm it compensates or ignores a severity
factor.
"""

from functools import cache

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskInstrument,
    LabCompensationAnswer,
)
from catmodel.contracts.incident import IncidentLaw
from catmodel.elicitation.contextual_risk import (
    ContextualPrompt,
    contextual_node_from_scenario,
    contextual_prompts,
    parse_contextual_answer,
)
from catmodel.resources import scenarios_path
from catmodel.scenarios.loader import load_scenarios
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_moments import (
    analytic_known_zero_views,
    capped_episode_lab_mean,
)
from test_support.contextual_risk import (
    contextual_answer,
    contextual_artifact,
    contextual_instrument,
)
from tests.test_contextual_study import _capture, _changed_artifact

LAB = "lab_loss::lab"
QUANTILES = ("p5", "p25", "p50", "p75", "p95", "p99", "p999")
NONE = {"status": "available", "positive_probability": 0}


@cache
def _scenarios():
    return {row.id: row for row in load_scenarios(scenarios_path())}


def _instrument(scenario_id: str) -> ContextualRiskInstrument:
    node = contextual_node_from_scenario(
        _scenarios()[scenario_id], "deployment", "public"
    )
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"] = [node.model_dump(mode="json")]
    return ContextualRiskInstrument.model_validate(payload)


def _answer(instrument, shares, *, total, business=None, compensation=0.0):
    node = instrument.nodes[0]
    assert set(shares) == {row.component_id for row in node.components}
    payload = contextual_answer().model_dump(mode="json")
    payload.update(
        total_loss=dict(zip(QUANTILES, total, strict=True)),
        # Exact-zero large-loss counts: no anchor, so the stated quantiles govern.
        large_loss_count=dict.fromkeys(QUANTILES[:5], 0),
        very_large_loss_count=dict.fromkeys(QUANTILES[:5], 0),
        lab_compensation={"status": "available", "share": compensation},
        lab_annual_business_loss=business or NONE,
        branches=[
            {
                "branch_id": "event",
                "probability": 1,
                "components": [
                    {"component_id": key, "status": "available", "share": value}
                    for key, value in shares.items()
                ],
            }
        ],
    )
    return ContextualRiskAnswer.model_validate(payload)


def _run(tmp_path, instrument, answer):
    def replace(case):
        case["answer"] = answer.model_dump(mode="json")

    _, _, report, capture = _capture(
        tmp_path,
        instrument,
        _changed_artifact(instrument, replace),
    )
    kernel = capture.kernel_at(instrument.context_id, 1, "reference")
    return report, {view: kernel.samples(view)[0] for view in ("gross", "welfare", LAB)}


def _fraud_shares(instrument):
    """Provider-borne components hold 0.3 of the total and the others 0.7."""
    components = [row.component_id for row in instrument.nodes[0].components]
    shares = dict.fromkeys(components, 0.0)
    shares.update(
        external_financial_loss=0.5,
        external_resource_loss=0.2,
        provider_resource_cost=0.15,
        provider_civil_fine=0.15,
    )
    return shares


FRAUD_TOTAL = (1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8)


def _fixture_answer(*, compensation=0.0, business=70.0) -> ContextualRiskAnswer:
    """The fixture's USD 100 third-party victim cell and USD 20 provider-borne
    fine, with a fixed business loss."""
    payload = contextual_answer().model_dump(mode="json")
    payload["lab_compensation"] = {"status": "available", "share": compensation}
    payload["lab_annual_business_loss"] = (
        {"status": "available", "positive_probability": 1, "amount": business}
        if business
        else NONE
    )
    return ContextualRiskAnswer.model_validate(payload)


def _fixture_law(answer, severity_factors=None, **options) -> IncidentLaw:
    instrument = contextual_instrument()
    law = compile_contextual_node(
        instrument,
        instrument.nodes[0],
        {0: answer, 2: answer},
        1,
        severity_factors=severity_factors,
        options=CompileOptions(**options),
    )
    assert isinstance(law, IncidentLaw)
    return law


def _annual(law: IncidentLaw) -> float:
    """The node's fixed annual business loss, zero without an annual law."""
    if not law.annual_lab_losses:
        return 0.0
    (annual,) = law.annual_lab_losses
    assert annual.amount is not None
    return annual.amount


def _episode_lab(law: IncidentLaw) -> float:
    """One episode's exact capped lab loss: every fixture amount is fixed."""
    kernel = law.segments[0].kernel
    (aggregate,) = kernel.campaign_lab_losses
    result = capped_episode_lab_mean(kernel, aggregate)
    assert result is not None
    mean, bound = result
    assert bound == 0
    return mean


def test_scenario_nodes_carry_authored_bearers():
    designation = _instrument("government_designation").nodes[0]
    assert {row.component_id: row.bearer for row in designation.components} == {
        "provider_resource_cost": "provider",
        "provider_transfer": "provider",
        "provider_civil_fine": "provider",
        "customer_migration_and_switching": "third_party",
        "deployer_migration_and_switching": "other",
    }
    campaign = _instrument("fraud_organizations").nodes[0]
    bearers = {row.component_id: row.bearer for row in campaign.components}
    assert bearers["provider_civil_fine"] == "provider"
    assert bearers["external_financial_loss"] == "third_party"
    assert bearers["deployer_operational_recovery"] == "other"


def test_designation_welfare_never_exceeds_the_lab_loss_of_the_same_draw(tmp_path):
    instrument = _instrument("government_designation")
    answer = _answer(
        instrument,
        {
            "provider_resource_cost": 0.3,
            "provider_transfer": 0.7,
            "provider_civil_fine": 0,
            "customer_migration_and_switching": 0,
            "deployer_migration_and_switching": 0,
        },
        total=(1e7, 5e7, 1e8, 3e8, 1e9, 3e9, 1e10),
    )
    report, samples = _run(tmp_path, instrument, answer)
    # Both cells fall on the lab, so its loss is the whole draw, and the
    # resource cell that welfare books is part of it.
    np.testing.assert_allclose(samples[LAB], samples["gross"], rtol=1e-12)
    np.testing.assert_allclose(samples["welfare"], 0.3 * samples["gross"], rtol=1e-12)
    assert np.all(samples[LAB] >= samples["welfare"])
    assert np.any(samples[LAB] > 0)
    assert report["arms"]["reference"]["metrics"][LAB]["status"] == "available"


def test_a_fine_the_lab_pays_is_lab_loss_without_compensation_or_business_loss(
    tmp_path,
):
    instrument = _instrument("enforcement_matter")
    answer = _answer(
        instrument,
        {
            "provider_civil_fine": 1,
            "provider_resource_cost": 0,
            "provider_noncivil_penalty": 0,
            "provider_ordinary_transfer": 0,
            "customer_migration_and_switching": 0,
            "deployer_migration_and_switching": 0,
        },
        total=(2e6, 5e6, 1e7, 3e7, 1e8, 3e8, 1e9),
    )
    _, samples = _run(tmp_path, instrument, answer)
    np.testing.assert_allclose(samples[LAB], samples["gross"], rtol=1e-12)
    assert np.all(samples["welfare"] == 0)
    positive = samples[LAB][samples[LAB] > 0]
    assert positive.size and np.all(positive >= 1e5)


def test_compensation_is_a_share_of_the_same_draw_and_business_loss_adds_to_it(
    tmp_path,
):
    instrument = _instrument("fraud_organizations")
    shares = _fraud_shares(instrument)
    _, zero = _run(
        tmp_path / "zero",
        instrument,
        _answer(instrument, shares, total=FRAUD_TOTAL),
    )
    np.testing.assert_allclose(zero[LAB], 0.3 * zero["gross"], rtol=1e-12)
    # The lab pays 40% of the other components' 0.7 of every draw: its loss is
    # 0.3 + 0.4 x 0.7 of the same year's gross, sample by sample.
    _, paid = _run(
        tmp_path / "paid",
        instrument,
        _answer(instrument, shares, total=FRAUD_TOTAL, compensation=0.4),
    )
    for view in ("gross", "welfare"):
        np.testing.assert_array_equal(paid[view], zero[view])
    np.testing.assert_allclose(paid[LAB], 0.58 * paid["gross"], rtol=1e-12)
    assert np.all(paid[LAB] <= paid["gross"])
    business = {"status": "available", "positive_probability": 1, "amount": 50_000}
    _, added = _run(
        tmp_path / "added",
        instrument,
        _answer(
            instrument, shares, total=FRAUD_TOTAL, compensation=0.4, business=business
        ),
    )
    np.testing.assert_array_equal(added["welfare"], zero["welfare"])
    # Each year with an episode, and only such a year, adds USD 50,000 of
    # business loss once, however many episodes it has, to the lab view and
    # to gross alike: counts and societal draws are paired, and the
    # compensation is not counted again.
    occurred = zero["gross"] > 0
    assert np.any(occurred) and not np.all(occurred)
    np.testing.assert_allclose(
        added[LAB] - paid[LAB], 50_000 * occurred, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        added["gross"] - zero["gross"], 50_000 * occurred, rtol=0, atol=1e-6
    )
    assert np.all(added[LAB] <= added["gross"] + 1e-6)


@pytest.mark.parametrize(("share", "expected"), [(0, 20), (0.25, 45), (1, 120)])
def test_episode_lab_loss_is_provider_plus_share_of_others(share, expected):
    law = _fixture_law(_fixture_answer(compensation=share))
    components = {
        row.component_id: row.lab_loss_weights
        for row in law.segments[0].kernel.member_outcomes[0].components
    }
    assert components["fine"] == {"lab": 1.0}
    assert components["victim"] == ({"lab": share} if share else {})
    # USD 20 fine + share x USD 100 victim cell: the share is applied once, and
    # at one it is the whole episode's gross loss, never more. The USD 70
    # business loss is the year's, outside the episode.
    assert _episode_lab(law) == pytest.approx(expected, rel=1e-12)
    assert _annual(law) == 70
    assert law.conditioning_context["lab_compensation_share"] == share


def test_compensation_share_interpolates_inside_the_unit_interval():
    instrument = contextual_instrument()
    low, high = _fixture_answer(compensation=0.2), _fixture_answer(compensation=1)
    law = compile_contextual_node(instrument, instrument.nodes[0], {0: low, 2: high}, 1)
    assert isinstance(law, IncidentLaw)
    (victim,) = (
        row
        for row in law.segments[0].kernel.member_outcomes[0].components
        if row.component_id == "victim"
    )
    assert victim.lab_loss_weights == {"lab": pytest.approx(0.6)}


@pytest.mark.parametrize("share", [-0.1, 1.5, True, "0.5"])
def test_compensation_share_must_be_a_number_in_the_unit_interval(share):
    with pytest.raises(ValidationError):
        LabCompensationAnswer.model_validate({"status": "available", "share": share})
    payload = contextual_answer().model_dump(mode="json")
    payload["lab_compensation"] = {"status": "available", "share": share}
    with pytest.raises(ValidationError):
        ContextualRiskAnswer.model_validate(payload)


def test_compensation_status_rule_is_explicit():
    assert (
        LabCompensationAnswer.model_validate({"status": "available", "share": 0}).share
        == 0
    )
    for payload in (
        {"status": "available"},
        {"status": "unavailable"},
        {"status": "unavailable", "unavailable_reason": "Unknown", "share": 0.2},
        {"status": "available", "share": 0.2, "unavailable_reason": "Unknown"},
    ):
        with pytest.raises(ValidationError):
            LabCompensationAnswer.model_validate(payload)


def test_severity_factor_scales_compensation_but_not_business_loss():
    answer = _fixture_answer(compensation=0.5)
    plain = _fixture_law(answer)
    scaled = _fixture_law(answer, severity_factors={0: 2.0, 2: 2.0})
    # Fixed amounts without a floor scale in proportion: USD 40 fine and half
    # of a USD 200 victim cell; the USD 70 annual business loss is unchanged.
    assert _episode_lab(plain) == pytest.approx(20 + 50, rel=1e-12)
    assert _episode_lab(scaled) == pytest.approx(40 + 100, rel=1e-12)
    assert _annual(plain) == _annual(scaled) == 70
    assert plain.annual_lab_losses == scaled.annual_lab_losses
    assert (
        plain.segments[0].kernel.campaign_lab_losses
        == scaled.segments[0].kernel.campaign_lab_losses
    )


def test_severity_factor_scales_the_compensation_share_of_a_shared_total():
    instrument = _instrument("fraud_organizations")
    business = {
        "status": "available",
        "positive_probability": 0.5,
        "loss": dict(zip(QUANTILES, (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7), strict=True)),
    }
    answer = _answer(
        instrument,
        _fraud_shares(instrument),
        total=FRAUD_TOTAL,
        compensation=0.4,
        business=business,
    )
    node = instrument.nodes[0]
    answers: dict[float, ContextualRiskAnswer] = {0: answer, 2: answer}
    factors: dict[float, float] = {0: 3.0, 2: 3.0}
    plain = compile_contextual_node(instrument, node, answers, 1)
    scaled = compile_contextual_node(
        instrument, node, answers, 1, severity_factors=factors
    )
    assert isinstance(plain, IncidentLaw) and isinstance(scaled, IncidentLaw)
    before, after = plain.segments[0].kernel, scaled.segments[0].kernel
    assert before.campaign_lab_losses == after.campaign_lab_losses
    assert plain.annual_lab_losses == scaled.annual_lab_losses
    for old, new in zip(
        before.member_outcomes[0].components,
        after.member_outcomes[0].components,
        strict=True,
    ):
        assert new.lab_loss_weights == old.lab_loss_weights
        assert new.severity != old.severity or new.post_multiplier == 0


@pytest.mark.parametrize(
    ("frontier", "episode", "annual"), [(60, 60, 60), (1e12, 70, 70)]
)
def test_lab_cap_applies_once_to_the_episode_sum_and_to_the_annual_loss(
    frontier, episode, annual
):
    # Each episode part is below USD 60 (20 and 50), so only one cap on their
    # sum can bind; capping each part separately would leave USD 70. The same
    # value caps the USD 70 annual business loss on its own.
    law = _fixture_law(_fixture_answer(compensation=0.5), frontier_lab_value=frontier)
    (aggregate,) = law.segments[0].kernel.campaign_lab_losses
    assert aggregate.episode_cap == frontier
    assert _episode_lab(law) == pytest.approx(episode, rel=1e-12)
    assert _annual(law) == annual


def test_simulated_lab_cap_applies_once_per_episode(tmp_path):
    instrument = contextual_instrument()

    def large(case):
        case["answer"]["lab_compensation"] = {"status": "available", "share": 0.5}
        case["answer"]["lab_annual_business_loss"] = {
            "status": "available",
            "positive_probability": 1,
            "amount": 2e12,
        }

    _, _, _, capture = _capture(
        tmp_path, instrument, _changed_artifact(instrument, large)
    )
    kernel = capture.kernel_at(instrument.context_id, 1, "reference")
    lab, welfare = kernel.samples(LAB)[0], kernel.samples("welfare")[0]
    gross = kernel.samples("gross")[0]
    # Every year with an episode costs the USD 1 trillion cap once for its
    # business loss, plus USD 20 of fine and USD 50 of compensation for each
    # of its episodes, each with USD 100 of welfare loss.
    episodes = welfare / 100
    np.testing.assert_allclose(episodes, np.round(episodes), atol=1e-9)
    assert np.any(episodes > 1)
    np.testing.assert_allclose(lab, 1e12 * (episodes > 0) + 70 * episodes, rtol=1e-15)
    # Gross counts the same capped business loss beside the USD 120 episodes.
    np.testing.assert_allclose(
        gross, 1e12 * (episodes > 0) + 120 * episodes, rtol=1e-15
    )


def test_provider_only_bound_drops_compensation_and_business_loss():
    law = _fixture_law(_fixture_answer(compensation=0.5), lab_loss_provider_only=True)
    kernel = law.segments[0].kernel
    weights = {
        row.component_id: row.lab_loss_weights
        for row in kernel.member_outcomes[0].components
    }
    assert weights == {"victim": {}, "fine": {"lab": 1.0}}
    assert [row.amount for row in kernel.campaign_lab_losses] == [0]
    assert _episode_lab(law) == 20
    assert law.annual_lab_losses == ()
    assert law.conditioning_context["lab_loss_provider_only"] is True


@pytest.mark.parametrize(
    ("frontier", "expected"), [(1e12, 1e12), (3e11, 3e11), (None, 1e14)]
)
def test_episode_cap_is_the_lower_of_value_and_endpoint_and_spares_welfare(
    frontier, expected
):
    instrument = _instrument("cbrn_biological_expert")
    node = instrument.nodes[0]
    components = [row.component_id for row in node.components]
    shares = dict.fromkeys(components, 0.0)
    shares.update(directly_affected_party_injury=0.5, provider_resource_cost=0.5)
    answer = _answer(
        instrument,
        shares,
        total=(1e9, 1e10, 1e11, 5e11, 2e12, 8e12, 3e13),
        business={"status": "available", "positive_probability": 1, "amount": 2e12},
        compensation=0.1,
    )
    law = compile_contextual_node(
        instrument,
        node,
        {0: answer, 2: answer},
        1,
        options=CompileOptions(frontier_lab_value=frontier),
    )
    reference = compile_contextual_node(
        instrument,
        node,
        {0: answer, 2: answer},
        1,
    )
    assert isinstance(law, IncidentLaw) and isinstance(reference, IncidentLaw)
    kernel = law.segments[0].kernel
    (episode,) = kernel.campaign_lab_losses
    assert episode.episode_cap == expected
    assert episode.amount == 0
    assert _annual(law) == min(2e12, expected)
    weights = {
        row.component_id: row.lab_loss_weights
        for row in kernel.member_outcomes[0].components
    }
    assert weights["provider_resource_cost"] == {"lab": 1.0}
    assert weights["directly_affected_party_injury"] == {"lab": 0.1}
    # The cap belongs to the lab view only: the societal marks are identical.
    assert kernel.member_outcomes == reference.segments[0].kernel.member_outcomes


def test_lab_view_is_known_zero_only_when_every_lab_part_is():
    instrument = contextual_instrument()

    def known_zero(fine, share, business=NONE):
        def change(case):
            components = case["answer"]["branches"][0]["components"]
            components[1]["amount"] = fine
            case["answer"]["lab_compensation"] = {"status": "available", "share": share}
            case["answer"]["lab_annual_business_loss"] = business

        setup = compile_contextual_setup(
            instrument,
            _changed_artifact(instrument, change),
            setup_id="known-zero",
            years_per_state=1,
        )
        state = next(row for row in setup.joint_states if row.weight > 0)
        compiled = compile_incident_state(setup, state.state_id)
        return LAB in analytic_known_zero_views(compiled, (LAB,))

    assert known_zero(0, 0)
    assert not known_zero(20, 0)
    assert not known_zero(0, 0.1)
    assert not known_zero(
        0, 0, {"status": "available", "positive_probability": 1, "amount": 5}
    )


def test_prompt_asks_for_compensation_share_and_annual_business_loss():
    instrument = _instrument("government_designation")
    rendered = ContextualPrompt(
        instrument, instrument.nodes[0], instrument.respondents[0], 0
    ).render()
    assert "(borne by the lab; " in rendered
    for wording in (
        "has three parts, each counted once",
        "already count the harmed parties' own losses",
        "hold only what the lab itself pays to others or loses itself",
        f"lab_compensation, defined as: {instrument.lab_compensation_definition} "
        "Give share: one value from 0 to 1",
        "so that large units count in proportion to their size",
        "use of a stolen model or of openly released weights, state in reasoning "
        "the basis for any payment",
        "lab_annual_business_loss, defined as: "
        f"{instrument.lab_annual_business_loss_definition} Do not multiply a "
        "per-unit loss by the number of units",
        "the probability, a point value, that it is positive in a year with at "
        "least one qualified unit",
        "of the year's total from one such year to the next",
        "do not count a component borne by the lab again",
    ):
        assert wording in rendered, wording
    # Each definition is stated once.
    assert rendered.count(instrument.lab_compensation_definition) == 1
    assert rendered.count(instrument.lab_annual_business_loss_definition) == 1
    assert '"lab_compensation"' in rendered and '"lab_annual_business_loss"' in rendered


def test_an_available_answer_needs_both_lab_answers():
    prompt = contextual_prompts(contextual_instrument())[0]
    payload = contextual_answer().model_dump(mode="json")
    for name in ("lab_compensation", "lab_annual_business_loss"):
        payload.pop(name)
    with pytest.raises(ValueError, match="lab business-loss answers"):
        parse_contextual_answer(
            ContextualRiskAnswer.model_validate(payload).model_dump_json(), prompt
        )


def test_answer_schema_offers_the_two_lab_answers():
    node = contextual_instrument().nodes[0]
    schema = ContextualRiskAnswer.json_schema_for(node)
    share = schema["$defs"]["LabCompensationAnswer"]["properties"]["share"]
    assert {"$ref": "#/$defs/Fraction"} in share["anyOf"]
    fraction = schema["$defs"]["Fraction"]
    assert (fraction["minimum"], fraction["maximum"]) == (0, 1)
    (answered,) = (
        row
        for row in schema["anyOf"]
        if row["properties"]["status"]["const"] == "available"
    )
    assert {"lab_compensation", "lab_annual_business_loss"} <= set(answered["required"])


def test_fixture_artifact_counts_the_fine_and_business_loss_once():
    instrument = contextual_instrument()
    setup = compile_contextual_setup(
        instrument,
        contextual_artifact(instrument),
        setup_id="fixture",
        years_per_state=1,
    )
    law = next(row for state in setup.joint_states for row in state.laws)
    # Free-form records hold their sequences as tuples.
    assert law.conditioning_context["provider_borne_component_ids"] == ("fine",)
    assert law.conditioning_context["lab_loss_method"] == (
        "episode_provider_components_plus_compensation_share_plus_annual_business_loss"
    )
