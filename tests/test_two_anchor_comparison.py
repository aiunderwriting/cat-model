"""The two-anchor comparison: the very-large-loss share as a second anchor.

No reference law reads the stated very-large-loss count. The comparison
(``CompileOptions.very_large_loss_anchor``) passes a law's tail through that
count's share wherever it is coherent: above a first anchor that shapes the
law, or above the law's own p99 where the first level lies within the stated
quantiles. Every contextual setup's ``large_loss_anchor`` exhibit reports its
EAL beside the reference as ``two_anchor``, and the opt-in
``very_large_loss_anchor`` family simulates it.
"""

import math
from typing import cast

import pytest

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    SharedTotal,
)
from catmodel.contracts.elicitation import QUANTILES
from catmodel.contracts.incident import IncidentLaw
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.elicitation.large_loss import very_large_loss_pin
from catmodel.probability.distributions import Distribution
from catmodel.setup.contextual_node import (
    CompileOptions,
    compile_contextual_node,
    shared_total_sampler,
)
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.public_frontier_comparisons import GENERATED_FAMILIES
from catmodel.setup.sensitivity_catalog import (
    DEFAULT_CONTEXT_BOUND_FAMILIES,
    SENSITIVITY_FAMILIES,
)
from test_support.contextual_risk import (
    contextual_case,
    governed_contextual_instrument,
    scenario_node_answer,
)

LEAF = "compromise_organizations"
LARGE = 1e9
VERY = 1e11
RATE = (6.0, 9.0, 12.0, 16.0, 24.0)
# One leaf of each tail class: its governed shape and outer cap.
TAIL_CLASS_LEAVES = {
    "private_claims_privacy": (0.5, None),
    "compromise_organizations": (0.7, None),
    "compromise_critical_infrastructure": (0.8, 1e14),
}
# A stated p95 and p99 that put the first level inside the stated quantiles.
BETWEEN_LEVELS = {"p95": 2e9, "p99": 5e9, "p999": 5e10}


def _answer(
    node,
    share: float = 2e-3,
    ratio: float = 0.01,
    *,
    loss: dict[str, float] | None = None,
    compensation: float = 0.1,
) -> ContextualRiskAnswer:
    """A merged-leaf answer (p95 about USD 6.1 million, p99 about USD 10.1
    million unless ``loss`` moves them) whose large-loss counts are ``share``
    times its counts and whose very-large-loss counts are ``ratio`` times
    those; the lab pays ``compensation`` of what others bear."""
    payload = scenario_node_answer(node).model_dump(mode="json")
    payload["rate"] = dict(zip(QUANTILES, RATE, strict=True))
    payload["large_loss_count"] = {
        key: share * value for key, value in zip(QUANTILES, RATE, strict=True)
    }
    payload["very_large_loss_count"] = {
        key: ratio * value for key, value in payload["large_loss_count"].items()
    }
    payload["lab_compensation"] = {"status": "available", "share": compensation}
    payload["total_loss"].update(loss or {})
    return ContextualRiskAnswer.model_validate(payload)


def _instrument(leaf: str = LEAF) -> ContextualRiskInstrument:
    return governed_contextual_instrument((leaf,))


def _law(instrument, answer, **options) -> IncidentLaw:
    law = compile_contextual_node(
        instrument,
        instrument.nodes[0],
        dict.fromkeys(instrument.anchor_scores, answer),
        0.5,
        options=CompileOptions(**options),
    )
    assert isinstance(law, IncidentLaw)
    return law


def _total(law: IncidentLaw) -> Distribution:
    total = shared_total_sampler(law)
    assert total is not None
    return total


def _second(law: IncidentLaw) -> dict[str, object]:
    return cast(dict[str, object], law.conditioning_context["very_large_loss_anchor"])


def _first_role(law: IncidentLaw) -> object:
    record = cast(dict[str, object], law.conditioning_context["large_loss_anchor"])
    return record["large_loss_role"]


def _spec(node) -> SharedTotal:
    assert node.total is not None
    return node.total


def _hand_law(node, answer, share, second=None) -> Distribution:
    """The shared total's law built from the answer's own quantiles, the
    node's governed shape, floor and cap and the given anchors."""
    loss, total = answer.total_loss, _spec(node)
    return Distribution(
        *(getattr(loss, key) for key in QUANTILES),
        gpd_shape=total.gpd_shape,
        gpd_tail_ratio=loss.p99 / loss.p95,
        severity_minimum=total.floor,
        severity_cap=total.cap,
        large_loss_amount=LARGE,
        large_loss_exceedance=share,
        very_large_loss_amount=None if second is None else VERY,
        very_large_loss_exceedance=second,
    )


def _exhibit(instrument, answers_for) -> dict:
    artifact = ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(
            contextual_case(instrument, prompt, answers_for(prompt))
            for prompt in contextual_prompts(instrument)
        ),
    )
    setup = compile_contextual_setup(
        instrument, artifact, setup_id="two-anchor", years_per_state=1
    )
    return next(
        cast(dict, row.acquisition["large_loss_anchor"])
        for row in setup.evidence.records
        if row.acquisition and "large_loss_anchor" in row.acquisition
    )


def test_no_reference_law_reads_the_very_large_loss_count():
    """Two answers that differ only in their very-large-loss counts compile
    one reference law, whose record does not name that count, so a state's
    identity does not depend on it."""
    instrument = _instrument()
    node = instrument.nodes[0]
    low, high = (_law(instrument, _answer(node, ratio=ratio)) for ratio in (1e-4, 0.5))
    assert low.model_dump(mode="json") == high.model_dump(mode="json")
    assert "very_large_loss_anchor" not in low.conditioning_context
    component = low.segments[0].kernel.member_outcomes[0].components[0]
    assert component.severity is not None
    stored = component.severity.components[0].model_dump(mode="json")
    assert not {"very_large_loss_amount", "very_large_loss_exceedance"} & set(stored)
    two = [
        _law(instrument, _answer(node, ratio=ratio), very_large_loss_anchor=True)
        for ratio in (1e-4, 0.5)
    ]
    assert _total(two[0]).analytic_mean() < _total(two[1]).analytic_mean()


def test_a_coherent_second_share_shapes_the_tail_above_the_large_loss_level():
    instrument = _instrument()
    answer = _answer(instrument.nodes[0], 2e-3, 0.01)
    reference = _law(instrument, answer)
    law = _law(instrument, answer, very_large_loss_anchor=True)
    record = _second(law)
    assert record["very_large_loss_role"] == "shapes_tail_above_large_loss_level"
    assert record["very_large_loss_level"] == VERY
    assert record["very_large_loss_share"] == pytest.approx(2e-5, rel=1e-12)
    total, base = _total(law), _total(reference)
    assert total.survival_probability(LARGE) == pytest.approx(2e-3, rel=1e-12)
    assert total.survival_probability(VERY) == pytest.approx(2e-5, rel=1e-12)
    # The reference tail reaches the higher level about seven times less often.
    assert base.survival_probability(VERY) == pytest.approx(2.76e-6, rel=0.01)
    # Under the rank-integrated count the compiled model's expected annual
    # number at or above the higher level is the stated count's mean.
    assert answer.very_large_loss_count is not None
    assert law.reference_annual_rate * total.survival_probability(VERY) == (
        pytest.approx(epistemic_mean(answer.very_large_loss_count), rel=1e-9)
    )
    # The first anchor's record, and the law at and below its level, are the
    # reference's.
    assert (
        law.conditioning_context["large_loss_anchor"]
        == reference.conditioning_context["large_loss_anchor"]
    )
    for rank in (0.5, 0.99, 0.997):
        assert total.quantile(rank) == base.quantile(rank)
    assert total.analytic_mean() == pytest.approx(
        _hand_law(instrument.nodes[0], answer, 2e-3, 2e-5).analytic_mean(), rel=1e-12
    )


@pytest.mark.parametrize(
    ("share", "ratio", "loss", "first", "second"),
    [
        # Every incident that reaches the first level reaches the second.
        (2e-3, 1.0, None, "shapes_tail_above_p99", "not_below_large_loss_share"),
        # A share within a tie below the first's is the first's.
        (
            2e-3,
            1 - 1e-10,
            None,
            "shapes_tail_above_p99",
            "not_below_large_loss_share",
        ),
        # An exact zero is no second anchor.
        (2e-3, 0.0, None, "shapes_tail_above_p99", "no_anchor"),
        # The first level within the stated quantiles and the second above
        # p99 with a share of 0.04, which implies a p99 above it.
        (
            0.2,
            0.2,
            BETWEEN_LEVELS,
            "diagnostic_within_stated_quantiles",
            "contradicts_stated_quantiles",
        ),
        # Both levels within the stated quantiles, with coherent shares.
        (
            0.5,
            0.2,
            {"p95": 2e11, "p99": 5e11, "p999": 5e12},
            "diagnostic_within_stated_quantiles",
            "diagnostic_within_stated_quantiles",
        ),
        # An unresolved first anchor: no anchor shapes the law.
        (
            0.06,
            0.01,
            None,
            "severity_unresolved_fallback",
            "large_loss_anchor_unresolved",
        ),
    ],
)
def test_a_second_share_the_law_cannot_take_leaves_the_reference_tail(
    share, ratio, loss, first, second
):
    instrument = _instrument()
    answer = _answer(instrument.nodes[0], share, ratio, loss=loss)
    reference = _law(instrument, answer)
    law = _law(instrument, answer, very_large_loss_anchor=True)
    assert _first_role(reference) == first
    assert _second(law)["very_large_loss_role"] == second
    assert law.model_dump(
        mode="json", exclude={"conditioning_context"}
    ) == reference.model_dump(mode="json", exclude={"conditioning_context"})


def test_within_the_stated_quantiles_the_second_share_alone_anchors_the_tail():
    """With a stated p99 between the levels the first anchor is a diagnostic
    and the stated quantiles govern up to p99; the second level lies above
    p99, so its share is the law's only anchor, under the first anchor's own
    rule."""
    instrument = _instrument()
    answer = _answer(instrument.nodes[0], 0.2, 0.02, loss=BETWEEN_LEVELS)
    reference = _law(instrument, answer)
    law = _law(instrument, answer, very_large_loss_anchor=True)
    assert _second(law)["very_large_loss_role"] == "shapes_tail_above_p99"
    total, base = _total(law), _total(reference)
    assert total.large_loss_amount == VERY
    assert total.large_loss_exceedance == pytest.approx(4e-3, rel=1e-12)
    assert total.very_large_loss_amount is None
    assert base.large_loss_amount is None
    assert total.survival_probability(VERY) == pytest.approx(4e-3, rel=1e-12)
    for rank in (0.5, 0.95, 0.99):
        assert total.quantile(rank) == base.quantile(rank)
    # The stated count is 34 times the reference tail's, so the mean rises.
    assert base.survival_probability(VERY) == pytest.approx(4e-3 / 34, rel=0.02)
    assert total.analytic_mean() > 4 * base.analytic_mean()


def test_the_law_is_continuous_where_p99_crosses_the_large_loss_level():
    """Coherent answers move p99 through the first level only with the first
    share at 0.01. Just below, both anchors shape the law; just above, the
    second alone does; the two laws meet there."""
    instrument = _instrument()
    node = instrument.nodes[0]
    step = 1e-6
    below, above = (
        _answer(
            node,
            0.01 * (1 + sign * step),
            0.01,
            loss={"p95": 6e8, "p99": LARGE * (1 + sign * step), "p999": 5e10},
        )
        for sign in (-1, 1)
    )
    laws = [
        _law(instrument, answer, very_large_loss_anchor=True)
        for answer in (below, above)
    ]
    assert [_second(law)["very_large_loss_role"] for law in laws] == [
        "shapes_tail_above_large_loss_level",
        "shapes_tail_above_p99",
    ]
    means = [_total(law).analytic_mean() for law in laws]
    assert means[0] == pytest.approx(means[1], rel=50 * step)
    for rank in (0.9, 0.995, 0.9995, 1 - 1e-6):
        assert _total(laws[0]).quantile(rank) == pytest.approx(
            _total(laws[1]).quantile(rank), rel=50 * step
        )


def test_a_level_at_or_above_the_cap_is_not_asked_and_has_no_second_anchor():
    payload = _instrument().model_dump(mode="json")
    total = payload["nodes"][0]["total"]
    total["cap"] = 5e10
    del total["very_large_loss_level"]
    total["large_loss_level_exclusions"] = ["The level is at or above the cap."]
    instrument = ContextualRiskInstrument.model_validate(payload)
    answer_payload = _answer(instrument.nodes[0]).model_dump(mode="json")
    del answer_payload["very_large_loss_count"]
    answer = ContextualRiskAnswer.model_validate(answer_payload)
    reference = _law(instrument, answer)
    law = _law(instrument, answer, very_large_loss_anchor=True)
    assert law.conditioning_context["very_large_loss_anchor"] is None
    assert _total(law) == _total(reference)


def test_ignoring_the_anchor_ignores_both():
    instrument = _instrument()
    answer = _answer(instrument.nodes[0])
    law = _law(instrument, answer, large_loss_factor=None, very_large_loss_anchor=True)
    total = _total(law)
    assert total.large_loss_amount is None and total.very_large_loss_amount is None
    assert law.conditioning_context["large_loss_anchor"] is None
    assert law.conditioning_context["very_large_loss_anchor"] is None


def test_a_severity_factor_maps_both_anchors():
    instrument = _instrument()
    node = instrument.nodes[0]
    answer = _answer(node)
    law = compile_contextual_node(
        instrument,
        node,
        dict.fromkeys(instrument.anchor_scores, answer),
        0.5,
        severity_factors=dict.fromkeys(instrument.anchor_scores, 2.0),
        options=CompileOptions(very_large_loss_anchor=True),
    )
    assert isinstance(law, IncidentLaw)
    total = _total(law)
    floor = _spec(node).floor
    for level, share in ((LARGE, 2e-3), (VERY, 2e-5)):
        assert total.survival_probability(floor + 2.0 * (level - floor)) == (
            pytest.approx(share, rel=1e-12)
        )


def test_between_anchors_the_second_share_comes_from_the_interpolated_quantiles():
    instrument = _instrument()
    node = instrument.nodes[0]
    lower, upper = _answer(node, 2e-3, 0.01), _answer(node, 4e-3, 0.05)
    scores = sorted(instrument.anchor_scores)
    answers = {scores[0]: lower, scores[1]: lower, scores[2]: upper}
    score = 0.5 * (scores[1] + scores[2])
    law = compile_contextual_node(
        instrument,
        node,
        answers,
        score,
        options=CompileOptions(very_large_loss_anchor=True),
    )
    assert isinstance(law, IncidentLaw)
    # The counts are equal at both anchors, so each share is the mean of the
    # anchors' shares: 3e-3 and (2e-5 + 2e-4)/2.
    total = _total(law)
    assert total.survival_probability(LARGE) == pytest.approx(3e-3, rel=1e-9)
    assert total.survival_probability(VERY) == pytest.approx(1.1e-4, rel=1e-9)
    assert _second(law)["very_large_loss_role"] == "shapes_tail_above_large_loss_level"


def test_the_second_pin_needs_a_positive_share_below_the_firsts():
    first = (LARGE, 2e-3)
    assert very_large_loss_pin(VERY, 2e-5, first) == (VERY, 2e-5)
    assert very_large_loss_pin(VERY, 2e-3 * (1 - 1e-8), first) is not None
    for share in (None, 0.0, 2e-3, 2e-3 * (1 - 1e-10), 3e-3):
        assert very_large_loss_pin(VERY, share, first) is None
    assert very_large_loss_pin(LARGE, 2e-5, first) is None


def test_exhibit_reports_the_two_anchor_tails_eal_and_counts():
    """The exhibit's ``two_anchor`` EAL is the count times the two-anchor
    law's mean, here checked against the law built by hand from the answer.
    Welfare and gross are fixed shares of the uncapped total, so both move by
    the ratio of the two laws' means."""
    instrument = _instrument()
    node = instrument.nodes[0]
    answer = _answer(node, 2e-3, 0.01)
    record = _exhibit(instrument, lambda _prompt: answer)
    row = record["by_scenario"][LEAF]
    one = _hand_law(node, answer, 2e-3)
    two = _hand_law(node, answer, 2e-3, 2e-5)
    expected = two.analytic_mean() / one.analytic_mean() - 1
    alternatives = row["eal_alternatives"]
    for view in ("welfare", "gross"):
        entry = alternatives[view]["two_anchor"]
        assert entry["status"] == "available"
        assert entry["relative_effect"] == pytest.approx(expected, rel=1e-6)
        assert entry["expected_annual_loss"] == pytest.approx(
            alternatives[view]["expected_annual_loss"] * (1 + expected), rel=1e-6
        )
    # The effect the documents quote for this leaf at a stated 1 in 100.
    assert expected == pytest.approx(0.79, abs=0.01)
    # The lab bears a fixed share of each incident up to its episode cap, so
    # its EAL moves a little less.
    lab = next(view for view in alternatives if view.startswith("lab_loss::"))
    assert (
        0.9 * expected < alternatives[lab]["two_anchor"]["relative_effect"] < (expected)
    )
    # The two-anchor tail reproduces the pooled stated count at the higher
    # level; the reference tail gives the shape's own, about a seventh of it.
    tail = row["two_anchor_tail"]
    assert tail["very_large_count"] == pytest.approx(
        row["pooled_very_large_count"], rel=1e-9
    )
    assert row["modelled_very_large_count"] == pytest.approx(
        row["pooled_very_large_count"] / 7.24, rel=0.01
    )
    assert tail["large_count"] == pytest.approx(row["modelled_large_count"], rel=1e-12)
    assert (
        tail["gross_eal_share_at_or_above_large"]
        > (row["gross_eal_share_at_or_above_large"])
    )
    assert row["very_large_law_roles"] == {"shapes_tail_above_large_loss_level": 1}
    assert record["total"]["very_large_law_roles"] == row["very_large_law_roles"]
    assert "two_anchor compiles every law" in record["interpretation"]


@pytest.mark.parametrize(
    ("leaf", "lower", "higher", "hundredth", "lab_lower", "lab_higher"),
    [
        ("private_claims_privacy", -0.073, 0.199, 1.022, -0.073, 0.198),
        ("compromise_organizations", -0.211, 1.085, 0.790, -0.201, 0.970),
        ("compromise_critical_infrastructure", -0.325, 2.090, 0.569, -0.296, 1.743),
    ],
)
def test_measured_effect_of_a_second_share_ten_times_off_the_shapes(
    leaf, lower, higher, hundredth, lab_lower, lab_higher
):
    """Welfare, gross and lab-loss EAL under the two-anchor tail, against the
    reference, on one leaf of each tail class with a first share of 2e-3: a
    second share a factor of ten below and above the one the governed shape
    implies, S(USD 100 billion)/S(USD 1 billion) of the reference law, and a
    stated 1 in 100. assumptions.md (SEV-LARGE-LOSS-ANCHOR) quotes these
    effects."""
    instrument = _instrument(leaf)
    node = instrument.nodes[0]
    shape, cap = TAIL_CLASS_LEAVES[leaf]
    assert (_spec(node).gpd_shape, _spec(node).cap) == (shape, cap)
    base = _total(_law(instrument, _answer(node, 2e-3, 0.0)))
    implied = base.survival_probability(VERY) / base.survival_probability(LARGE)
    assert implied == pytest.approx(100 ** (-1 / shape), rel=0.01)
    effects = {}
    ratios = {
        "lower": 0.1 * implied,
        "same": implied,
        "higher": 10 * implied,
        "hundredth": 0.01,
    }
    for label, ratio in ratios.items():
        answer = _answer(node, 2e-3, ratio)
        row = _exhibit(instrument, lambda _prompt, answer=answer: answer)[
            "by_scenario"
        ][leaf]
        effects[label] = {
            view.split("::")[0]: entry["two_anchor"]["relative_effect"]
            for view, entry in row["eal_alternatives"].items()
        }
        assert row["very_large_law_roles"] == {"shapes_tail_above_large_loss_level": 1}
    for view in ("welfare", "gross"):
        assert effects["lower"][view] == pytest.approx(lower, abs=0.003)
        assert effects["higher"][view] == pytest.approx(higher, abs=0.01)
        assert effects["hundredth"][view] == pytest.approx(hundredth, abs=0.01)
        # At the shape's own share only the form between the levels differs.
        assert abs(effects["same"][view]) < 2e-3
    assert effects["lower"]["lab_loss"] == pytest.approx(lab_lower, abs=0.003)
    assert effects["higher"]["lab_loss"] == pytest.approx(lab_higher, abs=0.01)


def test_a_share_that_does_not_shape_the_law_leaves_the_alternative_at_the_run_law():
    instrument = _instrument()
    node = instrument.nodes[0]
    record = _exhibit(instrument, lambda _prompt: _answer(node, 2e-3, 1.0))
    row = record["by_scenario"][LEAF]
    assert row["very_large_law_roles"] == {"not_below_large_loss_share": 1}
    for view, entry in row["eal_alternatives"].items():
        assert entry["two_anchor"]["status"] == "available", view
        assert entry["two_anchor"]["relative_effect"] == 0
    assert (
        row["two_anchor_tail"]["very_large_count"] == (row["modelled_very_large_count"])
    )
    # The stated count is then far above anything either tail gives.
    assert row["pooled_very_large_count"] > 100 * row["modelled_very_large_count"]


def test_total_two_anchor_alternative_sums_the_scenarios():
    instrument = governed_contextual_instrument((LEAF, "fraud_organizations"))
    ratios = {LEAF: 0.05, "fraud_organizations": 0.0}
    record = _exhibit(
        instrument,
        lambda prompt: _answer(prompt.node, 2e-3, ratios[prompt.node.node_id]),
    )
    rows = record["by_scenario"]
    assert rows[LEAF]["very_large_law_roles"] == {
        "shapes_tail_above_large_loss_level": 1
    }
    assert rows["fraud_organizations"]["very_large_law_roles"] == {"no_anchor": 1}
    assert record["total"]["very_large_law_roles"] == {
        "no_anchor": 1,
        "shapes_tail_above_large_loss_level": 1,
    }
    total = record["total"]["eal_alternatives"]["welfare"]
    scenario_eals = [
        rows[scenario]["eal_alternatives"]["welfare"]["two_anchor"][
            "expected_annual_loss"
        ]
        for scenario in ratios
    ]
    assert total["two_anchor"]["expected_annual_loss"] == pytest.approx(
        math.fsum(scenario_eals), rel=1e-12
    )
    assert (
        rows["fraud_organizations"]["eal_alternatives"]["welfare"]["two_anchor"][
            "relative_effect"
        ]
        == 0
    )
    assert (
        0
        < total["two_anchor"]["relative_effect"]
        < (rows[LEAF]["eal_alternatives"]["welfare"]["two_anchor"]["relative_effect"])
    )
    assert record["total"]["two_anchor_tail"]["very_large_count"] == pytest.approx(
        sum(rows[scenario]["two_anchor_tail"]["very_large_count"] for scenario in rows),
        rel=1e-12,
    )


def test_the_simulated_family_is_opt_in_and_reads_the_elicited_share():
    families = {row.family_id: row for row in SENSITIVITY_FAMILIES}
    family = families["very_large_loss_anchor"]
    assert family.assumption_ids == ("SEV-LARGE-LOSS-ANCHOR",)
    assert family.kind == "assumption" and family.range_basis == "elicited"
    assert "No reference law reads that share" in family.description
    assert "very_large_loss_anchor" not in DEFAULT_CONTEXT_BOUND_FAMILIES
    assert "very_large_loss_anchor" in GENERATED_FAMILIES


def test_a_run_reports_the_two_anchor_alternative_in_its_results_and_summary(tmp_path):
    """A contextual run carries the alternative in its results, whose two
    files conform to their committed schemas, and its readable summary shows
    each view's EAL under it beside the reference and the counts at both
    levels."""
    from catmodel.results.results_summary import results_summary
    from test_support.results_schema import schema_violations
    from tests.test_contextual_study import _capture

    instrument = _instrument()
    answer = _answer(instrument.nodes[0], 2e-3, 0.01)
    artifact = ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(
            contextual_case(instrument, prompt, answer)
            for prompt in contextual_prompts(instrument)
        ),
    )
    _, _, report, _ = _capture(tmp_path, instrument, artifact)
    assert schema_violations(tmp_path / "run" / "results.json") == []
    total = report["sensitivity_summary"]["large_loss_anchor"]["total"]
    welfare = total["eal_alternatives"]["welfare"]
    assert welfare["expected_annual_loss"] == pytest.approx(
        report["arms"]["reference"]["metrics"]["welfare"]["expected_annual_loss"],
        rel=1e-9,
    )
    assert welfare["two_anchor"]["relative_effect"] == pytest.approx(0.79, abs=0.01)
    markdown, rows = results_summary(report)
    assert "## Large-loss anchor" in markdown
    assert "| View | Run laws | Anchor ignored |" in markdown
    assert "1 `shapes_tail_above_large_loss_level`" in markdown
    listed = {
        (row["subject"], row["view"], row["statistic"]): cast(float, row["value"])
        for row in rows
        if row["section"] == "large_loss_anchor"
    }
    assert listed["two_anchor", "welfare", "relative_change"] == pytest.approx(
        welfare["two_anchor"]["relative_effect"]
    )
    assert listed["run", "welfare", "expected_annual_loss"] == pytest.approx(
        welfare["expected_annual_loss"]
    )
    assert listed["very_large_loss::two_anchor", None, "expected_annual_count"] == (
        pytest.approx(
            listed["very_large_loss::pooled", None, "expected_annual_count"], rel=1e-9
        )
    )
    assert listed["very_large_loss::modelled", None, "expected_annual_count"] < (
        listed["very_large_loss::pooled", None, "expected_annual_count"] / 7
    )
