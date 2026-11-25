"""The large-loss anchor's comparison alternatives and its analytic exhibit.

The ``large_loss_anchor`` family scales every shaping share's conditional odds
above p99 by one third and three; the ``large_loss_anchor_ignored`` diagnostic
uses the p99-only tail, a design comparison. Every contextual setup carries the
``large_loss_anchor`` exhibit: stated, pooled, extrapolated and modelled counts
at USD 1 billion, stated and modelled counts at USD 100 billion, the probability
of at least one incident in the year, the gross EAL share from such incidents,
the tail-shape alternatives and the anchor statuses.
"""

import copy
import json
import math
from collections import defaultdict
from typing import cast

import numpy as np
import pytest

from catmodel.collection.contextual_collection import run_contextual_collection
from catmodel.config import ModelProfile
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskInstrument,
)
from catmodel.contracts.elicitation import QUANTILES
from catmodel.contracts.incident import IncidentLaw
from catmodel.contracts.native_severity import native_severity_sampler
from catmodel.elicitation.epistemic import epistemic_mean, epistemic_quantiles
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import ProviderResponse
from catmodel.probability.distributions import LARGE_LOSS_BREAK_EXCEEDANCE
from catmodel.results.execution import run_setup
from catmodel.results.results_summary import results_summary
from catmodel.setup.contextual_node import (
    CompileOptions,
    compile_contextual_node,
    odds_scaled_large_loss_share,
)
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.public_study import (
    import_public_frontier_study,
    write_public_frontier_study,
)
from catmodel.setup.sensitivity_catalog import (
    DEFAULT_CONTEXT_BOUND_FAMILIES,
    LARGE_LOSS_ANCHOR_FACTORS,
    SENSITIVITY_FAMILIES,
)
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    contextual_case,
    fixture_response,
    governed_contextual_instrument,
    scenario_node_answer,
    with_access_bases,
)
from test_support.public_frontier import synthetic_forecast
from test_support.results import (
    available_metric,
    present,
    reduced_arm,
    reported_exhibit,
)

LEAF = "compromise_organizations"
LEVEL = 1e9
RATE = (6.0, 9.0, 12.0, 16.0, 24.0)


def _answer(node, share: float = 1e-3, *, contradicting: bool = False):
    """A merged-leaf answer (p95 about USD 6.1 million, p99 about USD 10.1
    million) whose large-loss counts are ``share`` times its counts."""
    payload = scenario_node_answer(node).model_dump(mode="json")
    payload["rate"] = dict(zip(QUANTILES, RATE, strict=True))
    factor = 0.06 if contradicting else share
    payload["large_loss_count"] = {
        key: factor * value for key, value in zip(QUANTILES, RATE, strict=True)
    }
    payload["very_large_loss_count"] = {
        key: 0.01 * value for key, value in payload["large_loss_count"].items()
    }
    return ContextualRiskAnswer.model_validate(payload)


def _instrument():
    return governed_contextual_instrument((LEAF,))


def _law(instrument, answer, **options):
    return compile_contextual_node(
        instrument,
        instrument.nodes[0],
        dict.fromkeys(instrument.anchor_scores, answer),
        0.5,
        options=CompileOptions(**options),
    )


def _survival(law, level=LEVEL):
    assert isinstance(law, IncidentLaw)
    severity = next(
        component.severity
        for component in law.segments[0].kernel.member_outcomes[0].components
        if component.severity is not None
    )
    return native_severity_sampler(severity).survival_probability(level)


def _shared_total_exceedance(law):
    assert isinstance(law, IncidentLaw)
    return next(
        component.severity.components[0].large_loss_exceedance
        for segment in law.segments
        for outcome in segment.kernel.member_outcomes
        for component in outcome.components
        if component.severity is not None
        and component.rank_group_id is not None
        and component.rank_group_id.endswith("::shared_total")
    )


SHAPING_SHARES = (1e-9, 1e-4, 2e-3, 1 / 300, 4e-3, 9e-3, 9.99e-3, 0.01 - 1e-10)


@pytest.mark.parametrize("share", SHAPING_SHARES)
def test_odds_scaled_large_loss_shares_stay_inside_the_bound_and_are_symmetric(
    share,
):
    bound = LARGE_LOSS_BREAK_EXCEEDANCE
    lower = odds_scaled_large_loss_share(share, 1 / 3)
    higher = odds_scaled_large_loss_share(share, 3.0)
    assert 0 < lower < share < higher < bound

    def log_odds(value):
        """log q/(1 - q) at q = value/bound, without forming 1 - q."""
        return math.log(value) - math.log(bound - value)

    # Within a few ulps of the bound, bound - share keeps few significant
    # digits, so the log-odds there are checked to that precision.
    rel = 1e-9 if bound - share > 1e-6 else 1e-4
    base = log_odds(share)
    assert log_odds(lower) - base == pytest.approx(-math.log(3), rel=rel)
    assert log_odds(higher) - base == pytest.approx(math.log(3), rel=rel)
    assert odds_scaled_large_loss_share(higher, 1 / 3) == pytest.approx(
        share, rel=1e-12
    )


def test_odds_scaled_large_loss_share_has_the_small_share_limit():
    share = 1e-7
    assert odds_scaled_large_loss_share(share, 3.0) / share == pytest.approx(
        3.0, rel=1e-4
    )
    assert odds_scaled_large_loss_share(share, 1 / 3) / share == pytest.approx(
        1 / 3, rel=1e-4
    )


def test_odds_scaled_large_loss_share_matches_the_odds_definition():
    share = 2e-3
    odds = share / (LARGE_LOSS_BREAK_EXCEEDANCE - share)

    def from_odds(factor):
        scaled_odds = factor * odds
        return LARGE_LOSS_BREAK_EXCEEDANCE * scaled_odds / (1 + scaled_odds)

    assert odds_scaled_large_loss_share(share, 3.0) == pytest.approx(
        from_odds(3.0), rel=1e-12
    )
    assert odds_scaled_large_loss_share(share, 1 / 3) == pytest.approx(
        from_odds(1 / 3), rel=1e-12
    )
    assert from_odds(3.0) == pytest.approx(
        3 * 2e-3 * 0.01 / (0.01 + 2 * 2e-3), rel=1e-12
    )
    assert from_odds(1 / 3) == pytest.approx(
        (2e-3 / 3) * 0.01 / (0.01 - (2 / 3) * 2e-3), rel=1e-12
    )


def test_odds_scaled_large_loss_share_stays_strictly_inside_at_float_endpoints():
    """Within an ulp of the bound, or at the smallest positive float, rounding
    would reach an endpoint; the result stays the nearest float inside."""
    below = math.nextafter(LARGE_LOSS_BREAK_EXCEEDANCE, 0.0)
    assert odds_scaled_large_loss_share(below, 3.0) == below
    assert 0 < odds_scaled_large_loss_share(below, 1 / 3) < below
    tiny = math.ulp(0.0)
    assert odds_scaled_large_loss_share(tiny, 1 / 3) == tiny
    assert odds_scaled_large_loss_share(tiny, 3.0) > tiny


def test_odds_scaled_large_loss_share_validates_its_inputs():
    assert odds_scaled_large_loss_share(0.0, 1 / 3) == 0.0
    for share in (-1.0, LARGE_LOSS_BREAK_EXCEEDANCE, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            odds_scaled_large_loss_share(share, 3.0)
    for factor in (0.0, -1.0, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            odds_scaled_large_loss_share(2e-3, factor)


def test_simulated_families_are_opt_in_and_the_ignored_arm_is_a_diagnostic():
    families = {row.family_id: row for row in SENSITIVITY_FAMILIES}
    scaled, ignored = (
        families["large_loss_anchor"],
        families["large_loss_anchor_ignored"],
    )
    assert scaled.assumption_ids == ignored.assumption_ids == ("SEV-LARGE-LOSS-ANCHOR",)
    assert scaled.kind == "assumption" and scaled.range_basis == "judgment"
    assert ignored.kind == "diagnostic"
    assert "not an assumption challenge or a lower bound" in ignored.description
    assert not {"large_loss_anchor", "large_loss_anchor_ignored"} & set(
        DEFAULT_CONTEXT_BOUND_FAMILIES
    )
    assert pytest.approx((1 / 3, 3.0)) == LARGE_LOSS_ANCHOR_FACTORS


def test_alternatives_ignore_or_scale_the_pinning_share():
    instrument = _instrument()
    share = 2e-3
    answer = _answer(instrument.nodes[0], share)
    base = _survival(_law(instrument, answer))
    assert base == pytest.approx(share, rel=1e-12)
    assert _survival(
        _law(instrument, answer, large_loss_factor=1 / 3)
    ) == pytest.approx(odds_scaled_large_loss_share(share, 1 / 3), rel=1e-12)
    assert _survival(_law(instrument, answer, large_loss_factor=3.0)) == pytest.approx(
        odds_scaled_large_loss_share(share, 3.0), rel=1e-12
    )
    ignored = _law(instrument, answer, large_loss_factor=None)
    assert _survival(ignored) < base / 10
    assert isinstance(ignored, IncidentLaw)
    assert ignored.conditioning_context["large_loss_anchor"] is None


def test_both_alternatives_compile_near_the_probability_bound():
    instrument = _instrument()
    for share in (4e-3, 9.9e-3):
        answer = _answer(instrument.nodes[0], share)
        base = _survival(_law(instrument, answer))
        lower = _law(instrument, answer, large_loss_factor=1 / 3)
        higher = _law(instrument, answer, large_loss_factor=3.0)
        assert isinstance(lower, IncidentLaw)
        assert isinstance(higher, IncidentLaw)
        lower_survival = _survival(lower)
        higher_survival = _survival(higher)
        assert lower_survival == pytest.approx(
            odds_scaled_large_loss_share(share, 1 / 3), rel=1e-12
        )
        assert higher_survival == pytest.approx(
            odds_scaled_large_loss_share(share, 3.0), rel=1e-12
        )
        assert 0 < lower_survival < 0.01
        assert 0 < higher_survival < 0.01
        assert lower_survival < base < higher_survival
    # A share just beyond the tie tolerance below the bound compiles both alternatives.
    answer = _answer(instrument.nodes[0], 0.01 - 1e-10)
    for factor in LARGE_LOSS_ANCHOR_FACTORS:
        law = _law(instrument, answer, large_loss_factor=factor)
        assert isinstance(law, IncidentLaw)
        assert 0 < _shared_total_exceedance(law) < LARGE_LOSS_BREAK_EXCEEDANCE
    # A share within rounding of the bound is the bound, which never shapes it.
    answer = _answer(instrument.nodes[0], math.nextafter(0.01, 0.0))
    for factor in LARGE_LOSS_ANCHOR_FACTORS:
        assert (
            _shared_total_exceedance(_law(instrument, answer, large_loss_factor=factor))
            is None
        )


def test_a_scaled_alternative_leaves_a_diagnostic_anchor_alone():
    instrument = _instrument()
    payload = _answer(instrument.nodes[0], 0.2).model_dump(mode="json")
    payload["total_loss"].update(p95=2e9, p99=5e9, p999=5e10)
    answer = ContextualRiskAnswer.model_validate(payload)
    assert _survival(_law(instrument, answer, large_loss_factor=3.0)) == _survival(
        _law(instrument, answer)
    )


def _case(instrument, prompt, answer, retried=None):
    return contextual_case(instrument, prompt, answer, retried=retried)


def _exhibit(instrument, answers_for):
    from catmodel.contracts.contextual_risk import ContextualRiskArtifact
    from catmodel.elicitation.contextual_risk import contextual_prompts

    artifact = ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(
            _case(instrument, prompt, *answers_for(prompt))
            for prompt in contextual_prompts(instrument)
        ),
    )
    setup = compile_contextual_setup(
        instrument, artifact, setup_id="anchor", years_per_state=1
    )
    record = next(
        cast(dict, row.acquisition["large_loss_anchor"])
        for row in setup.evidence.records
        if row.acquisition and "large_loss_anchor" in row.acquisition
    )
    return setup, record


def test_exhibit_reports_stated_extrapolated_and_modelled_counts():
    from catmodel.elicitation.contextual_risk import contextual_prompts
    from catmodel.elicitation.eal_crosscheck import pooled_answer

    instrument = _instrument()
    answer = _answer(instrument.nodes[0], 2e-3)
    setup, record = _exhibit(instrument, lambda prompt: (answer,))
    row = record["by_scenario"][LEAF]
    assert row["status"] == "available"
    assert record["levels"] == {"large_loss": LEVEL, "very_large_loss": 1e11}
    assert (
        answer.large_loss_count is not None and answer.very_large_loss_count is not None
    )
    stated = epistemic_mean(answer.large_loss_count)
    case = _case(instrument, contextual_prompts(instrument)[0], answer)
    pooled = pooled_answer(case)
    assert pooled.large_loss_count is not None
    pooled_large = epistemic_mean(pooled.large_loss_count)
    assert row["stated_large_count"] == pytest.approx(stated, rel=1e-12)
    assert row["pooled_large_count"] == pytest.approx(pooled_large, rel=1e-12)
    # The rank-integrated modelled count reproduces the pooled anchor.
    assert row["modelled_large_count"] == pytest.approx(pooled_large, rel=1e-9)
    assert row["extrapolated_large_count"] < stated / 10
    assert row["stated_very_large_count"] == pytest.approx(
        epistemic_mean(answer.very_large_loss_count), rel=1e-12
    )
    assert 0 < row["modelled_very_large_count"] < row["modelled_large_count"]
    assert 0 < row["gross_eal_share_at_or_above_large"] < 1
    assert 0 < row["probability_at_least_one_large"] < 1
    assert row["case_statuses"] == {"coherent": 1}
    assert row["severity_unresolved_gross_eal_share"] == 0
    alternatives = row["tail_shape_alternatives"]
    for alternative in ("shape_lower", "shape_higher"):
        # The shape alternatives keep the USD 1 billion count and move the tail.
        assert alternatives[alternative]["large_count"] == pytest.approx(
            row["modelled_large_count"], rel=1e-9
        )
        assert alternatives[alternative]["very_large_count"] > 0
    assert (
        alternatives["shape_higher"]["very_large_count"]
        > alternatives["shape_lower"]["very_large_count"]
    )
    assert record["total"]["modelled_large_count"] == row["modelled_large_count"]
    assert any(
        "0 of 3 answered cases are severity_unresolved" in line
        for line in setup.limitations
    )


def test_probability_of_at_least_one_is_computed_within_each_state():
    instrument = governed_contextual_instrument((LEAF, "fraud_organizations"))
    answers = {node.node_id: _answer(node, 2e-3) for node in instrument.nodes}
    _, record = _exhibit(instrument, lambda prompt: (answers[prompt.node.node_id],))
    rows = record["by_scenario"]
    marginals = [
        rows[key]["probability_at_least_one_large"]
        for key in (LEAF, "fraud_organizations")
    ]
    total = record["total"]["probability_at_least_one_large"]
    # Within a state the scenarios are independent; across states they share
    # the respondent's draw, so the total is not 1 - prod(1 - P_s) of the
    # marginals but lies between the largest marginal and their sum.
    assert max(marginals) < total < sum(marginals)
    assert record["total"]["modelled_large_count"] == pytest.approx(
        sum(rows[key]["modelled_large_count"] for key in rows), rel=1e-12
    )


def test_unresolved_cases_are_counted_with_their_eal_share():
    instrument = _instrument()
    contradicting = _answer(instrument.nodes[0], contradicting=True)
    setup, record = _exhibit(instrument, lambda prompt: (contradicting, contradicting))
    row = record["by_scenario"][LEAF]
    assert row["case_statuses"] == {"severity_unresolved": 1}
    assert row["severity_unresolved_gross_eal_share"] == pytest.approx(1)
    assert record["total"]["severity_unresolved_cases"] == 1
    # The provisional law is the p99-only tail.
    assert row["modelled_large_count"] == pytest.approx(
        row["extrapolated_large_count"], rel=1e-12
    )
    assert any(
        "3 of 3 answered cases are severity_unresolved" in line
        for line in setup.limitations
    )


class _Provider:
    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    def __init__(self, answer):
        self.answer = answer

    async def complete(self, **kwargs):
        return ProviderResponse(
            text=fixture_response(kwargs["user_prompt"], self.answer),
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=5,
            output_tokens=10,
        )


def test_public_study_simulates_both_anchor_alternatives_with_the_shared_transform(
    tmp_path, monkeypatch
):
    """Offline study with a 4e-3 pinning share exercises both odds-scaled
    alternatives and, with a very-large-loss share of 4e-5, the two-anchor
    tail."""
    bundle = write_source_bundle(tmp_path / "source")
    payload = governed_contextual_instrument((LEAF,)).model_dump(mode="json")
    payload["lab"].update({"entity_id": "lab_b", "status": "actual"})
    payload["lab"]["related_entity_ids"] = ["model"]
    payload["model"]["related_entity_ids"] = []
    payload["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        bundle
    ).manifest_sha256
    node = ContextualRiskInstrument.model_validate(payload).nodes[0]
    answer = _answer(node, 4e-3)
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
        with_access_bases(candidate["nodes"], candidate["capability_access"])
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
    # Record every odds-transform call with the large_loss_factor of the
    # _compile_cells call it happens in: 1.0 for the reference cells (whose
    # exhibit compiles the analytic alternatives) and the alternative factor for a simulated arm.
    import catmodel.setup.contextual_node as node_compiler
    import catmodel.setup.public_frontier as frontier

    outer: list[float | None] = []
    transform_calls: list[tuple[float | None, float, float]] = []
    transform, compile_cells = (
        node_compiler.odds_scaled_large_loss_share,
        frontier._compile_cells,
    )

    def recorded_transform(share, factor):
        transform_calls.append((outer[-1] if outer else None, share, factor))
        return transform(share, factor)

    def recorded_cells(*args, **kwargs):
        outer.append(args[3].large_loss_factor)
        try:
            return compile_cells(*args, **kwargs)
        finally:
            outer.pop()

    monkeypatch.setattr(
        node_compiler, "odds_scaled_large_loss_share", recorded_transform
    )
    monkeypatch.setattr(frontier, "_compile_cells", recorded_cells)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers",
        lambda *_args, **_kwargs: [_Provider(answer)],
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
        setup_id="anchor-alternatives",
        years_per_state=16,
        # 4,096 years for each of the 4 states, whatever the arm.
        max_reference_state_years=4 * 4096,
        max_comparison_state_years=4 * 4096,
        seed=3,
        comparison_families=(
            "large_loss_anchor",
            "large_loss_anchor_ignored",
            "very_large_loss_anchor",
        ),
        code_commit="fixture",
    )
    requirements = [
        row
        for row in package.execution.comparison_requirements
        if row.family_id == "large_loss_anchor"
    ]
    assert not [row for row in requirements if row.status == "unavailable"]
    arms = {row.alternative_id for row in package.alternatives}
    assert {
        "large_loss_anchor_ignored",
        "large_loss_anchor_lower",
        "large_loss_anchor_higher",
        "very_large_loss_anchor",
    } <= arms

    # The simulated arms call the shared transform with their own factor, on
    # the unscaled share; the reference cells' exhibit calls it with both.
    by_outer: dict[float | None, set[float]] = defaultdict(set)
    shares = set()
    for arm_factor, share, factor in transform_calls:
        by_outer[arm_factor].add(factor)
        shares.add(share)
    assert by_outer[1.0] == set(LARGE_LOSS_ANCHOR_FACTORS)
    for factor in LARGE_LOSS_ANCHOR_FACTORS:
        assert by_outer[factor] == {factor}
    assert shares and all(share == pytest.approx(4e-3, rel=1e-9) for share in shares)

    reference_states = {row.state_id: row for row in package.joint_states}
    for alternative_id, factor in (
        ("large_loss_anchor_lower", 1 / 3),
        ("large_loss_anchor_higher", 3.0),
    ):
        alternative = next(
            row for row in package.alternatives if row.alternative_id == alternative_id
        )
        overrides = package.overrides.of(alternative.alternative_id)
        assert overrides
        for override in overrides:
            reference_law = next(
                law
                for law in reference_states[override.state_id].laws
                if law.origin_id == override.law.origin_id
            )
            share = _shared_total_exceedance(reference_law)
            assert share is not None
            assert _shared_total_exceedance(override.law) == pytest.approx(
                odds_scaled_large_loss_share(share, factor), rel=1e-12
            )

    # The two-anchor arm keeps every first anchor and adds the second: the
    # stated very-large-loss share, 1 in 100 of the large-loss share.
    two_anchor = next(
        row
        for row in package.alternatives
        if row.alternative_id == "very_large_loss_anchor"
    )
    overrides = package.overrides.of(two_anchor.alternative_id)
    assert overrides
    for override in overrides:
        reference_law = next(
            law
            for law in reference_states[override.state_id].laws
            if law.origin_id == override.law.origin_id
        )
        assert _shared_total_exceedance(override.law) == (
            _shared_total_exceedance(reference_law)
        )
        assert _survival(override.law, 1e11) == pytest.approx(4e-5, rel=1e-9)
        assert _survival(reference_law, 1e11) < 4e-5 / 7
        second = cast(dict, override.law.conditioning_context["very_large_loss_anchor"])
        assert second["very_large_loss_role"] == "shapes_tail_above_large_loss_level"
        assert "very_large_loss_anchor" not in reference_law.conditioning_context

    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )

    def eal(arm, view="gross"):
        metric = available_metric(reduced_arm(report["arms"][arm])["metrics"][view])
        return metric["expected_annual_loss"]

    reference = eal("reference")
    assert (
        eal("large_loss_anchor_ignored")
        < eal("large_loss_anchor_lower")
        < reference
        < eal("large_loss_anchor_higher")
    )
    summary = report["sensitivity_summary"]
    ranked = json.dumps(summary["tornado"].get("rows", []))
    assert "large_loss_anchor_lower" in ranked
    assert "large_loss_anchor_ignored" not in ranked
    exhibit = reported_exhibit(present(summary.get("large_loss_anchor")))
    assert exhibit["by_scenario"][LEAF]["status"] == "available"
    total = exhibit["total"]
    assert total["status"] == "available"
    assert math.isfinite(present(total["modelled_large_count"]))
    # The simulated arm's analytic EAL is the exhibit's two_anchor EAL in every
    # view, the arm is ranked, and its simulated years do not contradict its
    # analytic mean.
    assert "very_large_loss_anchor" in ranked
    for view, alternatives in total["eal_alternatives"].items():
        assert alternatives["two_anchor"]["status"] == "available"
        assert eal("very_large_loss_anchor", view) == pytest.approx(
            alternatives["two_anchor"]["expected_annual_loss"], rel=1e-9
        )
        assert eal("reference", view) == pytest.approx(
            present(alternatives["expected_annual_loss"]), rel=1e-9
        )
    # A stated 1 in 100 is about seven times this class's shape-implied
    # share, which doubles gross EAL at this first share.
    assert eal("very_large_loss_anchor") / reference == pytest.approx(2.01, abs=0.02)
    metric = available_metric(
        reduced_arm(report["arms"]["very_large_loss_anchor"])["metrics"]["gross"]
    )
    assert present(metric["eal_reconciliation"])["status"] == "consistent"
    # The summary's scenario table carries the exhibit's share of expected
    # gross loss from incidents of at least USD 1 billion; with one leaf the
    # total's share is the leaf's.
    markdown, rows = results_summary(report)
    shares = {
        row["subject"]: row["value"]
        for row in rows
        if row["statistic"] == "gross_eal_share_at_or_above_large"
    }
    leaf = exhibit["by_scenario"][LEAF]
    assert leaf["status"] == "available"
    leaf_share = present(leaf["gross_eal_share_at_or_above_large"])
    assert 0 < leaf_share < 1
    assert shares == {LEAF: pytest.approx(leaf_share), None: pytest.approx(leaf_share)}
    assert "| Scenario | EAL | Share of EAL | Large-incident share | Note |" in markdown
    assert "incidents of at least 1B USD" in markdown


def test_gross_eal_share_matches_the_closed_form_tail_integral():
    """E[X; X >= A] = A pi + pi s_A / (1 - xi) for an uncapped anchored law,
    whose tail beyond A is a GPD with the first piece's scale s_A at A; the
    share divides it by E[X]."""
    from catmodel.probability.distributions import Distribution

    instrument = _instrument()
    answer = _answer(instrument.nodes[0], 2e-3)
    law = _law(instrument, answer)
    assert isinstance(law, IncidentLaw)
    severity = next(
        component.severity
        for component in law.segments[0].kernel.member_outcomes[0].components
        if component.severity is not None
    )
    total = native_severity_sampler(severity)
    assert isinstance(total, Distribution) and total.gpd_shape is not None
    assert total.severity_cap is None and total.large_loss_break is not None
    share = total.survival_probability(LEVEL)
    xi = total.gpd_shape
    above = LEVEL * share + share * total._large_loss_tail_scale / (1 - xi)
    _, record = _exhibit(instrument, lambda _prompt: (answer,))
    assert record["by_scenario"][LEAF][
        "gross_eal_share_at_or_above_large"
    ] == pytest.approx(above / total.analytic_mean(), rel=1e-9)


def test_probability_of_at_least_one_matches_the_within_state_closed_form():
    """P = sum over reference states of w (1 - prod over scenarios of
    E_U[exp(-count(U) x S(level))]), each scenario's rank independent; a
    product of marginals would differ."""
    instrument = governed_contextual_instrument((LEAF, "fraud_organizations"))
    answers = {node.node_id: _answer(node, 2e-3) for node in instrument.nodes}
    setup, record = _exhibit(instrument, lambda prompt: (answers[prompt.node.node_id],))
    expected = 0.0
    marginals: dict[str, float] = {}
    for state in setup.joint_states:
        if state.weight == 0:
            continue
        log_none = {
            law.scenario_id: _log_none(law, _survival(law))
            for law in state.laws
            if isinstance(law, IncidentLaw)
        }
        expected += state.weight * -math.expm1(sum(log_none.values()))
        for scenario_id, value in log_none.items():
            marginals[scenario_id] = marginals.get(scenario_id, 0.0) + (
                state.weight * -math.expm1(value)
            )
    # The exhibit integrates the rank on 4,096 midpoints, the check on 200,000.
    assert record["total"]["probability_at_least_one_large"] == pytest.approx(
        expected, rel=1e-4
    )
    for scenario_id, value in marginals.items():
        assert record["by_scenario"][scenario_id][
            "probability_at_least_one_large"
        ] == pytest.approx(value, rel=1e-4)


def test_probability_of_at_least_one_survives_an_underflowing_exponential():
    """With hundreds of thousands of expected large incidents a year at every
    rank, the probability of none underflows a double. The exhibit reports
    certainty; it does not fail the import."""
    instrument = _instrument()
    payload = _answer(instrument.nodes[0], 9e-3).model_dump(mode="json")
    for key in ("rate", "large_loss_count", "very_large_loss_count"):
        payload[key] = {name: value * 1e7 for name, value in payload[key].items()}
    heavy = ContextualRiskAnswer.model_validate(payload)
    _, record = _exhibit(instrument, lambda _prompt: (heavy,))
    assert record["total"]["probability_at_least_one_large"] == 1.0
    assert record["by_scenario"][LEAF]["probability_at_least_one_large"] == 1.0


def _log_none(law, survival):
    """log E_U[exp(-count(U) S)] by the midpoint rule on 200,000 ranks."""
    rank = law.parameter_rank
    ranks = (np.arange(200_000) + 0.5) / 200_000
    counts = epistemic_quantiles(
        rank.count, ranks, epistemic_tail_scale=rank.epistemic_tail_scale
    )
    return math.log(float(np.mean(np.exp(-survival * counts))))


def test_exhibit_eal_alternatives_are_analytic_and_feasible_near_the_bound():
    instrument = _instrument()
    _, record = _exhibit(instrument, lambda prompt: (_answer(prompt.node, 2e-3),))
    alternatives = record["by_scenario"][LEAF]["eal_alternatives"]["welfare"]
    assert (
        alternatives["anchor_ignored"]["relative_effect"]
        < alternatives["anchor_lower"]["relative_effect"]
        < 0
        < alternatives["anchor_higher"]["relative_effect"]
    )
    # The asymmetry quoted in assumptions.md (SEV-LARGE-LOSS-ANCHOR).
    assert alternatives["anchor_higher"]["relative_effect"] == pytest.approx(
        0.64, abs=0.01
    )
    assert alternatives["anchor_lower"]["relative_effect"] == pytest.approx(
        -0.37, abs=0.01
    )
    assert record["total"]["eal_alternatives"]["welfare"]["expected_annual_loss"] > 0
    assert "no VaR or TVaR" in record["interpretation"]
    serialized = json.dumps(record)
    assert "infeasible" not in serialized
    assert "partial" not in serialized

    for share in (4e-3, 9.9e-3):
        answer = _answer(instrument.nodes[0], share)
        _, record = _exhibit(instrument, lambda _prompt, answer=answer: (answer,))
        alternatives = record["by_scenario"][LEAF]["eal_alternatives"]["welfare"]
        assert alternatives["anchor_lower"]["status"] == "available"
        assert alternatives["anchor_higher"]["status"] == "available"
        assert alternatives["anchor_lower"]["relative_effect"] < 0
        assert alternatives["anchor_higher"]["relative_effect"] > 0
        serialized = json.dumps(record)
        assert "infeasible" not in serialized
        assert "partial" not in serialized
    # Near the bound the odds factors move the share little, and the law is
    # continuous in the share, so EAL moves little (quoted in assumptions.md).
    assert alternatives["anchor_higher"]["relative_effect"] == pytest.approx(
        0.006, abs=0.001
    )
    assert alternatives["anchor_lower"]["relative_effect"] == pytest.approx(
        -0.016, abs=0.001
    )


def test_the_exhibit_alternatives_use_the_shared_odds_transform(monkeypatch):
    """The exhibit's alternatives and the simulated family's arms (checked in the
    public-study test) pass through the same helper, once per alternative factor."""
    import catmodel.setup.contextual_node as node_compiler

    calls = []
    original = node_compiler.odds_scaled_large_loss_share

    def recorded(share, factor):
        calls.append((share, factor))
        return original(share, factor)

    monkeypatch.setattr(node_compiler, "odds_scaled_large_loss_share", recorded)
    instrument = _instrument()
    _exhibit(instrument, lambda prompt: (_answer(prompt.node, 2e-3),))
    assert {factor for _, factor in calls} == set(LARGE_LOSS_ANCHOR_FACTORS)
    assert all(share == pytest.approx(2e-3, rel=1e-12) for share, _ in calls)


def test_total_alternatives_are_available_and_sum_the_scenario_eals():
    instrument = governed_contextual_instrument((LEAF, "fraud_organizations"))
    shares = {LEAF: 4e-3, "fraud_organizations": 2e-3}
    _, record = _exhibit(
        instrument, lambda prompt: (_answer(prompt.node, shares[prompt.node.node_id]),)
    )
    per_scenario = {
        scenario: record["by_scenario"][scenario]["eal_alternatives"]["welfare"]
        for scenario in shares
    }
    total = record["total"]["eal_alternatives"]["welfare"]
    run_eal = total["expected_annual_loss"]
    for alternative in ("anchor_ignored", "anchor_lower", "anchor_higher"):
        assert total[alternative]["status"] == "available"
        scenario_eals = sum(
            per_scenario[scenario][alternative]["expected_annual_loss"]
            for scenario in shares
        )
        assert total[alternative]["expected_annual_loss"] == pytest.approx(
            scenario_eals, rel=1e-12
        )
        assert total[alternative]["relative_effect"] == pytest.approx(
            (scenario_eals - run_eal) / run_eal, rel=1e-12
        )


def test_each_frequency_arm_compiles_its_own_unit_law_without_tail_alternatives(
    monkeypatch,
):
    import catmodel.setup.contextual_setup as setup_compiler

    instrument = _instrument()
    calls = []
    original = setup_compiler._compile_admitted

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(setup_compiler, "_compile_admitted", counted)
    setup, record = _exhibit(instrument, lambda prompt: (_answer(prompt.node, 0.0),))
    # The direct, chain, before-crosscheck and before-hierarchy arms each
    # compile their own unit-count law; this instrument has no parents, so no
    # branch-implied arm.
    assert len(calls) == len(setup.joint_states) * len(instrument.nodes) + 4
    assert not any("large_loss_factor" in row for row in calls)
    assert record["by_scenario"][LEAF]["law_roles"] == {"no_anchor": 1}
    alternatives = record["by_scenario"][LEAF]["eal_alternatives"]["welfare"]
    assert alternatives["anchor_ignored"]["relative_effect"] == 0
    assert alternatives["anchor_lower"]["relative_effect"] == 0
    assert alternatives["anchor_higher"]["relative_effect"] == 0
    calls.clear()
    # A shaping anchor compiles its three alternatives once per reference state.
    setup, _ = _exhibit(instrument, lambda prompt: (_answer(prompt.node, 2e-3),))
    reference = sum(1 for state in setup.joint_states if state.weight > 0)
    tail_alternatives = [row for row in calls if "large_loss_factor" in row]
    assert len(tail_alternatives) == 3 * reference


def test_unresolved_and_shaping_cases_are_counted_by_law_role():
    instrument = _instrument()
    contradicting = _answer(instrument.nodes[0], contradicting=True)
    _, record = _exhibit(instrument, lambda prompt: (contradicting, contradicting))
    assert record["by_scenario"][LEAF]["law_roles"] == {
        "severity_unresolved_fallback": 1
    }
    _, record = _exhibit(instrument, lambda prompt: (_answer(prompt.node, 1e-7),))
    row = record["by_scenario"][LEAF]
    assert row["law_roles"] == {"shapes_tail_above_p99": 1}
    assert record["total"]["law_roles"] == {"shapes_tail_above_p99": 1}


def test_non_poisson_nodes_withhold_what_they_cannot_support():
    payload = _instrument().model_dump(mode="json")
    payload["nodes"][0].update(
        count_family="negative_binomial", count_variance_to_mean=2
    )
    instrument = ContextualRiskInstrument.model_validate(payload)

    def answer(prompt):
        return (_answer(prompt.node, 2e-3),)

    _, record = _exhibit(instrument, answer)
    row = record["by_scenario"][LEAF]
    assert row["probability_at_least_one_large"] is None
    assert "not Poisson" in row["probability_withheld_reasons"][0]
    assert row["modelled_large_count"] > 0


def test_a_shared_total_has_no_cost_outside_it():
    # A node's shared total is the whole estimand, so the question and the
    # large-loss shares never meet a component outside it.
    payload = _instrument().model_dump(mode="json")
    payload["nodes"][0]["components"].append(
        {
            "component_id": "additive_cost",
            "definition": "A synthetic cost outside the shared total",
            "nature": "resource_cost",
            "bearer": "third_party",
            "rank_group_id": "event_severity",
            "gpd_shape": 0.7,
        }
    )
    with pytest.raises(ValueError, match="shared total is the whole estimand"):
        ContextualRiskInstrument.model_validate(payload)
