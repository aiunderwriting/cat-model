"""A setup derived with other joint states is the setup construction builds:
the same inputs, the same refusals, and only the new states validated again.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic import ValidationError

from catmodel.contracts import run_setup
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.contracts.incident import (
    IncidentLaw,
    JointState,
    LawOverride,
    PopulationAbsence,
)
from catmodel.contracts.law_overrides import HeldOverrides
from catmodel.contracts.run_setup import RunSetup
from catmodel.setup.contextual_node import (
    _ADMISSION_PROBES,
    admit_node,
    compile_contextual_node,
    constant_path,
)
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    governed_contextual_answer,
    governed_contextual_instrument,
)
from test_support.run_setup import scientific_digest, setup_from_payload
from test_support.study import study_payload


def _setup() -> RunSetup:
    return setup_from_payload(study_payload())


def _states(setup: RunSetup, **changes: object) -> tuple[JointState, ...]:
    payload = setup.joint_states[0].model_dump(mode="json")
    payload["laws"][0].update(changes)
    return (JointState.model_validate(payload),)


def _refusal(build) -> str:
    with pytest.raises(ValidationError) as refused:
        build()
    return str(refused.value)


def test_a_derived_setup_is_the_one_construction_builds() -> None:
    setup = _setup()
    states = _states(setup, reference_annual_rate=3)
    derived = setup.with_joint_states(states)
    built = replace(setup, joint_states=states)
    assert derived == built
    assert scientific_digest(derived) == scientific_digest(built)
    assert derived.joint_states[0].laws[0].reference_annual_rate == 3
    # The setup it was derived from is unchanged.
    assert setup.joint_states[0].laws[0].reference_annual_rate == 1


def test_only_the_new_states_are_validated_again(monkeypatch) -> None:
    setup = _setup()
    states = _states(setup, reference_annual_rate=3)
    sections = []
    revalidated = run_setup._revalidated

    def counted(model, value):
        sections.append(model.__name__)
        return revalidated(model, value)

    monkeypatch.setattr(run_setup, "_revalidated", counted)
    derived = setup.with_joint_states(states)
    assert sections == []
    # The other inputs are the first setup's own objects, and the states are
    # validated copies, not the ones given.
    assert derived.scope is setup.scope
    assert derived.execution is setup.execution
    assert derived.evidence is setup.evidence
    assert derived.limitations is setup.limitations
    assert derived.joint_states == states
    assert derived.joint_states[0] is not states[0]
    replace(setup, joint_states=states)
    assert sections == ["RunScope", "AssumptionSet", "ExecutionPlan"]


def test_a_derived_setup_refuses_what_construction_refuses() -> None:
    setup = _setup()
    cases = {
        # A state that fails its own validation, as an unvalidated copy.
        "copied": (setup.joint_states[0].model_copy(update={"weight": -1.0}),),
        # A law outside the setup's rosters: a check across inputs.
        "roster": _states(setup, scenario_id="absent"),
        # No state at all.
        "empty": (),
    }
    for label, states in cases.items():
        refused = _refusal(lambda states=states: setup.with_joint_states(states))
        assert refused == _refusal(
            lambda states=states: replace(setup, joint_states=states)
        ), label


def test_overrides_are_checked_against_the_new_states() -> None:
    payload = study_payload()
    setup = setup_from_payload(payload)
    law = setup.joint_states[0].laws[0]
    changed = law.model_copy(update={"reference_annual_rate": 2.0})
    with_arm = deepcopy(payload)
    with_arm["alternatives"] = [
        {
            "alternative_id": "arm",
            "purpose": "scientific_alternative",
            "description": "A doubled count.",
        }
    ]
    held = replace(
        setup_from_payload(with_arm),
        overrides=HeldOverrides({"arm": (LawOverride(state_id="state", law=changed),)}),
    )
    renamed = setup.joint_states[0].model_dump(mode="json")
    renamed["state_id"] = "other"
    states = (JointState.model_validate(renamed),)
    # The arm's override names a state the new states do not hold.
    refused = _refusal(lambda: held.with_joint_states(states))
    assert refused == _refusal(lambda: replace(held, joint_states=states))
    # With the same states the arm stands, checked against them.
    kept = held.with_joint_states(held.joint_states)
    assert kept == held
    assert kept.overrides.of("arm") == held.overrides.of("arm")


def test_admitting_laws_at_one_node_builds_its_setup_once(monkeypatch) -> None:
    instrument = governed_contextual_instrument()
    node = instrument.nodes[0]
    score = instrument.anchor_scores[0]
    assert node.population.threshold is not None
    law = compile_contextual_node(
        instrument, node, {score: governed_contextual_answer(node)}, score
    )
    assert isinstance(law, IncidentLaw)
    built = []
    init = RunSetup.__post_init__

    def counted(self) -> None:
        built.append(self.setup_id)
        init(self)

    monkeypatch.setattr(RunSetup, "__post_init__", counted)
    _ADMISSION_PROBES.clear()
    path = constant_path(instrument, score)
    first = admit_node(instrument, node, path, law)
    again = admit_node(instrument, node, path, law)
    # The law comes back as the setup validated it: equal, and its own object.
    assert first == law and again == law
    assert first is not law
    assert built == ["contextual-admission"]
    # Another node object, as a comparison arm's edit makes, has its own.
    edited = node.model_copy()
    assert admit_node(instrument, edited, path, law) == law
    assert built == ["contextual-admission"] * 2


def test_admission_validates_the_law_it_is_given() -> None:
    """A law may reach admission as an unvalidated copy, as conditioning a
    compiled law on its trajectory makes. It comes back validated, or the
    copy's fault is refused, with or without a threshold at its node."""
    for instrument in (governed_contextual_instrument(), contextual_instrument()):
        node = instrument.nodes[0]
        score = instrument.anchor_scores[0]
        answer = (
            governed_contextual_answer(node)
            if node.population.threshold is not None
            else contextual_answer()
        )
        law = compile_contextual_node(instrument, node, {score: answer}, score)
        assert isinstance(law, IncidentLaw)
        path = constant_path(instrument, score)
        context = {"note": ["kept", "as", "written"]}
        copied = law.model_copy(update={"conditioning_context": context})
        admitted = admit_node(instrument, node, path, copied)
        assert isinstance(admitted, IncidentLaw)
        # Validation froze the open payload the copy held as a plain dict.
        assert admitted.conditioning_context == {"note": ("kept", "as", "written")}
        assert admitted == IncidentLaw.model_validate(copied.model_dump(mode="json"))
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            admit_node(
                instrument,
                node,
                path,
                law.model_copy(update={"reference_annual_rate": -1.0}),
            )


def test_a_law_refused_at_a_node_is_refused_from_the_kept_setup_too() -> None:
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["population"].update(threshold=200, threshold_comparator="gt")
    instrument = ContextualRiskInstrument.model_validate(payload)
    node = instrument.nodes[0]
    law = compile_contextual_node(instrument, node, {0.5: contextual_answer()}, 0.5)
    assert isinstance(law, IncidentLaw)
    _ADMISSION_PROBES.clear()
    path = constant_path(instrument, 0.5)
    refusals = [admit_node(instrument, node, path, law) for _ in range(2)]
    assert len(_ADMISSION_PROBES) == 1
    assert refusals[0] == refusals[1]
    assert isinstance(refusals[0], PopulationAbsence)
    assert refusals[0].disposition == "unavailable"
    assert "contradicts qualification" in refusals[0].reason
