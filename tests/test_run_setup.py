"""The public setup composes inputs without changing their scientific identity."""

import copy
import dataclasses
import pickle
import typing
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, TypeAliasType

import pytest
from pydantic import BaseModel, ValidationError

from catmodel.contracts import incident
from catmodel.contracts._setup_codec import (
    INLINE_FIELDS,
    decode_scientific,
    encode_scientific,
)
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.primitives import Frozen, FrozenDict
from catmodel.contracts.run_setup import (
    AssumptionSet,
    EvidenceSet,
    RunScope,
    RunSetup,
)
from catmodel.hashing import canonical_json, canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.resources import config_path
from catmodel.simulation.incident import compile_incident_state
from test_support.run_setup import (
    mutable_containers,
    scientific_digest,
    setup_from_json,
    with_execution,
)
from test_support.study import (
    _financial_study,
    _native_payment_study,
    _study,
    stipulated_study,
)


def _execution():
    return ExecutionPlan(
        plan_id="composition-check",
        required_arms=("reference",),
        requested_views=("gross",),
        simulations=8,
        seed=19,
    )


@pytest.fixture
def setup():
    return with_execution(
        stipulated_study(), _execution(), setup_id="separate-package-id"
    )


@pytest.mark.parametrize(
    "producer",
    [
        stipulated_study,
        _study,
        _financial_study,
        lambda: _native_payment_study(mean=0, recovery_rate=0),
        lambda: _native_payment_study(mean=0.4, recovery_rate=0.6),
        lambda: setup_from_json(
            config_path("setups", "stipulated-campaign", "setup.json").read_bytes()
        ),
    ],
    ids=[
        "stipulated",
        "multi-state",
        "financial",
        "zero-payment",
        "payment",
        "campaign",
    ],
)
def test_codec_preserves_fields_order_and_scientific_identity(producer):
    original = producer()
    composed = with_execution(original, _execution(), setup_id="independent-package-id")
    reconstructed = decode_scientific(
        encode_scientific(composed),
        composed.execution,
        setup_id=composed.setup_id,
        evidence_files=composed.evidence.files,
    )
    assert reconstructed == composed
    assert list(encode_scientific(reconstructed)) == list(encode_scientific(original))
    assert canonical_json(encode_scientific(reconstructed)) == canonical_json(
        encode_scientific(original)
    )
    assert reconstructed.scientific_sha256 == original.scientific_sha256


def test_public_setup_keeps_executable_inputs_beside_standalone_assumptions(setup):
    assert isinstance(setup, RunSetup)
    assert setup.scope.scope_id == "stipulated_fixture"
    assert setup.setup_id == "separate-package-id"
    assert {row.name for row in fields(setup)} >= {
        "scope",
        "evidence",
        "assumptions",
        "execution",
    }
    assert setup.assumptions.independence == stipulated_study().assumptions.independence
    assert setup.joint_states[0].laws == stipulated_study().joint_states[0].laws
    # Laws, weights, contracts and bindings are executable inputs on the setup,
    # whatever their origin; only standalone statements are assumptions.
    assert set(AssumptionSet.model_fields) == {"independence"}
    assert {row.name for row in fields(setup)} >= {
        "joint_states",
        "alternatives",
        "conditional_contexts",
        "contracts",
    }


def test_ordered_origin_records_keep_stipulations_with_their_original_bytes():
    source = config_path("setups", "stipulated-campaign")
    record = setup_from_json((source / "setup.json").read_bytes())
    original = record.evidence.records[0]
    second = original.model_copy(update={"evidence_id": "additional_fixture_origin"})
    record = replace(record, evidence=EvidenceSet(records=(second, original)))
    retained = {
        "assumptions.json": (source / "evidence" / "assumptions.json").read_bytes()
    }
    setup = with_execution(
        record, _execution(), setup_id="origin-order", evidence_files=retained
    )
    assert setup.evidence.records == (second, original)
    assert setup.evidence.files == retained
    assert all(row.origin == "stipulation" for row in setup.evidence.records)
    assert setup.assumptions.independence == record.assumptions.independence
    assert canonical_json(encode_scientific(setup)) == canonical_json(
        encode_scientific(record)
    )


def test_judgment_assumptions_do_not_require_evidence(setup):
    judgment = replace(
        setup,
        evidence=EvidenceSet(),
        assumptions=setup.assumptions.model_copy(
            update={
                "independence": (
                    "Conditional origin independence is an authored judgment.",
                )
            }
        ),
    )
    assert not judgment.evidence.records and not judgment.evidence.files
    assert compile_incident_state(judgment, "state").incidents[0].annual_rate == 1


def test_construction_rejects_invalid_scope_and_unrepresentable_metadata(setup):
    invalid_scope = setup.scope.model_copy(
        update={"period_end": setup.scope.period_start}
    )
    with pytest.raises(ValueError, match="exact one-year"):
        replace(setup, scope=invalid_scope)
    with pytest.raises(TypeError, match="expected AssumptionSet, got dict"):
        replace(
            setup,
            assumptions={"independence": (), "derived_estimates": {"rate": 1}},
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        replace(setup, annotations={"silently_discarded": True})


def test_execution_seed_and_budget_do_not_change_scientific_laws(setup):
    changed = replace(
        setup,
        execution=setup.execution.model_copy(
            update={
                "seed": 2027,
                "simulations": 100,
                "numerical_controls": {"max_campaign_members_per_year": 10},
            }
        ),
    )
    assert changed.joint_states == setup.joint_states
    assert changed.execution_plan_sha256 != setup.execution_plan_sha256
    assert changed.scientific_sha256 == setup.scientific_sha256
    original = compile_incident_state(setup, "state")
    alternate = compile_incident_state(changed, "state")
    assert alternate.law_sha256 == original.law_sha256
    assert alternate.conditional_key == original.conditional_key


def test_saved_location_and_manifest_identity_do_not_change_scientific_laws(
    setup, tmp_path
):
    from catmodel.setup.store import build_setup

    first = build_setup(
        destination=tmp_path / "one", setup=setup, code_commit="fixture-one"
    )
    second = build_setup(
        destination=tmp_path / "two",
        setup=replace(setup, setup_id="another-package-id"),
        code_commit="fixture-two",
    )
    assert first.root != second.root
    assert first.manifest_sha256 != second.manifest_sha256
    assert (
        first.scientific_sha256 == second.scientific_sha256 == setup.scientific_sha256
    )
    first_law = compile_incident_state(first, "state")
    second_law = compile_incident_state(second, "state")
    assert first_law.conditional_key == second_law.conditional_key
    assert first_law.law_sha256 == second_law.law_sha256


def test_derived_copy_drops_saved_metadata(setup, tmp_path):
    from catmodel.setup.store import build_setup

    saved = build_setup(
        destination=tmp_path / "saved", setup=setup, code_commit="fixture"
    )
    derived = replace(saved, execution=saved.execution.model_copy(update={"seed": 29}))
    assert saved.root == tmp_path / "saved"
    assert derived._package is None
    with pytest.raises(ValueError, match="no saved package"):
        _ = derived.root


def _with_nested_context(setup):
    """The setup with free-form records that nest a dict and a list."""
    state = setup.joint_states[0]
    law = state.laws[0].model_copy(
        update={"conditioning_context": {"source": {"rows": [1, {"cell": 2}]}}}
    )
    return replace(
        setup,
        joint_states=(state.model_copy(update={"laws": (law,)}),),
        execution=setup.execution.model_copy(
            update={"numerical_controls": {"tail_gate": {"views": ["gross"]}}}
        ),
    )


# Each changes a saved input in place: a typed or free-form mapping at any
# depth, a sequence inside a free-form record, a model field or a setup field.
MUTATIONS = {
    "typed-mapping": lambda s: s.joint_states[0].physical_factor_state.update(f=1),
    "free-form": lambda s: (
        s.joint_states[0].laws[0].conditioning_context.update(changed=True)
    ),
    "nested-free-form": lambda s: (
        s.joint_states[0].laws[0].conditioning_context["source"].__setitem__("rows", [])
    ),
    "dict-in-list": lambda s: (
        s.joint_states[0].laws[0].conditioning_context["source"]["rows"][1].pop("cell")
    ),
    "list-append": lambda s: (
        s.joint_states[0].laws[0].conditioning_context["source"]["rows"].append(3)
    ),
    "execution": lambda s: s.execution.numerical_controls["tail_gate"].clear(),
    "model-field": lambda s: setattr(s.joint_states[0], "weight", 0.5),
    "setup-field": lambda s: setattr(s, "limitations", ()),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_saved_inputs_cannot_change_after_package_attachment(setup, tmp_path, mutation):
    """Nothing reachable from a saved setup can change in place, so its cached
    scientific digest and attached files always describe its current inputs."""
    from catmodel.setup.store import build_setup, load_setup

    saved = build_setup(
        destination=tmp_path / "saved",
        setup=_with_nested_context(setup),
        code_commit="fixture",
    )
    digest, files = saved.scientific_sha256, dict(saved.files)
    inline = canonical_json_sha256(encode_scientific(saved))
    with pytest.raises(
        (TypeError, AttributeError, ValidationError, FrozenInstanceError)
    ):
        MUTATIONS[mutation](saved)
    assert canonical_json_sha256(encode_scientific(saved)) == inline
    assert saved.scientific_sha256 == digest == scientific_digest(saved)
    assert saved.files == files
    assert load_setup(saved.root) == saved


def test_every_model_and_mapping_a_setup_holds_is_frozen():
    """Every model and dataclass a setup can hold is frozen, each mapping field
    freezes its value at any depth when validated, and no field defaults to a
    value that can change."""

    def unfrozen(annotation, where, seen):
        if isinstance(annotation, TypeAliasType):
            return unfrozen(annotation.__value__, where, seen)
        origin = typing.get_origin(annotation)
        if origin is Frozen:
            return []
        if origin in (dict, list, set) or annotation in (dict, list, set, object, Any):
            return [where]
        if not isinstance(annotation, type) or annotation in seen:
            return [
                row
                for arg in typing.get_args(annotation)
                for row in unfrozen(arg, where, seen)
            ]
        found = []
        if issubclass(annotation, BaseModel):
            seen.add(annotation)
            if not annotation.model_config.get("frozen"):
                found.append(f"{annotation.__name__} is not frozen")
            for name, info in annotation.model_fields.items():
                path = f"{annotation.__name__}.{name}"
                default = info.get_default(call_default_factory=True)
                found += mutable_containers(default, path + " default")
                found += unfrozen(info.annotation, path, seen)
        elif dataclasses.is_dataclass(annotation):
            seen.add(annotation)
            if not annotation.__dataclass_params__.frozen:
                found.append(f"{annotation.__name__} is not frozen")
            hints = typing.get_type_hints(annotation, include_extras=True)
            for row in dataclasses.fields(annotation):
                if row.init:
                    path = f"{annotation.__name__}.{row.name}"
                    found += unfrozen(hints[row.name], path, seen)
        return found

    seen = set()
    assert unfrozen(RunSetup, "RunSetup", seen) == []
    assert {
        incident.Evidence,
        incident.IncidentLaw,
        incident.ExecutionPlan,
    } <= seen
    # An arm's plan is copied without validation, so it freezes its own controls.
    plan = ExecutionPlan.model_validate(
        {
            **_execution().model_dump(),
            "numerical_controls": {"tail_gate": {"levels": [0.999]}},
        }
    )
    assert mutable_containers(plan.for_arm("reference")) == []


def test_frozen_values_survive_pickle_and_copies():
    plan = ExecutionPlan.model_validate(
        {
            **_execution().model_dump(),
            "numerical_controls": {"tail_gate": {"levels": [0.999]}},
        }
    )
    copies = [
        pickle.loads(pickle.dumps(plan, protocol))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    ]
    copies += [copy.copy(plan), copy.deepcopy(plan), plan.model_copy(deep=True)]
    for value in copies:
        assert value == plan
        controls = value.numerical_controls
        assert type(controls) is FrozenDict
        assert type(controls["tail_gate"]) is FrozenDict
        assert controls["tail_gate"]["levels"] == (0.999,)


def test_an_unvalidated_copy_arrives_frozen_and_detached(setup):
    """A law copied without validation is frozen when placed in a setup, and
    later changes to the caller's dict do not reach the setup."""
    rows = [1]
    context: dict[str, object] = {"k": rows}
    state = setup.joint_states[0]
    law = state.laws[0].model_copy(update={"conditioning_context": context})
    changed = replace(setup, joint_states=(state.model_copy(update={"laws": (law,)}),))
    held = changed.joint_states[0].laws[0].conditioning_context
    assert type(held) is FrozenDict and held == {"k": (1,)}
    digest = changed.scientific_sha256
    rows.append(2)
    context["other"] = 3
    assert changed.joint_states[0].laws[0].conditioning_context == {"k": (1,)}
    assert scientific_digest(changed) == digest


def test_construction_revalidates_nested_resolved_input_types(setup):
    invalid_state = setup.joint_states[0].model_copy(update={"weight": "invalid"})
    with pytest.raises(ValidationError, match="weight"):
        replace(setup, joint_states=(invalid_state,))


def test_changing_a_resolved_incident_law_changes_scientific_identity(setup):
    original_state = setup.joint_states[0]
    original_law = original_state.laws[0]
    changed_law = original_law.model_copy(
        update={"reference_annual_rate": 2 * original_law.reference_annual_rate}
    )
    changed_state = original_state.model_copy(update={"laws": (changed_law,)})
    changed = replace(
        setup,
        joint_states=(changed_state,),
    )
    assert changed.execution_plan_sha256 == setup.execution_plan_sha256
    assert changed.scientific_sha256 != setup.scientific_sha256
    assert (
        compile_incident_state(changed, "state").incidents[0].annual_rate
        == 2 * compile_incident_state(setup, "state").incidents[0].annual_rate
    )


def _saved_example(name: str) -> RunSetup:
    root = config_path("setups", name)
    return decode_scientific(
        strict_json_loads((root / "setup.json").read_bytes()),
        ExecutionPlan.model_validate_json((root / "plan.json").read_bytes()),
        setup_id=name,
    )


# Known identities of the shipped example setups' inline encoding, less its
# disclosure prose. A change here changes the identity of every saved setup,
# capture and result built on it.
@pytest.mark.parametrize(
    ("name", "digest"),
    [
        (
            "stipulated-campaign",
            "5b84e18467a2504321ce054a95d2d8038c353b4a97a718a0d8f5e86ae80bc58a",
        ),
        (
            "stipulated-conditional",
            "4677b9b13e9fb2ddf0d60cd2dc29ff529035d314b822661ec70ea4d646a7bea3",
        ),
    ],
)
def test_saved_example_setups_keep_their_scientific_identity(name, digest):
    assert _saved_example(name).scientific_sha256 == digest


def test_one_field_table_stores_every_setup_field():
    # The overrides have no key of their own: each alternative lists its own.
    stored = {row.name for row in fields(RunSetup) if row.init} - {
        "setup_id",
        "execution",
        "scope",
        "assumptions",
        "overrides",
    }
    assert set(INLINE_FIELDS.values()) == {
        *(("scope", name) for name in RunScope.model_fields),
        *(("assumptions", name) for name in AssumptionSet.model_fields),
        *((name, None) for name in stored),
    }
    assert len(set(INLINE_FIELDS.values())) == len(INLINE_FIELDS)
    # The conditional example has conditional contexts, so no key is omitted.
    setup = _saved_example("stipulated-conditional")
    payload = encode_scientific(setup)
    assert set(payload) == set(INLINE_FIELDS)
    assert (
        encode_scientific(
            decode_scientific(payload, setup.execution, setup_id=setup.setup_id),
        )
        == payload
    )
    with pytest.raises(ValueError, match="unexpected scientific setup fields"):
        decode_scientific(
            {**payload, "unmodelled_input": 1},
            setup.execution,
            setup_id=setup.setup_id,
        )
    for required in ("joint_states", "independence_assumptions", "limitations"):
        with pytest.raises(ValueError, match="requires fields"):
            decode_scientific(
                {key: value for key, value in payload.items() if key != required},
                setup.execution,
                setup_id=setup.setup_id,
            )
