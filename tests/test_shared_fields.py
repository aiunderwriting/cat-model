"""Frozen contracts share their fields set between instances (``SharedFieldsSet``)."""

import pickle

import pytest
from pydantic import ConfigDict, ValidationError

from catmodel.contracts.components import ComponentAmountLaw
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import (
    Alternative,
    IncidentLaw,
    IncidentSegment,
    JointState,
    LawOverride,
    MarkKernel,
)
from catmodel.contracts.native_severity import (
    NativeSeverityComponent,
    NativeSeverityLaw,
)
from catmodel.contracts.shared_fields import SharedFieldsSet
from catmodel.elicitation.question_flow import QUESTION_KINDS
from catmodel.frozen_calls import parser_protocol


class _Pair(SharedFieldsSet):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left: int
    right: int = 0


def _fields(model: SharedFieldsSet) -> set[str]:
    return model.__pydantic_fields_set__


def test_instances_given_the_same_fields_share_one_set():
    first, second = _Pair(left=1, right=2), _Pair(left=3, right=4)
    assert _fields(first) is _fields(second)
    assert first.model_fields_set == {"left", "right"}


def test_instances_given_different_fields_keep_what_each_was_given():
    given, defaulted = _Pair(left=1, right=0), _Pair(left=1)
    assert given == defaulted
    assert given.model_fields_set == {"left", "right"}
    assert defaulted.model_fields_set == {"left"}
    assert given.model_dump(exclude_unset=True) == {"left": 1, "right": 0}
    assert defaulted.model_dump(exclude_unset=True) == {"left": 1}
    assert _fields(defaulted) is _fields(_Pair(left=9))
    assert _fields(defaulted) is not _fields(given)


def test_validation_construction_and_revalidation_share_the_set():
    direct = _Pair(left=1, right=2)
    validated = _Pair.model_validate({"left": 5, "right": 6})
    constructed = _Pair.model_construct(left=7, right=8)
    again = _Pair.model_validate(direct.model_dump())
    assert again == direct
    for other in (validated, constructed, again):
        assert _fields(other) is _fields(direct)


def test_a_copy_with_an_update_leaves_the_shared_set_unchanged():
    original = _Pair(left=1)
    copy = original.model_copy(update={"right": 5})
    assert copy.model_fields_set == {"left", "right"}
    assert original.model_fields_set == {"left"}
    assert _Pair(left=2).model_fields_set == {"left"}


def test_assignment_is_refused_so_a_shared_set_cannot_change():
    model = _Pair(left=1)
    with pytest.raises(ValidationError):
        model.right = 3  # ty: ignore[invalid-assignment]
    assert model.model_fields_set == {"left"}
    assert _Pair(left=2).model_fields_set == {"left"}


def test_pickling_keeps_values_and_given_fields():
    first, second = pickle.loads(pickle.dumps((_Pair(left=1), _Pair(left=2, right=3))))
    assert (first, second) == (_Pair(left=1), _Pair(left=2, right=3))
    assert first.model_fields_set == {"left"}
    assert second.model_fields_set == {"left", "right"}


def test_a_subclass_that_is_not_frozen_is_refused():
    with pytest.raises(TypeError, match="must be frozen"):

        class _Loose(SharedFieldsSet):
            model_config = ConfigDict(frozen=False)

            value: int


@pytest.mark.parametrize(
    "contract",
    [
        Alternative,
        ComponentAmountLaw,
        IncidentLaw,
        IncidentSegment,
        JointState,
        LawOverride,
        MarkKernel,
        NativeSeverityComponent,
        NativeSeverityLaw,
    ],
)
def test_the_contracts_a_setup_holds_most_of_share_their_sets(contract):
    """A loaded setup is mostly these classes; one that stopped sharing would
    give back the memory this saves without failing anything else."""
    assert issubclass(contract, SharedFieldsSet)
    assert contract.model_config["frozen"] is True


def test_no_question_kind_reaches_a_shared_fields_set():
    """A response model's base class is part of the parser identity every
    stored call records, so an answer contract that inherited the shared set
    would give later calls another identity for the same parsing."""
    for kind in QUESTION_KINDS:
        protocol = parser_protocol(kind.parser, kind.response_model)
        assert "catmodel.contracts.shared_fields" not in protocol.source_definitions
    assert not issubclass(Quantiles, SharedFieldsSet)
