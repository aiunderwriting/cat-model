"""A record's encoding check covers what it covered when every record encoded
all it held: validation accepts exactly the records whose whole encoding
succeeds, and encodes each value once when a record is validated from its
fields.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Annotated, Any

import pytest
from pydantic import AfterValidator, BaseModel, Field, ValidationError

from catmodel.contracts import incident
from catmodel.contracts.incident import (
    IncidentLaw,
    IncidentSegment,
    JointState,
    Record,
    _only_records,
    _record_fields,
)
from catmodel.contracts.primitives import Frozen, FrozenDict
from catmodel.setup.contextual_node import (
    compile_contextual_node,
    constant_path,
)
from test_support.contextual_risk import contextual_answer, contextual_instrument

type Path = tuple[str | int, ...]


@pytest.fixture(scope="module")
def state() -> JointState:
    """One state holding a compiled fixture law several records deep."""
    instrument = contextual_instrument()
    law = compile_contextual_node(
        instrument, instrument.nodes[0], {0.5: contextual_answer()}, 0.5
    )
    assert isinstance(law, IncidentLaw)
    return JointState(
        state_id="state",
        epistemic_group_id="state",
        weight=1,
        entity_paths=(constant_path(instrument, 0.5),),
        laws=(law,),
    )


def _leaves(value: object, path: Path = ()) -> Iterator[tuple[Path, object]]:
    """Every value that is not a model, mapping or sequence, with its path."""
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _leaves(getattr(value, name), (*path, name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def _at(value: Any, path: Path) -> Any:
    for part in path:
        value = (
            getattr(value, str(part)) if isinstance(value, BaseModel) else value[part]
        )
    return value


def _copied(value: Any, path: Path, new: object) -> Any:
    """``value`` with ``new`` at ``path``, every model, mapping and tuple on
    the way copied without validation."""
    if not path:
        return new
    head, *rest = path
    if isinstance(value, BaseModel):
        assert isinstance(head, str)
        return value.model_copy(
            update={head: _copied(getattr(value, head), tuple(rest), new)}
        )
    if isinstance(value, Mapping):
        return FrozenDict({**value, head: _copied(value[head], tuple(rest), new)})
    return tuple(
        _copied(item, tuple(rest), new) if index == head else item
        for index, item in enumerate(value)
    )


def _fields(value: Any, path: Path, new: object) -> Any:
    """``value`` as plain fields with ``new`` at ``path``: models on the way
    become dictionaries, so validation builds each of them again, and models
    off the way stay instances."""
    if not path:
        return new
    head, *rest = path
    if isinstance(value, BaseModel):
        return {
            name: _fields(getattr(value, name), tuple(rest), new)
            if name == head
            else getattr(value, name)
            for name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {**value, head: _fields(value[head], tuple(rest), new)}
    return tuple(
        _fields(item, tuple(rest), new) if index == head else item
        for index, item in enumerate(value)
    )


def _whole_encoding_succeeds(record: Record) -> bool:
    try:
        json.dumps(record.model_dump(mode="json"), allow_nan=False)
    except ValueError:
        return False
    return True


def _validated(model: type[Record], value: object) -> Record | None:
    """``value`` validated; None where it is refused, or a validator cannot
    read an unvalidated copy's field."""
    try:
        return model.model_validate(value)
    except (ValidationError, TypeError):
        return None


def _validated_unchecked(
    monkeypatch: pytest.MonkeyPatch, model: type[Record], value: object
) -> Record | None:
    """``value`` validated with the encoding check switched off."""
    with monkeypatch.context() as patch:
        patch.setattr(incident.json, "dumps", lambda *_args, **_kwargs: "")
        return _validated(model, value)


def _model_depths(value: Any, path: Path) -> list[int]:
    """The lengths of ``path``'s prefixes that end at a model."""
    depths = []
    for depth in range(len(path)):
        if isinstance(_at(value, path[:depth]), BaseModel):
            depths.append(depth)
    return depths


def _cases(state: JointState) -> Iterator[tuple[str, object]]:
    """Inputs with a value the encoding refuses, or one it turns to null, in
    turn at one place of each kind, given as fields down to every depth and
    as an unvalidated copy from there in."""
    seen = set()
    for path, leaf in _leaves(state):
        kind = tuple("*" if isinstance(part, int) else part for part in path)
        if kind in seen or isinstance(leaf, bool) or not isinstance(leaf, float | int):
            continue
        seen.add(kind)
        for bad in (math.nan, math.inf, object()):
            for depth in _model_depths(state, path):
                copy = _copied(_at(state, path[:depth]), path[depth:], bad)
                yield (
                    f"{'.'.join(map(str, kind))} {bad!r:.8} copied from depth {depth}",
                    copy if depth == 0 else _fields(state, path[:depth], copy),
                )
            yield (
                f"{'.'.join(map(str, kind))} {bad!r:.8} as fields",
                _fields(state, path, bad),
            )


# Serializing an unvalidated copy whose field holds another type warns; the
# suite turns warnings into errors, and here the encoding's own verdict counts.
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_validation_accepts_exactly_what_encodes_whole(
    state: JointState, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = refused = 0
    for label, value in _cases(state):
        unchecked = _validated_unchecked(monkeypatch, JointState, value)
        checked = _validated(JointState, value)
        if unchecked is None:
            # The fields refuse it, with or without the check.
            assert checked is None, label
        elif _whole_encoding_succeeds(unchecked):
            assert checked is not None, label
            accepted += 1
        else:
            assert checked is None, label
            refused += 1
    # Both happen: an unvalidated copy with a number that is not finite is
    # refused, and an open payload's is encoded as null and accepted.
    assert refused > 100
    assert accepted > 10


def test_a_record_validated_from_its_fields_encodes_each_value_once(
    state: JointState, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded: list[object] = []
    dumps = json.dumps

    def recording(value: object, *, allow_nan: bool) -> str:
        encoded.append(value)
        return dumps(value, allow_nan=allow_nan)

    def count(value: object) -> int:
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, list):
            return sum(count(item) for item in value)
        return 1

    payload = state.model_dump(mode="python")
    monkeypatch.setattr(incident.json, "dumps", recording)
    JointState.model_validate(payload)
    whole = count(state.model_dump(mode="json"))
    assert sum(count(row) for row in encoded) == whole
    # Given as an instance, a record is encoded whole: it holds four levels of
    # records, and each used to encode the law's components again.
    encoded.clear()
    JointState.model_validate(state)
    assert [count(row) for row in encoded] == [whole]


def test_a_record_built_around_instances_encodes_each_whole(
    state: JointState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record given as an instance may be an unvalidated copy, so a record
    built around it encodes all of it; only the new record's own fields are
    not encoded twice."""
    encoded: list[object] = []
    dumps = json.dumps

    def recording(value: object, *, allow_nan: bool) -> str:
        encoded.append(value)
        return dumps(value, allow_nan=allow_nan)

    law = state.laws[0]
    monkeypatch.setattr(incident.json, "dumps", recording)
    JointState(
        state_id="state",
        epistemic_group_id="state",
        weight=1,
        entity_paths=state.entity_paths,
        laws=(law,),
    )
    assert law.model_dump(mode="json") in encoded
    own = [row for row in encoded if isinstance(row, dict) and "state_id" in row]
    assert len(own) == 1
    assert "laws" not in own[0] and "entity_paths" not in own[0]


class _Child(Record):
    value: float


def _spoiled(child: _Child) -> _Child:
    return child.model_copy(update={"value": math.nan})


class _Parent(Record):
    child: _Child
    children: tuple[_Child, ...] = ()
    optional: _Child | None = None
    replaced: Annotated[_Child, AfterValidator(_spoiled)] | None = None
    mixed: tuple[_Child | float, ...] = ()
    open: Frozen[dict[str, object]] = Field(default_factory=FrozenDict)


def test_only_fields_that_hold_nothing_but_validated_records_are_left_out() -> None:
    assert _record_fields(_Parent) == {"child", "children", "optional"}
    assert _record_fields(_Child) is None
    assert _only_records(Annotated[tuple[_Child, ...], Field(min_length=1)])
    assert not _only_records(Annotated[_Child, AfterValidator(_spoiled)])
    assert not _only_records(tuple[_Child | float, ...])
    assert not _only_records(Frozen[dict[str, _Child]])
    assert not _only_records(float)
    # The law's own records are found through its fields' annotations.
    assert _record_fields(IncidentLaw) == {"parameter_rank", "response", "segments"}
    assert _record_fields(IncidentSegment) == {"kernel"}


def test_a_value_a_field_validator_puts_in_a_records_place_is_encoded() -> None:
    child = {"value": 1.0}
    _Parent.model_validate({"child": child})
    with pytest.raises(ValidationError, match="JSON compliant"):
        _Parent.model_validate({"child": child, "replaced": child})
    with pytest.raises(ValidationError, match="JSON compliant"):
        _Parent.model_validate({"child": child, "mixed": (math.nan,)})


def test_an_unvalidated_copy_is_refused_at_any_depth() -> None:
    child = _Child(value=1.0)
    spoiled = _spoiled(child)
    parent = _Parent(child=child)
    for value in (
        {"child": spoiled},
        {"child": child, "children": (child, spoiled)},
        {"child": child, "optional": spoiled},
        parent.model_copy(update={"child": spoiled}),
        parent.model_copy(update={"children": (spoiled,)}),
    ):
        with pytest.raises(ValidationError, match="JSON compliant"):
            _Parent.model_validate(value)


def test_the_check_runs_before_a_subclass_s_own_validators() -> None:
    from pydantic import model_validator

    seen: list[float] = []

    class Checked(Record):
        value: float

        @model_validator(mode="after")
        def note(self) -> Checked:
            seen.append(self.value)
            return self

    with pytest.raises(ValidationError, match="JSON compliant"):
        Checked(value=math.nan)
    assert seen == []
    Checked(value=2.0)
    assert seen == [2.0]
