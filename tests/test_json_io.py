"""Strict JSON refuses over-deep nesting by its own rule, not the interpreter's."""

import json

import pytest

from catmodel.json_io import (
    MAX_JSON_NESTING,
    exact_marker,
    require_marker,
    strict_json_loads,
)
from test_support.version_markers import unstated_markers

# Quotes, backslashes and brackets, which a string holds without nesting.
_CONTENT = '[{"\\]}\\\\" \\" é ['


def _nested(levels: int) -> object:
    """A value with exactly ``levels`` arrays and objects open at its deepest
    point, and bracket-laden strings at every level."""
    value: object = _CONTENT
    for level in range(levels):
        value = [_CONTENT, value] if level % 2 else {_CONTENT: value, "k": _CONTENT}
    return value


def _refused(text: str | bytes) -> None:
    with pytest.raises(ValueError, match=f"more than {MAX_JSON_NESTING} deep") as error:
        strict_json_loads(text)
    # Plain, so that an answer's retry feedback is the generic value error.
    assert type(error.value) is ValueError


@pytest.mark.parametrize("encode", [str, str.encode], ids=["str", "bytes"])
def test_nesting_is_accepted_to_the_limit_and_refused_one_level_deeper(encode):
    deepest = _nested(MAX_JSON_NESTING)
    assert strict_json_loads(encode(json.dumps(deepest))) == deepest
    _refused(encode(json.dumps(_nested(MAX_JSON_NESTING + 1))))


@pytest.mark.parametrize(
    "text",
    [
        "[" * 200_000 + "]" * 200_000,
        '{"a":' * 200_000 + "1" + "}" * 200_000,
        # Never closed: the decoder would descend before it found the end.
        "[" * 200_000,
        '{"value": ' + "[" * MAX_JSON_NESTING,
    ],
    ids=["arrays", "objects", "unclosed", "unclosed-one-level-beyond"],
)
def test_nesting_beyond_the_limit_is_refused_without_recursing(text):
    # 200,000 levels are far more than CPython 3.13 or 3.14 recurses through:
    # the decoder itself would raise RecursionError.
    _refused(text)


def test_brackets_and_quotes_inside_strings_are_not_nesting():
    # Tens of thousands of brackets, but two levels: strings hold the rest.
    document = {
        "open": "é[{" * 5000,
        "escaped": '\\"[' * 5000,
        "backslashes": ["\\" * 7, "\\" * 8 + "[" * 1000, '"' * 999],
        "closed": "]}" * 5000,
    }
    for ensure_ascii in (True, False):
        text = json.dumps(document, ensure_ascii=ensure_ascii)
        assert strict_json_loads(text) == document


def test_text_within_the_limit_keeps_its_own_error():
    # Not over-deep, so the decoder still names what is wrong with it.
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads("[" * MAX_JSON_NESTING)
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads('{"text": "' + "[" * 5000)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "utf-8-sig"])
def test_text_in_another_encoding_is_measured_as_text(encoding):
    _refused(json.dumps(_nested(MAX_JSON_NESTING + 1)).encode(encoding))
    _refused(("[" * (MAX_JSON_NESTING + 1)).encode(encoding))
    # In UTF-16 the two bytes of each of these characters are a quote and a
    # bracket. As text, a thousand of them in an unclosed string nest nothing.
    misleading = (chr(0x5B22) + chr(0x225B)) * 1000
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads(('["' + misleading).encode(encoding))


def test_only_the_same_value_of_the_same_type_is_the_marker():
    assert exact_marker(1, 1) and exact_marker("cat.results/1", "cat.results/1")
    for found in (True, 1.0, "1", 2, None, [1]):
        assert not exact_marker(found, 1)
    assert not exact_marker(1, "1")


def test_a_saved_file_must_state_its_marker():
    require_marker({"schema_version": 3, "rows": []}, "schema_version", 3, "a file")
    current = {"schema_version": 3}
    for label, payload in unstated_markers(current).items():
        with pytest.raises(ValueError, match="a file schema version") as refusal:
            require_marker(
                payload, "schema_version", 3, "a file schema version", "write it again"
            )
        assert str(refusal.value).endswith("; write it again"), label
        assert ("is not stated" in str(refusal.value)) is (label == "absent")
    # A file that holds no object states nothing.
    for payload in ([3], 3, None):
        with pytest.raises(ValueError, match="is not stated"):
            require_marker(payload, "schema_version", 3, "a file schema version")
