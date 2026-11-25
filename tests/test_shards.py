"""``pytest --shard I/N`` gives every collected test to exactly one of N parts."""

from types import SimpleNamespace

import pytest

from test_support.shards import parse_shard, shard_of
from tests import conftest

_IDS = [
    f"tests/test_example_{file}.py::test_case[{case}]"
    for file in range(40)
    for case in range(25)
]


def _selected(shard: str | None, ids: list[str]) -> tuple[list[str], list[str]]:
    """The ids the collection hook keeps and those it reports as deselected."""
    reported: list[str] = []
    config = SimpleNamespace(
        getoption=lambda name: shard,
        hook=SimpleNamespace(
            pytest_deselected=lambda items: reported.extend(i.nodeid for i in items)
        ),
    )
    items = [SimpleNamespace(nodeid=test_id) for test_id in ids]
    conftest.pytest_collection_modifyitems(config, items)  # ty: ignore[invalid-argument-type]
    return [item.nodeid for item in items], reported


@pytest.mark.parametrize("parts", [1, 2, 4, 7])
def test_the_parts_are_disjoint_and_together_hold_every_test(parts: int) -> None:
    kept = [_selected(f"{part}/{parts}", _IDS)[0] for part in range(1, parts + 1)]
    assert sorted(test_id for part in kept for test_id in part) == sorted(_IDS)
    # Each part keeps the collection's order and takes a real share of it.
    for part in kept:
        assert part == [test_id for test_id in _IDS if test_id in set(part)]
        assert len(part) > len(_IDS) // (2 * parts)


def test_a_part_reports_the_tests_it_leaves_to_the_others() -> None:
    kept, reported = _selected("2/4", _IDS)
    assert sorted(kept + reported) == sorted(_IDS)
    assert reported and not set(kept) & set(reported)


def test_without_the_option_every_test_runs() -> None:
    assert _selected(None, _IDS) == (_IDS, [])


def test_a_test_keeps_its_part_on_every_machine() -> None:
    # CRC-32 of the id, not Python's per-process hash.
    assert (
        shard_of("tests/test_shards.py::test_a_test_keeps_its_part_on_every_machine", 4)
        == 1
    )
    assert {shard_of(test_id, 4) for test_id in _IDS} == {1, 2, 3, 4}


def test_the_option_names_a_part_from_1_to_n() -> None:
    assert parse_shard("2/4") == (2, 4)
    assert parse_shard("1/1") == (1, 1)
    for text in (
        "0/4",
        "5/4",
        "2",
        "2/",
        "/4",
        "a/4",
        "2/b",
        "-1/4",
        "1/0",
        "1.0/4",
        "²/4",
    ):
        with pytest.raises(ValueError, match="--shard"):
            parse_shard(text)
