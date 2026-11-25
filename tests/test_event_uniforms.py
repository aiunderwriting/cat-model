"""Counter-addressed physical ranks retain pairing independently of execution."""

import numpy as np
import pytest

from catmodel.probability.keyed_streams import _philox4x32, event_uniforms


def _ranks(years, ordinals, **changes):
    return event_uniforms(
        years,
        ordinals,
        root_seed=changes.get("root_seed", 42),
        state_id=changes.get("state_id", "state"),
        origin_id=changes.get("origin_id", "origin"),
        purpose=changes.get("purpose", "rank"),
    )


def test_philox_matches_random123_published_known_answer():
    # DEShawResearch/random123 tests/kat_vectors: Philox4x32, ten rounds,
    # zero counter and zero key. This checks the algorithm independently of
    # the stream-key encoding and coordinate layout.
    actual = _philox4x32(
        tuple(np.array([0], dtype=np.uint64) for _ in range(4)), (0, 0)
    )
    assert [int(word[0]) for word in actual] == [
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    ]


def test_event_ranks_are_identical_across_masks_batches_reordering_and_large_coordinates():
    years = np.array([0, 0, 1, 1, 2**32, 2**32 + 1], dtype=np.uint64)
    ordinals = np.array([0, 1, 0, 2**32, 1, 2**32 + 1], dtype=np.uint64)
    ranks = _ranks(years, ordinals)
    np.testing.assert_array_equal(
        np.concatenate(
            [_ranks(years[:2], ordinals[:2]), _ranks(years[2:], ordinals[2:])]
        ),
        ranks,
    )
    order = np.array([3, 1, 0, 4, 5, 2])
    np.testing.assert_array_equal(_ranks(years[order], ordinals[order]), ranks[order])
    np.testing.assert_array_equal(
        _ranks(years[[0, 2, 5]], ordinals[[0, 2, 5]]), ranks[[0, 2, 5]]
    )
    assert len(set(ranks)) == len(ranks)
    assert np.all((ranks >= 0) & (ranks < 1))
    for changes in (
        {"origin_id": "other"},
        {"purpose": "lab"},
        {"root_seed": 43},
        {"state_id": "other"},
    ):
        assert not np.any(_ranks(years, ordinals, **changes) == ranks)


@pytest.mark.parametrize(
    "years,ordinals",
    [([-1], [0]), ([0], [-1]), ([0.0], [0]), ([0], [True]), ([0, 1], [0])],
)
def test_event_coordinates_reject_unidentified_or_invalid_addresses(years, ordinals):
    with pytest.raises(ValueError, match="aligned nonnegative integers"):
        _ranks(np.asarray(years), np.asarray(ordinals))
