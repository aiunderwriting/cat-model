"""A setup's scientific digest hashes its inline encoding, less its disclosure
prose, field by field."""

from dataclasses import replace

import pytest

from test_support.conditional import conditional_study
from test_support.run_setup import scientific_digest
from test_support.study import _financial_study, _study


@pytest.mark.parametrize(
    "build", [_financial_study, _study, conditional_study], ids=lambda f: f.__name__
)
def test_the_digest_is_that_of_the_inline_encoding_without_its_prose(build):
    setup = build()
    assert setup.scientific_sha256 == scientific_digest(setup)


def test_rewording_the_disclosure_prose_keeps_the_identity():
    setup = _study()
    assert setup.alternatives
    reworded = replace(
        setup,
        scope=setup.scope.model_copy(update={"title": "A reworded title"}),
        limitations=(*setup.limitations, "A limitation added by a copy edit."),
        alternatives=tuple(
            row.model_copy(update={"description": f"Reworded: {row.description}"})
            for row in setup.alternatives
        ),
    )
    # The setup keeps the new prose to disclose; its scientific identity is
    # unchanged.
    assert reworded != setup
    assert reworded.scientific_sha256 == setup.scientific_sha256
