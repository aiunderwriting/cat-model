from collections.abc import Iterator
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_crowdstrike_anchor_separates_direct_and_insured_loss() -> None:
    source_cards = yaml.safe_load(
        (REPO_ROOT / "data" / "elicitation_source_cards.yaml").read_text()
    )
    # Check each prompt-bearing excerpt so one correctly labeled card cannot vouch
    # for a mislabeled one elsewhere in the file.
    cited = [
        " ".join(text.split()) for text in _strings(source_cards) if "5.4B" in text
    ]
    assert cited, "CrowdStrike empirical anchor disappeared from source cards"
    for text in cited:
        assert "Fortune 500" in text, text
        assert any(
            amount in text for amount in ("0.54-$1.08B", "0.54-1.08B", "0.54B-1.08B")
        ), text
        assert "$5.4B insured" not in text, text
        assert "$5.4B in insured" not in text, text
        assert "USD 5.4B insured" not in text, text


def test_kev_card_states_what_a_catalog_entry_is_and_is_not() -> None:
    # KEV has no field that identifies exploitation before disclosure, so the
    # card states what an entry supports about population and timing and
    # asserts no zero-day share.
    context = " ".join(
        (REPO_ROOT / "data" / "elicitation_source_cards.yaml").read_text().split()
    )
    assert "CISA KEV records vulnerabilities with known exploitation" in context
    assert "An entry is not an attack, victim, or loss occurrence" in context
    assert (
        "catalog-addition date need not be its discovery or first-exploitation date"
        in context
    )
