"""The assumption register and the code sensitivity catalog agree entry by entry.

assumptions.md gives every register entry its own section, headed by its ID,
whose labeled fields come in a fixed order and end with Challenge mechanisms.
That field states which catalogued sensitivity families challenge the entry,
why any required challenge is unavailable, or which check other than a study
alternative applies. Each group of entries opens with an index table linking
to its sections. The frozen plan decides what a study executes and reports;
the catalog names reusable source-specific challenge mechanisms.
"""

import re
from collections import defaultdict
from pathlib import Path

from catmodel.setup.sensitivity_catalog import (
    FIXED_ANCHOR_UNAVAILABLE_CHALLENGES,
    PUBLIC_STUDY_UNAVAILABLE_CHALLENGES,
    REGISTER_CHECKS_WITHOUT_STUDY_ALTERNATIVE,
    SENSITIVITY_FAMILIES,
    UNAVAILABLE_ASSUMPTION_SENSITIVITIES,
)

REGISTER = Path(__file__).resolve().parents[1] / "assumptions.md"
_REGISTER_ID = r"[A-Z]+(?:-[A-Z0-9]+)+"
_ID_HEADING = re.compile(rf"^### ({_REGISTER_ID})$")
_FIELD = re.compile(r"^\*\*([A-Z][a-z]+(?: [a-z]+)*)\.\*\* (.+)$")
_INDEX_ROW = re.compile(rf"^\| \[`({_REGISTER_ID})`\]\(#([a-z0-9-]+)\) \|")
_STATUS_LABEL = re.compile(r"`(OK|INITIAL_ESTIMATE|PLACEHOLDER|GAP)`")
_FAMILY_LIST = re.compile(r"(`[a-z_]+`(, `[a-z_]+`)*\.?)?")
_UNAVAILABLE = "Unavailable: "
_NOT_AN_ALTERNATIVE = "Not a study alternative: "
# An assumption's fields, and those of a numerical control, in their fixed order.
ASSUMPTION_FIELDS = (
    "Assumption",
    "Rationale",
    "Status",
    "Required sensitivity",
    "Challenge mechanisms",
)
CONTROL_FIELDS = (
    "Control",
    "Rationale",
    "Required numerical comparison",
    "Challenge mechanisms",
)
# Each kind of entry's index columns.
_INDEX_HEADERS: dict[tuple[str, ...], list[str]] = {
    ASSUMPTION_FIELDS: ["ID", "Assumption", "Status"],
    CONTROL_FIELDS: ["ID", "Control"],
}


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_register() -> tuple[
    dict[str, dict[str, str]], dict[str, str], dict[str, list[list[str]]]
]:
    """Each entry's fields by ID, each entry's group heading, and each group's
    index rows.

    Every level-3 heading is an entry ID, and every nonblank line of an entry
    is one labeled field, in one of the fixed orders. A table may carry a
    register ID only as an index row whose first cell links to a section.
    """
    sections: dict[str, dict[str, str]] = {}
    groups: dict[str, str] = {}
    indexes: dict[str, list[list[str]]] = defaultdict(list)
    group: str | None = None
    entry: str | None = None
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            group, entry = line.removeprefix("## "), None
            continue
        if line.startswith("### "):
            heading = _ID_HEADING.match(line)
            assert heading is not None, line
            entry = heading.group(1)
            assert entry not in sections, f"duplicate register ID {entry}"
            assert group is not None, entry
            sections[entry] = {}
            groups[entry] = group
            continue
        if entry is not None:
            if not line.strip():
                continue
            field = _FIELD.match(line)
            assert field is not None, (entry, line[:80])
            label, text = field.groups()
            assert label not in sections[entry], (entry, label)
            sections[entry][label] = text
            continue
        if line.startswith("|") and re.search(rf"`{_REGISTER_ID}`", _cells(line)[0]):
            assert _INDEX_ROW.match(line) is not None, line[:80]
        if line.startswith("|") and group is not None:
            indexes[group].append(_cells(line))
    for register_id, fields in sections.items():
        assert tuple(fields) in _INDEX_HEADERS, (register_id, tuple(fields))
    return sections, groups, dict(indexes)


def _register_sections() -> dict[str, dict[str, str]]:
    """Each register entry's labeled fields, by ID."""
    return _parse_register()[0]


def _register_rows() -> dict[str, str]:
    """Each register entry's Challenge mechanisms field, by ID."""
    return {
        register_id: fields["Challenge mechanisms"]
        for register_id, fields in _register_sections().items()
    }


def _catalog_families() -> dict[str, set[str]]:
    families: dict[str, set[str]] = defaultdict(set)
    for family in SENSITIVITY_FAMILIES:
        for register_id in family.assumption_ids:
            families[register_id].add(family.family_id)
    return families


def test_every_register_row_declares_challenges_or_a_non_alternative_check():
    rows = _register_rows()
    coded = set(_catalog_families()) | set(UNAVAILABLE_ASSUMPTION_SENSITIVITIES)
    # The code names no register entry that the register does not have.
    assert coded <= set(rows), sorted(coded - set(rows))
    # Every other entry is a code-owned check without a study alternative, and
    # an entry the catalog or the unavailable reasons name is not.
    without_alternative = set(REGISTER_CHECKS_WITHOUT_STUDY_ALTERNATIVE)
    assert without_alternative == set(rows) - coded
    assert not without_alternative & coded


def test_challenge_cells_restate_the_catalog_and_unavailable_reasons():
    catalog = _catalog_families()
    for register_id, cell in _register_rows().items():
        check = REGISTER_CHECKS_WITHOUT_STUDY_ALTERNATIVE.get(register_id)
        if check is not None or cell.startswith(_NOT_AN_ALTERNATIVE):
            assert cell == f"{_NOT_AN_ALTERNATIVE}{check}", register_id
            continue
        listed, _, reason = cell.partition(_UNAVAILABLE)
        assert _FAMILY_LIST.fullmatch(listed.strip()), (register_id, cell)
        assert set(re.findall(r"`([a-z_]+)`", listed)) == catalog.get(
            register_id, set()
        ), register_id
        assert (reason or None) == UNAVAILABLE_ASSUMPTION_SENSITIVITIES.get(
            register_id
        ), register_id


def test_register_cross_references_name_existing_rows():
    text = REGISTER.read_text(encoding="utf-8")
    referenced = set(re.findall(rf"`({_REGISTER_ID})`", text))
    assert referenced <= set(_register_rows()), sorted(
        referenced - set(_register_rows())
    )


def test_rows_challenged_by_a_judgmental_range_state_its_status():
    # An alternative's size set by judgment in code states its evidence status where the
    # register defines the alternative, its Required sensitivity field.
    sections = _register_sections()
    for family in SENSITIVITY_FAMILIES:
        if family.range_basis != "judgment":
            continue
        for register_id in family.assumption_ids:
            # Only assumption entries, not numerical controls, have the field.
            assert tuple(sections[register_id]) == ASSUMPTION_FIELDS, register_id
            required = sections[register_id]["Required sensitivity"]
            assert re.search(r"`(INITIAL_ESTIMATE|PLACEHOLDER)`", required), (
                family.family_id,
                register_id,
            )


def test_route_challenges_are_unavailable_register_rows():
    # Each frozen route challenge carries the register's unavailable reason and
    # is not also a catalog family.
    for route in (
        PUBLIC_STUDY_UNAVAILABLE_CHALLENGES,
        FIXED_ANCHOR_UNAVAILABLE_CHALLENGES,
    ):
        assert len(set(route)) == len(route)
        assert set(route) <= set(UNAVAILABLE_ASSUMPTION_SENSITIVITIES)
        assert not set(route) & {family.family_id for family in SENSITIVITY_FAMILIES}


def test_each_entry_has_one_section_and_its_group_index_links_to_it():
    text = REGISTER.read_text(encoding="utf-8")
    sections, groups, indexes = _parse_register()
    headings = re.findall(rf"^### ({_REGISTER_ID})$", text, flags=re.MULTILINE)
    assert sorted(headings) == sorted(sections)
    assert len(headings) == len(set(headings))
    # GitHub's anchor for a heading of an ID is the ID in lower case.
    anchors = {register_id.lower(): register_id for register_id in sections}
    for group, rows in indexes.items():
        entries = [
            register_id for register_id in sections if groups[register_id] == group
        ]
        assert entries, f"the index of {group!r} has no sections"
        header, rule, *body = rows
        # A group holds one kind of entry, whose columns its index has.
        (kind,) = {tuple(sections[register_id]) for register_id in entries}
        assert header == _INDEX_HEADERS[kind], (group, header)
        assert set("".join(rule)) <= set("-: "), (group, rule)
        linked = []
        for row in body:
            match = re.fullmatch(rf"\[`({_REGISTER_ID})`\]\(#([a-z0-9-]+)\)", row[0])
            assert match is not None, row
            register_id, anchor = match.groups()
            assert anchors.get(anchor) == register_id, row
            assert len(row) == len(header) and row[1], row
            if "Status" in header:
                status = sections[register_id]["Status"]
                labels = dict.fromkeys(_STATUS_LABEL.findall(status))
                assert row[2] == ", ".join(f"`{label}`" for label in labels), row
            linked.append(register_id)
        # The index lists exactly its group's entries, in section order.
        assert linked == entries, group
    assert set(indexes) == set(groups.values())
