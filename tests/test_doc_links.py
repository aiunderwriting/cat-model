from pathlib import Path

import pytest

from scripts.check_sdist import (
    EXPLICIT_ANCHOR_RE,
    anchors,
    link_errors,
    strip_fenced_code,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _lint_targets() -> list[Path]:
    files = sorted(_PROJECT_ROOT.glob("*.md"))
    files += sorted((_PROJECT_ROOT / "docs").rglob("*.md"))
    files += sorted((_PROJECT_ROOT / "references").rglob("*.md"))
    assert files, "doc-link lint found no markdown files; scope is broken"
    return files


@pytest.mark.parametrize(
    "md_path", _lint_targets(), ids=lambda p: str(p.relative_to(_PROJECT_ROOT))
)
def test_internal_doc_links_resolve(md_path: Path) -> None:
    errors = link_errors(md_path, _PROJECT_ROOT)
    assert not errors, (
        f"{md_path.relative_to(_PROJECT_ROOT)} has broken internal links:\n- "
        + "\n- ".join(errors)
    )


def test_explicit_anchors_are_unique_per_file() -> None:
    for md_path in _lint_targets():
        text = strip_fenced_code(md_path.read_text(encoding="utf-8"))
        ids = [match.lower() for match in EXPLICIT_ANCHOR_RE.findall(text)]
        duplicated = sorted({anchor for anchor in ids if ids.count(anchor) > 1})
        assert not duplicated, (
            f"{md_path.relative_to(_PROJECT_ROOT)} declares duplicate explicit "
            f"anchors {duplicated}; anchor ids must be unique within a file."
        )


def test_lint_detects_broken_anchor_and_moved_file(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text(
        '# Title\n\n<a id="good-anchor"></a>\n\n## Kept Section\n', encoding="utf-8"
    )
    source = tmp_path / "source.md"
    source.write_text(
        "[ok file](target.md) [ok slug](target.md#kept-section) "
        "[ok id](target.md#good-anchor)\n"
        "[bad anchor](target.md#gone-section)\n"
        "[moved file](renamed.md)\n",
        encoding="utf-8",
    )
    errors = link_errors(source, tmp_path)
    assert len(errors) == 2
    assert any("gone-section" in error for error in errors)
    assert any("renamed.md" in error for error in errors)


def test_result_limitation_references_name_limitations_headings():
    from catmodel.results.result_disclosures import (
        BASE_CASE_ANCHOR,
        LIMITATIONS_DOCUMENT,
        MODEL_LIMITATIONS,
    )

    headings = anchors(_PROJECT_ROOT / LIMITATIONS_DOCUMENT)
    referenced = [BASE_CASE_ANCHOR, *(anchor for anchor, _ in MODEL_LIMITATIONS)]
    assert [anchor for anchor in referenced if anchor not in headings] == []
