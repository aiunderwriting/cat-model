"""Check a built source distribution against the tracked tree.

The source distribution carries the tree ``git archive`` exports: every file
Git tracks except those that ``.gitattributes`` marks ``export-ignore``. The
links in its pages therefore resolve, and its tests and scripts find the files
they read. Given the archive, this script reports:

- a file in it that Git does not track, other than the generated PKG-INFO;
- a tracked file that it leaves out, or an ``export-ignore`` file that it
  carries;
- a relative Markdown link in any of its pages whose target file, or whose
  heading in a target page, it does not carry.

Run it from a Git checkout of the revision the archive was built from. It
uses only the standard library, so it needs no project environment.
"""

import argparse
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Collection
from functools import cache
from pathlib import Path, PurePosixPath

GENERATED = frozenset({"PKG-INFO"})
"""Files the build writes into the source distribution."""

EXPLICIT_ANCHOR_RE = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"", re.IGNORECASE)
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)<>\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def strip_fenced_code(text: str) -> str:
    kept: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        kept.append(line)
    return "\n".join(kept)


def _github_slug(heading: str) -> str:
    """Approximate GitHub's heading-to-anchor slug for this repo's headings."""
    text = re.sub(r"<!--.*?-->", "", heading)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


@cache
def anchors(md_path: Path) -> frozenset[str]:
    """Return the heading slugs and explicit anchor ids of a Markdown page."""
    text = strip_fenced_code(md_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    slug_counts: dict[str, int] = {}
    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading is None:
            continue
        slug = _github_slug(heading.group(1))
        count = slug_counts.get(slug, 0)
        slug_counts[slug] = count + 1
        found.add(slug if count == 0 else f"{slug}-{count}")
    found.update(match.lower() for match in EXPLICIT_ANCHOR_RE.findall(text))
    return frozenset(found)


def link_errors(md_path: Path, root: Path) -> list[str]:
    """Return the relative links of a Markdown page that do not resolve.

    A link resolves when its target exists under ``root`` and, for a link to
    a heading of a Markdown page, when that page has the heading.
    """
    text = _INLINE_CODE_RE.sub(
        "", strip_fenced_code(md_path.read_text(encoding="utf-8"))
    )
    root = root.resolve()
    errors: list[str] = []
    for match in _LINK_RE.finditer(text):
        target = match.group(1)
        if target.startswith(_EXTERNAL_PREFIXES):
            continue
        path_part, _, fragment = target.partition("#")
        if path_part:
            resolved = (md_path.parent / path_part).resolve()
            if not resolved.exists() or not resolved.is_relative_to(root):
                errors.append(f"{target!r}: target file {path_part!r} does not exist")
                continue
        else:
            resolved = md_path
        if (
            fragment
            and resolved.suffix == ".md"
            and resolved.is_file()
            and fragment.lower() not in anchors(resolved)
        ):
            errors.append(
                f"{target!r}: no heading slug or explicit <a id=...> anchor "
                f"{fragment!r} in {resolved.name}"
            )
    return errors


def content_errors(
    shipped: set[str], tracked: set[str], export_ignored: Collection[str] = ()
) -> list[str]:
    """Compare the archive's file names with the names Git tracks, less the
    ``export_ignored`` ones, which the archive must leave out."""
    left_out = set(export_ignored)
    errors = [
        f"{path}: in the source distribution but not tracked by Git"
        for path in sorted(shipped - tracked - GENERATED)
    ]
    errors += [
        f"{path}: marked export-ignore but in the source distribution"
        for path in sorted(shipped & left_out)
    ]
    errors += [
        f"{path}: tracked by Git but not in the source distribution"
        for path in sorted(tracked - shipped - left_out)
    ]
    return errors


def sdist_errors(
    archive: Path, tracked: set[str], export_ignored: Collection[str] = ()
) -> list[str]:
    """Unpack a source distribution and return everything wrong with it."""
    with tempfile.TemporaryDirectory() as directory:
        with tarfile.open(archive) as tar:
            tar.extractall(directory, filter="data")
        roots = list(Path(directory).iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            return [f"{archive.name}: expected one top-level directory"]
        (root,) = roots
        shipped = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if not path.is_dir()
        }
        errors = content_errors(shipped, tracked, export_ignored)
        for page in sorted(root.rglob("*.md")):
            errors += [
                f"{page.relative_to(root).as_posix()}: broken link {error}"
                for error in link_errors(page, root)
            ]
        return errors


def tracked_files(checkout: Path) -> set[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {name for name in listing.split("\0") if name}


def _directories(name: str) -> list[str]:
    """The directories above a file, each with the trailing slash that makes
    Git match directory patterns against it."""
    return [f"{parent}/" for parent in PurePosixPath(name).parents if parent.name]


def export_ignored_files(checkout: Path, tracked: set[str]) -> set[str]:
    """The tracked files ``git archive`` leaves out of the tree it exports:
    each one that has the ``export-ignore`` attribute, or lies under a
    directory that has it, in the checkout's ``.gitattributes`` files."""
    queried = sorted({*tracked, *(d for name in tracked for d in _directories(name))})
    if not queried:
        return set()
    # Each path's report is three fields: the path, the attribute, its value.
    fields = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "export-ignore"],
        cwd=checkout,
        input="\0".join(queried) + "\0",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    marked = {path for path, value in zip(fields[0::3], fields[2::3]) if value == "set"}
    return {
        name
        for name in tracked
        if name in marked or any(d in marked for d in _directories(name))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path, help="The built .tar.gz archive.")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    tracked = tracked_files(checkout)
    errors = sdist_errors(args.sdist, tracked, export_ignored_files(checkout, tracked))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"{args.sdist.name}: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    print(f"{args.sdist.name}: matches the tracked tree and its links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
