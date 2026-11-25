import re
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement

from scripts.check_dependency_pins import _pin_problem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_build_requirements_are_exact_pins() -> None:
    requirements = _pyproject()["build-system"]["requires"]
    assert {
        entry: problem
        for entry in requirements
        if (problem := _pin_problem(Requirement(entry))) is not None
    } == {}


def test_the_project_refuses_no_uv_version() -> None:
    # The workflows name the uv that CI runs; a contributor's own uv is not
    # refused. A uv too old to read the relative exclude-newer ignores the
    # whole [tool.uv] table, a floor included, so a floor could only refuse
    # versions that work.
    assert "required-version" not in _pyproject()["tool"]["uv"]


def test_release_metadata_and_console_script_are_declared() -> None:
    data = _pyproject()
    project = data["project"]

    assert project["readme"] == "README.md"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert project["authors"]
    assert project["scripts"]["catmodel"] == "catmodel.cli:cli"


def test_citation_file_names_the_project_version() -> None:
    citation = yaml.load(
        (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert citation["version"] == _pyproject()["project"]["version"]


def test_sdist_ships_the_lock_file_and_names_no_directory_of_local_downloads() -> None:
    data = _pyproject()
    included = set(data["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"])

    assert "uv.lock" in included
    # .gitignore keeps local downloads under these directories, so the
    # tracked paths inside them are named one by one.
    assert not {"data", "data/external", "references"} & included


def test_sdist_ships_everything_its_pages_link_to() -> None:
    """A page the source distribution includes may link to a file or
    directory only if that is included too, so the archive has no dangling
    relative links. The release workflow checks the built archive with
    scripts/check_sdist.py; this checks the configuration it is built from."""
    data = _pyproject()
    entries = [
        *data["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"],
        *data["project"]["license-files"],
    ]
    assert {"ARCHITECTURE.md", "CONTRIBUTING.md", "docs"} <= set(entries)

    def included(path: Path) -> bool:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        return any(
            relative == entry or relative.startswith(f"{entry}/") for entry in entries
        )

    pages = []
    for entry in entries:
        path = PROJECT_ROOT / entry
        if path.is_dir():
            pages.extend(path.rglob("*.md"))
        elif path.suffix == ".md":
            pages.append(path)
    missing = set()
    for page in pages:
        for target in re.findall(r"\]\(([^)\s]+)\)", page.read_text(encoding="utf-8")):
            if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            linked = (page.parent / target.split("#", 1)[0]).resolve()
            if not linked.is_relative_to(PROJECT_ROOT) or not included(linked):
                missing.add(f"{page.relative_to(PROJECT_ROOT)} -> {target}")
    assert not missing, sorted(missing)


def test_release_workflow_checks_the_built_source_distribution() -> None:
    release = yaml.load(
        (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    (check,) = (
        step
        for step in release["jobs"]["artifact"]["steps"]
        if "scripts/check_sdist.py" in step.get("run", "")
    )
    assert "if" not in check
    assert check["run"].endswith("scripts/check_sdist.py dist/*.tar.gz")


def test_wheel_does_not_package_elicitation_source_cards() -> None:
    # Source cards reach prompts only through captured source bundles.
    force_include = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]
    assert "data/elicitation_source_cards.yaml" not in force_include


def test_model_ci_runs_every_job_and_the_required_check_waits_for_all() -> None:
    workflows = PROJECT_ROOT / ".github" / "workflows"
    ci = yaml.load((workflows / "ci.yml").read_text(), Loader=yaml.BaseLoader)

    assert {"pull_request", "push"} <= set(ci["on"])
    jobs = ci["jobs"]
    gated = set(jobs) - {"ci-green"}
    assert "release-artifact" in gated
    # No job is selected by what a change touches.
    assert not [name for name in gated if "if" in jobs[name]]
    # The required check runs even when a job fails, and names every job.
    assert jobs["ci-green"]["if"] == "${{ always() }}"
    assert set(jobs["ci-green"]["needs"]) == gated
