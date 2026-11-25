"""The required release lane executes reproducible installed-wheel acceptance."""

import re
from pathlib import Path

from catmodel.yaml_io import safe_load
from tests import test_cli_stages

_ROOT = Path(__file__).resolve().parents[1]


def test_required_ci_reuses_the_release_wheel_for_offline_study_acceptance():
    ci = safe_load((_ROOT / ".github/workflows/ci.yml").read_bytes())
    assert "release-artifact" in ci["jobs"]["ci-green"]["needs"]
    assert ci["jobs"]["release-artifact"]["uses"] == "./.github/workflows/release.yml"
    release = safe_load((_ROOT / ".github/workflows/release.yml").read_bytes())
    steps = release["jobs"]["artifact"]["steps"]
    wheel_check = next(
        step
        for step in steps
        if "scripts/check_incident_wheel.py" in step.get("run", "")
    )
    assert "if" not in wheel_check
    assert "--wheel dist/*.whl" in wheel_check["run"]
    assert '"$RUNNER_TEMP/incident-wheel/venv/bin/python"' in wheel_check["run"]
    assert "uv pip install" not in wheel_check["run"]
    # The installed wheel's CLI is checked against the pinned command inventory.
    assert "from tests import test_cli_stages" in wheel_check["run"]
    (inventory_test,) = re.findall(r"test_cli_stages\.(\w+)\(\)", wheel_check["run"])
    assert callable(getattr(test_cli_stages, inventory_test))
    retained = next(
        step
        for step in steps
        if step.get("name") == "Retain installed-wheel acceptance"
    )
    assert retained["if"] == "always()"
    assert retained["with"]["path"].endswith("/incident-wheel/acceptance.json")


def test_a_tag_retains_the_checked_distributions_with_their_checksums():
    release = safe_load((_ROOT / ".github/workflows/release.yml").read_bytes())
    steps = release["jobs"]["artifact"]["steps"]
    names = [step.get("name") for step in steps]
    checksums = steps[names.index("Write release checksums")]
    retained = steps[names.index("Retain release distributions")]
    acceptance = steps[names.index("Retain installed-wheel acceptance")]
    wheel_check = names.index("Check installed wheel and smoke-test CLI")

    # Only a tag's files are retained, and only once every check has passed.
    assert names.index("Write release checksums") > wheel_check
    assert names.index("Retain release distributions") > names.index(
        "Write release checksums"
    )
    assert checksums["if"] == retained["if"] == "github.ref_type == 'tag'"
    assert checksums["working-directory"] == "dist"
    assert "sha256sum -- *.whl *.tar.gz > SHA256SUMS" in checksums["run"]
    assert retained["uses"] == acceptance["uses"]
    assert retained["with"]["path"].split() == [
        "dist/*.whl",
        "dist/*.tar.gz",
        "dist/SHA256SUMS",
    ]
    assert retained["with"]["if-no-files-found"] == "error"
