"""Re-reduction cannot write into immutable packages or captures, including aliases."""

import hashlib

import pytest

from catmodel.results.execution import replay_capture, run_setup
from test_support.study import _package, _plan


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "target", ["package", "capture", "package_alias", "capture_alias"]
)
def test_replay_rejects_evidence_ancestors_without_mutation(tmp_path, target):
    package = _package(tmp_path, plan=_plan(32))
    run = tmp_path / "run"
    run_setup(setup_path=package.root, output=run, code_commit="producer")
    capture = run / "capture"
    before = (_snapshot(package.root), _snapshot(capture))
    root = package.root if target.startswith("package") else capture
    if target.endswith("alias"):
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    with pytest.raises(ValueError, match="immutable evidence"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=root / "nested.json",
            code_commit="reducer",
        )
    assert (_snapshot(package.root), _snapshot(capture)) == before
