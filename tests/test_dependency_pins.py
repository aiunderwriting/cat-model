from pathlib import Path

import pytest

from scripts import check_dependency_pins
from scripts.check_dependency_pins import _check_manifest, _discover_manifests, main


def _pyproject_with_git_source(source_line: str) -> str:
    return (
        "[project]\n"
        "name = 'demo'\n"
        "dependencies = ['example-lib']\n\n"
        "[tool.uv.sources]\n"
        f"{source_line}\n"
    )


_FULL_SHA = "910dc6a7e182f133e7afc9091485577917186565"


def test_bare_requirement_pinned_by_full_sha_rev_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        _pyproject_with_git_source(
            f'example-lib = {{ git = "https://example.com/example-lib", rev = "{_FULL_SHA}" }}'
        ),
        encoding="utf-8",
    )

    assert _check_manifest(manifest) == []


def test_bare_requirement_on_abbreviated_sha_rev_is_flagged(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        _pyproject_with_git_source(
            'example-lib = { git = "https://example.com/example-lib", rev = "910dc6a7" }'
        ),
        encoding="utf-8",
    )

    violations = _check_manifest(manifest)

    assert [item.requirement for item in violations] == ["example-lib"]


def test_bare_requirement_pinned_by_git_tag_source_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        _pyproject_with_git_source(
            'example-lib = { git = "https://example.com/example-lib", tag = "v0.2.0" }'
        ),
        encoding="utf-8",
    )

    assert _check_manifest(manifest) == []


def test_bare_requirement_on_git_branch_source_is_flagged(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        _pyproject_with_git_source(
            'example-lib = { git = "https://example.com/example-lib", branch = "main" }'
        ),
        encoding="utf-8",
    )

    violations = _check_manifest(manifest)

    assert [item.requirement for item in violations] == ["example-lib"]


def test_specifier_bearing_requirement_with_git_source_is_still_checked(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        _pyproject_with_git_source(
            f'example-lib = {{ git = "https://example.com/example-lib", rev = "{_FULL_SHA}" }}'
        ).replace("['example-lib']", "['example-lib>=0.2']"),
        encoding="utf-8",
    )

    violations = _check_manifest(manifest)

    assert [item.requirement for item in violations] == ["example-lib>=0.2"]


def test_dependency_pin_manifest_discovery_includes_nested_pyproject(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n",
        encoding="utf-8",
    )
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    nested = app_dir / "pyproject.toml"
    nested.write_text("[project]\nname = 'app'\n", encoding="utf-8")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyproject.toml").write_text("[project]\nname = 'env'\n", encoding="utf-8")

    manifests = _discover_manifests(tmp_path)

    assert manifests == [nested, tmp_path / "pyproject.toml"]


@pytest.mark.parametrize("ancestor_name", ["build", "dist", "venv", ".cache"])
def test_manifest_discovery_checks_checkout_under_excluded_ancestor(
    tmp_path: Path, ancestor_name: str
) -> None:
    root = tmp_path / ancestor_name / "repo"
    root.mkdir(parents=True)
    manifest = root / "pyproject.toml"
    manifest.write_text(
        "[project]\nname = 'demo'\ndependencies = ['example-lib>=1']\n",
        encoding="utf-8",
    )

    violations = [
        violation
        for discovered in _discover_manifests(root)
        for violation in _check_manifest(discovered)
    ]

    assert [violation.requirement for violation in violations] == ["example-lib>=1"]


@pytest.mark.parametrize("excluded_name", ["build", "dist", "venv", ".cache", ".venv"])
def test_manifest_discovery_still_excludes_nested_generated_directories(
    tmp_path: Path, excluded_name: str
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    excluded_dir = tmp_path / "app" / excluded_name / "nested"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "pyproject.toml").write_text(
        "[project]\nname = 'generated'\n", encoding="utf-8"
    )

    assert _discover_manifests(tmp_path) == [manifest]


def test_default_manifest_discovery_is_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    manifests = _discover_manifests()

    assert Path(__file__).resolve().parents[1] / "pyproject.toml" in manifests
    assert all(
        path.is_relative_to(Path(__file__).resolve().parents[1]) for path in manifests
    )


def test_build_requirements_must_be_exact_pins(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        "[build-system]\n"
        "requires = ['hatchling==1.33.0', 'example-plugin>=2', 'example-hook']\n",
        encoding="utf-8",
    )

    violations = _check_manifest(manifest)

    assert [(item.section, item.requirement) for item in violations] == [
        ("build-system.requires", "example-plugin>=2"),
        ("build-system.requires", "example-hook"),
    ]


def test_a_named_manifest_that_is_missing_fails_the_check(
    tmp_path: Path, capsys
) -> None:
    present = tmp_path / "pyproject.toml"
    present.write_text(
        "[project]\nname = 'demo'\ndependencies = ['example-lib==1.0']\n",
        encoding="utf-8",
    )
    missing = tmp_path / "absent" / "pyproject.toml"

    assert main([str(present)]) == 0
    assert "1 manifest" in capsys.readouterr().out
    for arguments in ([str(missing)], [str(present), str(missing)], [str(tmp_path)]):
        assert main(arguments) != 0
        captured = capsys.readouterr()
        assert arguments[-1] in captured.err
        assert "passed" not in captured.out


def test_a_run_that_checks_no_manifest_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_dependency_pins, "_discover_manifests", lambda: [])

    assert main([]) != 0
    assert "passed" not in capsys.readouterr().out
