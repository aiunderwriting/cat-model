"""Native source authoring captures selected bytes without fitting risk parameters."""

import hashlib

import pytest
import yaml

from catmodel.evidence import source_inputs
from catmodel.evidence.source_bundle import (
    MANIFEST_PATH,
    PROVENANCE_PATH,
    SourceBundleProvenance,
    capture_source_bundle_snapshot,
)
from catmodel.evidence.source_inputs import build_source_bundle
from catmodel.scenarios.hierarchy_contract import (
    HierarchyContract,
    hierarchy_authority,
    load_hierarchy_contract,
)
from catmodel.worktree import CodeProvenance
from test_support.capability_sources import (
    AS_OF,
    BUNDLE_ID,
    BUNDLE_VERSION,
    CSV_PATH,
    authored_source_files,
)

PROVENANCE_FILE = "official-eci.provenance.yaml"


@pytest.fixture(autouse=True)
def clean_method(monkeypatch):
    monkeypatch.setattr(
        source_inputs,
        "resolve_code_provenance",
        lambda _: CodeProvenance("a" * 40, True, "clean", None),
    )


@pytest.fixture
def input_recipe(tmp_path):
    inputs = tmp_path / "inputs"
    files = authored_source_files()
    for relative, content in files.items():
        path = inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (inputs / PROVENANCE_FILE).write_text(
        yaml.safe_dump(
            {
                "source_id": "synthetic-eci-snapshot",
                "uri": "urn:fixture:synthetic-official-eci",
                "pulled_at": AS_OF,
                "qa_status": "fixture",
            }
        )
    )
    payload = {
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "created_at": AS_OF,
        "files": {relative: relative for relative in files},
        "sources": [{"snapshot_path": CSV_PATH, "metadata_file": PROVENANCE_FILE}],
    }
    path = inputs / "source-inputs.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _edit_recipe(path, edit):
    payload = yaml.safe_load(path.read_text())
    edit(payload)
    path.write_text(yaml.safe_dump(payload))


def _edit_provenance(recipe_path, edit):
    _edit_recipe(recipe_path.parent / PROVENANCE_FILE, edit)


def test_build_captures_only_authored_evidence_and_relocates(input_recipe, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build_source_bundle(input_recipe, first)
    build_source_bundle(input_recipe, second)
    assert manifest.bundle_schema_version == 1
    assert set(manifest.files) == set(authored_source_files())
    assert (first / MANIFEST_PATH).read_bytes() == (second / MANIFEST_PATH).read_bytes()
    for relative in manifest.files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    assert (
        manifest.external_sources[0].content_hash
        == hashlib.sha256((first / CSV_PATH).read_bytes()).hexdigest()
    )
    assert manifest.external_sources[0].captured_paths == [CSV_PATH]
    provenance = SourceBundleProvenance.model_validate(
        yaml.safe_load((first / PROVENANCE_PATH).read_text())
    )
    assert (
        provenance.source_input_recipe_sha256
        == hashlib.sha256(input_recipe.read_bytes()).hexdigest()
    )
    assert provenance.repo_git_sha == "a" * 40
    assert (
        provenance.model_dump(include={"hierarchy_version", "hierarchy_sha256"})
        == hierarchy_authority().model_dump()
    )
    snapshot = capture_source_bundle_snapshot(first)
    assert provenance.manifest_sha256 == snapshot.manifest_sha256
    assert (
        snapshot.read_bytes(PROVENANCE_PATH) == (first / PROVENANCE_PATH).read_bytes()
    )
    (input_recipe.parent / CSV_PATH).write_text("changed authoring input")
    assert capture_source_bundle_snapshot(first).files == snapshot.files


def test_identity_excludes_build_provenance_but_not_captured_bytes(
    input_recipe, tmp_path, monkeypatch
):
    first = tmp_path / "first"
    build_source_bundle(input_recipe, first)
    identity = capture_source_bundle_snapshot(first).manifest_sha256
    # The same evidence rebuilt at another commit after an assumptions.md edit
    # and an edit to the hierarchy's display text.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "assumptions.md").write_text("# An edited assumptions register\n")
    monkeypatch.setattr(source_inputs, "source_checkout_root", lambda: checkout)
    monkeypatch.setattr(
        source_inputs,
        "resolve_code_provenance",
        lambda _: CodeProvenance("b" * 40, True, "clean", None),
    )
    hierarchy = load_hierarchy_contract().model_dump(mode="json")
    hierarchy["nodes"][1]["display_prose"] += " Reworded."
    edited = hierarchy_authority(HierarchyContract.model_validate(hierarchy))
    monkeypatch.setattr(source_inputs, "hierarchy_authority", lambda: edited)
    rebuilt = tmp_path / "rebuilt"
    build_source_bundle(input_recipe, rebuilt)
    assert capture_source_bundle_snapshot(rebuilt).manifest_sha256 == identity
    assert (rebuilt / MANIFEST_PATH).read_bytes() == (
        first / MANIFEST_PATH
    ).read_bytes()
    before, after = (
        yaml.safe_load((root / PROVENANCE_PATH).read_text())
        for root in (first, rebuilt)
    )
    assert before["manifest_sha256"] == after["manifest_sha256"] == identity
    assert before["repo_git_sha"] != after["repo_git_sha"]
    assert before["assumptions_md_sha256"] != after["assumptions_md_sha256"]
    assert before["hierarchy_sha256"] != after["hierarchy_sha256"]
    # Changing a captured file changes the identity.
    (input_recipe.parent / CSV_PATH).write_bytes(
        b"Model,ECI\nsynthetic-release,151.0\n"
    )
    changed = tmp_path / "changed"
    build_source_bundle(input_recipe, changed)
    assert capture_source_bundle_snapshot(changed).manifest_sha256 != identity


def test_provenance_must_describe_its_own_manifest(input_recipe, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    build_source_bundle(input_recipe, first)
    _edit_recipe(
        input_recipe, lambda recipe: recipe.update(created_at="2026-09-02T00:00:00Z")
    )
    build_source_bundle(input_recipe, second)
    (first / PROVENANCE_PATH).write_bytes((second / PROVENANCE_PATH).read_bytes())
    with pytest.raises(ValueError, match="provenance describes a different manifest"):
        capture_source_bundle_snapshot(first)
    (first / PROVENANCE_PATH).write_text(
        (first / PROVENANCE_PATH).read_text() + "run_id: downstream\n"
    )
    with pytest.raises(ValueError, match="downstream-only elicitation dependency"):
        capture_source_bundle_snapshot(first)


def test_optional_empirical_evidence_is_captured_without_a_fit(input_recipe, tmp_path):
    path = input_recipe.parent / "observations.yaml"
    path.write_text("description: upstream event observations\nlosses: [100, 200]\n")
    _edit_recipe(
        input_recipe,
        lambda recipe: recipe["files"].update(
            {"evidence/observations.yaml": "observations.yaml"}
        ),
    )
    manifest = build_source_bundle(input_recipe, tmp_path / "bundle")
    assert "evidence/observations.yaml" in manifest.files
    assert not any(name.startswith("calibration_artifacts/") for name in manifest.files)


def test_existing_outputs_are_never_overwritten(input_recipe, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("original evidence")
    with pytest.raises(ValueError, match="must be new"):
        build_source_bundle(input_recipe, output)
    assert (output / "sentinel").read_text() == "original evidence"


@pytest.mark.parametrize(
    "destination",
    ["manifest.yaml", "provenance.yaml", "../escaped.yaml", "/tmp/escaped.yaml"],
)
def test_generated_manifest_and_path_escapes_are_rejected(
    input_recipe, tmp_path, destination
):
    _edit_recipe(
        input_recipe,
        lambda recipe: recipe["files"].update({destination: "source-inputs.yaml"}),
    )
    with pytest.raises(ValueError, match=r"Generated manifest|traverse|absolute"):
        build_source_bundle(input_recipe, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_input_symlink_cannot_capture_outside_recipe(input_recipe, tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("outside evidence")
    (input_recipe.parent / "linked.txt").symlink_to(outside)
    _edit_recipe(
        input_recipe,
        lambda recipe: recipe["files"].update({"evidence/linked.txt": "linked.txt"}),
    )
    with pytest.raises(ValueError, match="must remain inside its root"):
        build_source_bundle(input_recipe, tmp_path / "invalid")


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("downstream.txt", "run_id: downstream\n"),
        ("downstream.txt", "model_id: respondent\n"),
        ("downstream.txt", "notes: elicitation-run://risk-output\n"),
        ("downstream.txt", '{"run_id":"downstream","model_id":"respondent"}'),
        ("downstream.csv", "run_id,model_id\ndownstream,respondent\n"),
    ],
)
def test_downstream_content_never_publishes(input_recipe, tmp_path, name, payload):
    (input_recipe.parent / name).write_text(payload)
    _edit_recipe(
        input_recipe,
        lambda recipe: recipe["files"].update({f"evidence/{name}": name}),
    )
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="downstream-only elicitation dependency"):
        build_source_bundle(input_recipe, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".source-build-*"))


def test_required_capture_failure_never_publishes(input_recipe, tmp_path):
    _edit_recipe(input_recipe, lambda recipe: recipe["files"].pop(CSV_PATH))
    with pytest.raises(ValueError, match="Source snapshot is not an exported file"):
        build_source_bundle(input_recipe, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_source_hashes_are_generated_from_selected_bytes(input_recipe, tmp_path):
    _edit_provenance(
        input_recipe, lambda provenance: provenance.update(content_hash="invented")
    )
    with pytest.raises(ValueError, match="Source hashes are generated"):
        build_source_bundle(input_recipe, tmp_path / "invalid")


def test_a_recipe_refuses_a_field_its_contract_does_not_define(input_recipe, tmp_path):
    _edit_recipe(input_recipe, lambda recipe: recipe.update({"unknown_block": {}}))
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        build_source_bundle(input_recipe, tmp_path / "invalid")


def test_a_recipe_repeating_a_key_never_publishes(input_recipe, tmp_path):
    with open(input_recipe, "a") as recipe:
        recipe.write("created_at: '2026-09-02T00:00:00Z'\n")
    with pytest.raises(yaml.YAMLError, match="duplicate key 'created_at'"):
        build_source_bundle(input_recipe, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_clean_checkout_is_required(input_recipe, tmp_path, monkeypatch):
    monkeypatch.setattr(
        source_inputs,
        "resolve_code_provenance",
        lambda _: CodeProvenance("a" * 40, True, "dirty", "b" * 64),
    )
    with pytest.raises(ValueError, match="clean committed method checkout"):
        build_source_bundle(input_recipe, tmp_path / "invalid")


def test_method_change_during_build_never_publishes(
    input_recipe, tmp_path, monkeypatch
):
    commits = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(
        source_inputs,
        "resolve_code_provenance",
        lambda _: CodeProvenance(next(commits), True, "clean", None),
    )
    with pytest.raises(ValueError, match="Method checkout changed"):
        build_source_bundle(input_recipe, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_bundle_output_must_be_outside_existing_evidence(input_recipe, tmp_path):
    first = tmp_path / "first"
    build_source_bundle(input_recipe, first)
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    for output in (first / "nested", alias / "nested"):
        with pytest.raises(ValueError, match="outside every existing bundle"):
            build_source_bundle(input_recipe, output)


def test_source_provenance_is_authored_in_a_metadata_file(input_recipe, tmp_path):
    manifest = build_source_bundle(input_recipe, tmp_path / "bundle")
    source = manifest.external_sources[0]
    assert (source.source_id, source.pulled_at) == ("synthetic-eci-snapshot", AS_OF)
    assert (
        source.content_hash
        == hashlib.sha256((input_recipe.parent / CSV_PATH).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda source: source.update(metadata={"source_id": "inline"}),
            r"metadata\n  Extra inputs are not permitted",
        ),
        (
            lambda source: source.pop("metadata_file"),
            r"metadata_file\n  Field required",
        ),
        (
            lambda source: source.update(metadata_file="../outside.yaml"),
            "must remain inside its root",
        ),
        (
            lambda source: source.update(metadata_file=CSV_PATH),
            "must contain a mapping",
        ),
    ],
    ids=["inline", "absent", "outside-the-recipe", "not-a-mapping"],
)
def test_a_source_names_one_metadata_file_beside_the_recipe(
    input_recipe, tmp_path, edit, message
):
    (tmp_path / "outside.yaml").write_text("source_id: outside\n")
    _edit_recipe(input_recipe, lambda recipe: edit(recipe["sources"][0]))
    with pytest.raises(ValueError, match=message):
        build_source_bundle(input_recipe, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
