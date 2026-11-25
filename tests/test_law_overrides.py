"""An alternative's law overrides are held by the setup, in memory or in the
alternative's own file, and a saved setup reads one arm at a time."""

import hashlib
import json
from dataclasses import replace
from functools import cache
from typing import Any, cast

import pytest
from pydantic import ValidationError

from catmodel.contracts import law_overrides
from catmodel.contracts._setup_codec import saved_setup, stored_inputs_sha256
from catmodel.contracts.incident import Alternative, LawOverride
from catmodel.contracts.law_overrides import (
    ARM_FILE_SCHEMA_VERSION,
    HeldOverrides,
    OverrideWriter,
    StoredOverrides,
    arm_file,
    arm_file_name,
    read_arm_file,
)
from catmodel.hashing import canonical_json, canonical_json_sha256
from catmodel.results.execution import (
    StateLayout,
    load_with_workload,
    setup_workload,
)
from catmodel.setup.store import build_setup, load_setup
from catmodel.simulation.incident import (
    compile_incident_state,
    declared_view_identifiers,
)
from catmodel.stored_files import StoredFiles
from test_support.run_setup import scientific_digest
from tests.test_public_frontier_comparisons import _compile


@cache
def _study():
    """A public study with the default comparison families: several arms
    that override laws and several that only reweight states."""
    return _compile()


def _overriding(setup):
    return [
        row.alternative_id
        for row in setup.alternatives
        if setup.overrides.of(row.alternative_id)
    ]


def _file(setup, alternative_id):
    data = arm_file(alternative_id, setup.overrides.inline(alternative_id))
    return arm_file_name(data), data


def test_a_setup_holds_its_overrides_by_alternative_in_inline_order():
    setup = _study()
    arms = _overriding(setup)
    assert len(arms) >= 3 and len(arms) < len(setup.alternatives)
    assert setup.overrides.arms() == tuple(arms)
    assert not hasattr(setup.alternatives[0], "overrides")
    for arm in arms:
        rows = setup.overrides.of(arm)
        assert all(isinstance(row, LawOverride) for row in rows)
        assert setup.overrides.keys(arm) == tuple(
            (row.state_id, row.law.origin_id) for row in rows
        )
        assert [row["law"]["origin_id"] for row in setup.overrides.inline(arm)] == [
            row.law.origin_id for row in rows
        ]
        state_id = rows[0].state_id
        assert setup.overrides.laws(arm, state_id) == tuple(
            row.law for row in rows if row.state_id == state_id
        )
    reweighting = next(
        row.alternative_id
        for row in setup.alternatives
        if row.alternative_id not in arms
    )
    assert setup.overrides.of(reweighting) == ()
    assert setup.overrides.inline(reweighting) == []
    assert setup.overrides.laws(reweighting, setup.joint_states[0].state_id) == ()


def test_an_alternative_file_is_named_by_its_digest_and_reads_back_exactly():
    setup = _study()
    arm = _overriding(setup)[0]
    inline = setup.overrides.inline(arm)
    name, data = _file(setup, arm)
    assert name == f"alternatives/{hashlib.sha256(data).hexdigest()}.json"
    assert data.endswith(b"\n") and b"\n " not in data
    saved = json.loads(data)
    assert saved["schema_version"] == ARM_FILE_SCHEMA_VERSION
    assert saved["alternative_id"] == arm
    # Shared records sit in the file's own tables, each stored once.
    assert all(
        isinstance(row["law"]["segments"][0]["kernel"], str)
        for row in saved["overrides"]
    )
    assert len(saved["mark_kernels"]) < len(inline)
    assert read_arm_file(data, arm, name) == inline


def _rewritten(setup, arm, change):
    """The alternative's file after ``change`` to its decoded content, named
    by its new digest."""
    saved = json.loads(_file(setup, arm)[1])
    change(saved)
    data = (canonical_json(saved) + "\n").encode()
    return arm_file_name(data), data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda saved: saved.update(schema_version=0), "schema version"),
        (lambda saved: saved.update(schema_version=True), "schema version"),
        (
            lambda saved: saved.update(schema_version=float(ARM_FILE_SCHEMA_VERSION)),
            "schema version",
        ),
        (lambda saved: saved.pop("schema_version"), "schema version"),
        (lambda saved: saved.update(alternative_id="another"), "does not hold"),
        (lambda saved: saved.update(unmodelled=1), "unexpected fields"),
        (lambda saved: saved.update(overrides={}), "must be a list of objects"),
        (lambda saved: saved.pop("texts"), "lacks its texts table"),
        (
            lambda saved: saved["texts"].update({canonical_json_sha256("x"): "x"}),
            "no law uses",
        ),
        (
            lambda saved: saved["mark_kernels"].update(
                {next(iter(saved["mark_kernels"])): {"changed": 1}}
            ),
            "keys differ from their digests",
        ),
        (
            lambda saved: saved["overrides"][0]["law"]["segments"][0].update(
                kernel="0" * 64
            ),
            "refers to no mark_kernels entry",
        ),
    ],
    ids=[
        "schema",
        "schema-true",
        "schema-float",
        "schema-absent",
        "alternative",
        "extra-field",
        "overrides-object",
        "missing-table",
        "unused-entry",
        "wrong-key",
        "dangling-key",
    ],
)
def test_a_malformed_alternative_file_is_refused(change, message):
    setup = _study()
    arm = _overriding(setup)[0]
    name, data = _rewritten(setup, arm, change)
    with pytest.raises(ValueError, match=message):
        read_arm_file(data, arm, name)


def test_an_alternative_file_must_be_named_by_its_digest():
    setup = _study()
    arm = _overriding(setup)[0]
    _, data = _file(setup, arm)
    with pytest.raises(ValueError, match="not named by the SHA-256"):
        read_arm_file(data, arm, "alternatives/" + "0" * 64 + ".json")


def test_stored_overrides_equal_the_held_ones_and_read_an_arm_when_asked():
    setup = _study()
    arms = _overriding(setup)
    files = dict(_file(setup, arm) for arm in arms)
    names = {arm: arm_file_name(files[name]) for arm, name in zip(arms, files)}
    stored = StoredOverrides(files, names)
    assert stored == setup.overrides and setup.overrides == stored
    assert stored.arms() == tuple(arms)
    assert stored.file_names() == names
    assert set(stored.stored()) == set(files)
    for arm in arms:
        assert stored.of(arm) == setup.overrides.of(arm)
        assert stored.keys(arm) == setup.overrides.keys(arm)
    # An arm is read again after another has replaced it, or after a release.
    first = arms[0]
    state_id = setup.overrides.of(first)[0].state_id
    assert stored.laws(first, state_id) == setup.overrides.laws(first, state_id)
    kept = stored.of(first)
    assert stored.of(first) is kept
    stored.release()
    assert stored.of(first) == kept and stored.of(first) is not kept
    assert stored.keys(first) == setup.overrides.keys(first)
    # What ``inline`` gives is the caller's own: changing it changes nothing
    # the store reads later.
    plain = stored.inline(first)
    assert plain == setup.overrides.inline(first)
    plain[0]["law"]["origin_id"] = "changed"
    plain.clear()
    stored.release()
    assert stored.inline(first) == setup.overrides.inline(first)
    assert stored.of(first) == setup.overrides.of(first)
    assert stored.of("absent") == () and stored.inline("absent") == []
    setup.overrides.release()
    with pytest.raises(ValueError, match="alternative files are missing"):
        StoredOverrides({}, names)


def test_a_stored_override_that_is_not_as_it_is_written_is_refused():
    """A file must hold each law exactly as its validated form is written, so
    what the digest reads is what the run compiles."""
    setup = _study()
    arm = _overriding(setup)[0]

    def integer_rate(saved):
        # A whole-number rate written as an integer validates to a float.
        saved["overrides"][0]["law"]["reference_annual_rate"] = 3
        saved["overrides"][0]["law"].pop("parameter_rank", None)

    name, data = _rewritten(setup, arm, integer_rate)
    stored = StoredOverrides({name: data}, {arm: name})
    assert stored.inline(arm)
    with pytest.raises(ValueError, match="not as it is written"):
        stored.of(arm)


def _recorded_reads(monkeypatch):
    """The arm files validated from here on, in order."""
    read = []
    validated = law_overrides._validated

    def recorded(rows, where):
        read.append(where)
        return validated(rows, where)

    monkeypatch.setattr(law_overrides, "_validated", recorded)
    return read


def test_a_stored_arm_is_checked_against_its_setup_whenever_it_is_read(
    tmp_path, monkeypatch
):
    setup = _study()
    arms = _overriding(setup)
    faulty, sound = arms[0], arms[1]
    rows = setup.overrides.inline(faulty)
    # The faulty arm's file is well formed, but overrides one law twice.
    data = arm_file(faulty, [*rows, rows[0]])
    files = {arm_file_name(data): data, **dict(_file(setup, arm) for arm in arms[1:])}
    names = {faulty: arm_file_name(data)} | {
        arm: _file(setup, arm)[0] for arm in arms[1:]
    }
    stored = StoredOverrides(files, names)
    # The files alone are valid overrides; only a setup can refuse them.
    assert len(stored.of(faulty)) == len(rows) + 1
    read = _recorded_reads(monkeypatch)
    # Nothing is read when the setup is made, so it is made.
    bound = replace(setup, overrides=stored)
    assert read == []
    assert bound.overrides.of(sound) == setup.overrides.of(sound)
    state_id = rows[0]["state_id"]
    for attempt in (
        lambda: bound.overrides.of(faulty),
        lambda: bound.overrides.laws(faulty, state_id),
        lambda: compile_incident_state(bound, state_id, faulty),
        bound.overrides.read_every_arm,
    ):
        with pytest.raises(ValueError, match="state/origin overrides must be unique"):
            attempt()
    # A setup with such an arm is not built: the build reads every arm.
    with pytest.raises(ValueError, match="state/origin overrides must be unique"):
        build_setup(destination=tmp_path / "setup", setup=bound, code_commit="fixture")
    assert not (tmp_path / "setup").exists()


def test_every_arm_is_read_once_to_load_a_setup_and_no_arm_is_kept(
    tmp_path, monkeypatch
):
    setup = _study()
    build_setup(destination=tmp_path / "setup", setup=setup, code_commit="fixture")
    read = _recorded_reads(monkeypatch)
    package = load_setup(tmp_path / "setup")
    assert isinstance(package.overrides, StoredOverrides)
    names = package.overrides.file_names()
    # Once each, whether or not the plan requires the arm.
    assert sorted(read) == sorted(names.values())
    assert package.overrides._kept is None
    package.overrides.read_every_arm()
    assert len(read) == len(names)
    # The load can hand over the states it compiles, so a run that takes them
    # (``StateLayout.loading``) reads each arm once too and compiles nothing
    # again.
    read.clear()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with StateLayout.loading(tmp_path / "setup", scratch) as layout:
        assert sorted(read) == sorted(names.values())
        assert list(layout.keys) == list(layout.package.execution.required_arms)
        (directory,) = scratch.iterdir()
        assert directory.name.startswith(".compiled-")
        assert len(list(directory.iterdir())) == sum(map(len, layout.keys.values()))
        for arm, keys in layout.keys.items():
            for key, conditional_key in keys.items():
                assert layout.compiled(arm, key).conditional_key == conditional_key
        assert len(read) == len(names)
        assert layout.package == package
    assert list(scratch.iterdir()) == []
    assert layout.keys == StateLayout.keys_of(package)
    # A scratch directory that does not exist yet is not made: the states go
    # to the nearest directory above it that does.
    with StateLayout(package, scratch / "absent" / "run"):
        assert [path.name[:10] for path in scratch.iterdir()] == [".compiled-"]
    assert list(scratch.iterdir()) == []
    # ``setup inspect`` counts the workload from the same single read.
    read.clear()
    inspected, workload = load_with_workload(tmp_path / "setup")
    assert sorted(read) == sorted(names.values())
    assert inspected == package and workload == setup_workload(package)


def _stored(setup, directory):
    """``setup`` with its overrides written to ``directory``, as an import
    hands a setup to the build."""
    writer = OverrideWriter(directory)
    for arm in setup.overrides.arms():
        writer.add(arm, setup.overrides.of(arm))
    return replace(setup, overrides=writer.overrides())


def test_building_a_setup_with_stored_overrides_reads_its_arms_as_loading_does(
    tmp_path, monkeypatch
):
    """The saved package names the arm files the composed setup already
    stores, so the two are compared by those names: the build reads an arm
    only as loading the package reads it, to validate and check it."""
    held = _study()
    stored = _stored(held, tmp_path / "arms")
    read = []
    read_arm_file = law_overrides.read_arm_file

    def recorded(data, alternative_id, name):
        read.append(name)
        return read_arm_file(data, alternative_id, name)

    monkeypatch.setattr(law_overrides, "read_arm_file", recorded)
    package = build_setup(
        destination=tmp_path / "setup", setup=stored, code_commit="fixture"
    )
    assert isinstance(package.overrides, StoredOverrides)
    building = sorted(read)
    assert set(building) == set(package.overrides.file_names().values())
    read.clear()
    load_setup(tmp_path / "setup")
    assert sorted(read) == building
    monkeypatch.undo()
    # It is the package the same setup gives when it holds its overrides.
    other = build_setup(
        destination=tmp_path / "held", setup=held, code_commit="fixture"
    )
    assert isinstance(other.overrides, StoredOverrides)
    for name in ("setup.json", "execution-plan.json"):
        assert (tmp_path / "setup" / name).read_bytes() == (
            tmp_path / "held" / name
        ).read_bytes()
    assert package.scientific_sha256 == other.scientific_sha256
    assert package.overrides.file_names() == other.overrides.file_names()


def test_stored_setups_compare_by_their_arm_files_names(tmp_path):
    held = _study()
    stored = _stored(held, tmp_path / "arms")
    again = _stored(held, tmp_path / "again")
    assert stored_inputs_sha256(stored) == stored_inputs_sha256(again)
    assert stored.scientific_sha256 == again.scientific_sha256
    # It is not the setup's identity, and a setup that holds its overrides
    # has no files to name.
    assert stored_inputs_sha256(stored) != stored.scientific_sha256
    with pytest.raises(ValueError, match="stored overrides"):
        stored_inputs_sha256(held)
    # Another arm file, or any other input, changes it as it changes the
    # scientific digest.
    arm = _overriding(held)[0]
    first, *rest = held.overrides.of(arm)
    changed = replace(
        held,
        overrides=HeldOverrides(
            {
                **{name: held.overrides.of(name) for name in held.overrides.arms()},
                arm: (
                    LawOverride(
                        state_id=first.state_id,
                        law=first.law.model_copy(
                            update={
                                "reference_annual_rate": first.law.reference_annual_rate
                                * 2
                            }
                        ),
                    ),
                    *rest,
                ),
            }
        ),
    )
    other_arm = _stored(changed, tmp_path / "changed")
    assert other_arm.scientific_sha256 != stored.scientific_sha256
    assert stored_inputs_sha256(other_arm) != stored_inputs_sha256(stored)
    reweighted = replace(
        stored,
        joint_states=tuple(
            state.model_copy(update={"weight": weight})
            for state, weight in zip(
                stored.joint_states,
                reversed([state.weight for state in stored.joint_states]),
                strict=True,
            )
        ),
    )
    if reweighted.scientific_sha256 != stored.scientific_sha256:
        assert stored_inputs_sha256(reweighted) != stored_inputs_sha256(stored)


def test_a_writer_holds_overrides_or_writes_each_arm_as_it_is_given(tmp_path):
    setup = _study()
    arms = _overriding(setup)
    held, written = OverrideWriter(), OverrideWriter(tmp_path / "arms")
    for arm in arms:
        held.add(arm, setup.overrides.of(arm))
        written.add(arm, setup.overrides.of(arm))
        # The arm's file exists as soon as the arm is given.
        assert len(list((tmp_path / "arms" / "alternatives").iterdir())) == len(
            written.overrides().arms()
        )
    held.add("empty", ())
    written.add("empty", ())
    assert isinstance(held.overrides(), HeldOverrides)
    stored = written.overrides()
    assert isinstance(stored, StoredOverrides)
    assert held.overrides() == stored == setup.overrides
    assert "empty" not in stored.arms() and "empty" not in held.overrides().arms()
    for name in stored.file_names().values():
        assert name == arm_file_name((tmp_path / "arms" / name).read_bytes())
    with pytest.raises(ValueError, match="written twice"):
        written.add(arms[0], setup.overrides.of(arms[0]))


def test_a_setup_validates_what_a_writer_was_given(tmp_path):
    """The writer validates nothing. The setup that takes its overrides does:
    held ones when it is made, stored ones when their arm is read."""
    setup = _study()
    arm = _overriding(setup)[0]
    row = setup.overrides.of(arm)[0]
    constructed = LawOverride.model_construct(
        state_id=row.state_id,
        law=row.law.model_copy(update={"reference_annual_rate": -1.0}),
    )
    held, written = OverrideWriter(), OverrideWriter(tmp_path / "arms")
    for name in setup.overrides.arms():
        rows = (constructed,) if name == arm else setup.overrides.of(name)
        held.add(name, rows)
        written.add(name, rows)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        replace(setup, overrides=held.overrides())
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        # Stored overrides are validated when their arm is read, at the
        # latest when the setup is built or loaded.
        replace(setup, overrides=written.overrides()).overrides.of(arm)
    with pytest.raises(TypeError, match="LawOverride records"):
        OverrideWriter().add(arm, cast(Any, (row.model_dump(mode="json"),)))


def test_a_stored_override_must_be_written_as_its_validated_form(tmp_path):
    """A writer writes an override as it is given. One that validates to
    another form, as padded text validates to trimmed text, is refused when
    its arm is read, never silently changed."""
    setup = _study()
    arm = _overriding(setup)[0]
    row = setup.overrides.of(arm)[0]
    padded = row.law.response.model_copy(
        update={"held_fixed": f" {row.law.response.held_fixed} "}
    )
    copied = LawOverride.model_construct(
        state_id=row.state_id, law=row.law.model_copy(update={"response": padded})
    )
    written = OverrideWriter(tmp_path / "arms")
    written.add(arm, (copied,))
    with pytest.raises(ValueError, match="not as it is written"):
        replace(setup, overrides=written.overrides()).overrides.of(arm)


def test_overrides_must_belong_to_a_declared_alternative_and_be_unique():
    setup = _study()
    arm = _overriding(setup)[0]
    rows = setup.overrides.of(arm)
    held = {name: setup.overrides.of(name) for name in setup.overrides.arms()}
    with pytest.raises(ValidationError, match="undeclared alternatives"):
        replace(setup, overrides=HeldOverrides({**held, "undeclared": rows}))
    with pytest.raises(ValidationError, match="state/origin overrides must be unique"):
        replace(setup, overrides=HeldOverrides({**held, arm: (*rows, rows[0])}))
    with pytest.raises(TypeError, match="must be LawOverrides"):
        replace(setup, overrides={arm: rows})
    with pytest.raises(TypeError, match="LawOverride models"):
        HeldOverrides({arm: (rows[0].law,)})


def test_an_alternative_no_longer_takes_overrides():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Alternative(
            alternative_id="arm",
            purpose="scientific_alternative",
            description="An arm.",
            overrides=(),  # ty: ignore[unknown-argument]
        )


def test_a_built_setup_stores_each_arm_in_its_own_file_and_reads_it_back(tmp_path):
    setup = _study()
    arms = _overriding(setup)
    package = build_setup(
        destination=tmp_path / "setup", setup=setup, code_commit="fixture"
    )
    saved = json.loads((tmp_path / "setup" / "setup.json").read_bytes())
    named = {
        row["alternative_id"]: row["overrides"]
        for row in saved["alternatives"]
        if row["overrides"] is not None
    }
    assert list(named) == arms
    on_disk = sorted(
        path.relative_to(tmp_path / "setup").as_posix()
        for path in (tmp_path / "setup" / "alternatives").iterdir()
    )
    assert on_disk == sorted(named.values())
    assert set(named.values()) <= set(package.manifest.files)
    for name in named.values():
        assert name == arm_file_name((tmp_path / "setup" / name).read_bytes())
    for built in (package, load_setup(tmp_path / "setup")):
        assert isinstance(built.overrides, StoredOverrides)
        assert built.overrides.file_names() == named
        assert built.overrides == setup.overrides
        assert built == setup
        assert built.scientific_sha256 == setup.scientific_sha256
        assert built.scientific_sha256 == scientific_digest(setup)
    # A stored setup is saved again from its files, not re-encoded.
    text, files = saved_setup(package)
    assert text + "\n" == (tmp_path / "setup" / "setup.json").read_text()
    assert set(files) == set(named.values())


def test_a_setup_compiles_alike_from_held_and_stored_overrides(tmp_path):
    setup = _study()
    package = build_setup(
        destination=tmp_path / "setup", setup=setup, code_commit="fixture"
    )
    assert declared_view_identifiers(package) == declared_view_identifiers(setup)
    for arm in ("reference", *_overriding(setup)):
        for state in setup.joint_states:
            held = compile_incident_state(setup, state.state_id, arm)
            stored = compile_incident_state(package, state.state_id, arm)
            assert (stored.conditional_key, stored.law_sha256) == (
                held.conditional_key,
                held.law_sha256,
            )


def test_an_alternative_file_edited_or_missing_after_a_build_is_refused(tmp_path):
    setup = _study()
    arm = _overriding(setup)[0]
    package = build_setup(
        destination=tmp_path / "setup", setup=setup, code_commit="fixture"
    )
    assert isinstance(package.overrides, StoredOverrides)
    name = package.overrides.file_names()[arm]
    path = tmp_path / "setup" / name
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"alternative_id"', b'"alternative_id" '))
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        package.overrides.inline(arm)
    with pytest.raises(ValueError, match="setup checksum mismatch"):
        load_setup(tmp_path / "setup")
    path.unlink()
    with pytest.raises(FileNotFoundError):
        load_setup(tmp_path / "setup")
    path.write_bytes(original)
    assert load_setup(tmp_path / "setup") == setup
    # A file the manifest lists but no alternative names is not a payload.
    stray = tmp_path / "setup" / "alternatives" / "stray.json"
    stray.write_bytes(b"{}")
    with pytest.raises(ValueError, match="untracked files"):
        load_setup(tmp_path / "setup")


def test_stored_files_are_the_arms_a_setup_names_and_no_others():
    setup = _study()
    arms = _overriding(setup)
    text, files = saved_setup(setup)
    assert isinstance(files, StoredFiles)
    saved = json.loads(text)
    assert [
        row["alternative_id"] for row in saved["alternatives"] if row["overrides"]
    ] == arms
    assert all(
        row["overrides"] is None
        for row in saved["alternatives"]
        if row["alternative_id"] not in arms
    )
    assert "overrides" not in saved and "mark_kernels" in saved
