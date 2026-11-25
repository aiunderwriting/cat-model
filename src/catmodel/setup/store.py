"""Build, load and validate immutable run setups."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Callable, Mapping
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from catmodel.contracts._setup_codec import (
    disclosure_prose,
    saved_setup,
    stored_inputs_sha256,
)
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.law_overrides import StoredOverrides
from catmodel.contracts.run_setup import RunSetup, SetupManifest
from catmodel.evidence.source_bundle import normalize_source_bundle_path
from catmodel.json_io import require_marker, strict_json_loads
from catmodel.output_paths import require_new_evidence_output
from catmodel.simulation.incident import (
    CompiledIncidentState,
    compile_incident_state,
)
from catmodel.stored_files import DiskFile, Stored, StoredFiles
from catmodel.worktree import resolve_code_provenance

MANIFEST = "setup-manifest.json"
OMITTED_EVIDENCE = "omitted-evidence.json"
"""The record, beside the manifest, of the files a copy of a setup leaves out
(``omitted_evidence_record``). A whole setup has none."""
_PAYLOADS = {"setup.json", "execution-plan.json"}

_CAPTURED = "evidence/captured/"
# What a study and a collection write about themselves, which holds no prompt
# and no answer: the study's plan, forecast and fitted trend, and a
# collection's instrument, plan, call settings and parser identities.
_DESCRIPTIVE_FILES = frozenset(
    {
        "public-study.json",
        "forecast.json",
        "public-frontier.json",
        "trend-posterior.json",
        "contextual-instrument.json",
        "collection-plan.json",
    }
)
_DESCRIPTIVE_DIRECTORIES = frozenset({"acquisition", "parser-protocols"})


def holds_captured_answers(name: str) -> bool:
    """Whether the setup file ``name`` holds what a collection captured from
    its respondents: a prompt as sent, a response, or the answers parsed from
    the responses.

    Those are the files under ``evidence/captured/`` other than a source
    bundle's and the ones a study and a collection write about themselves
    (``_DESCRIPTIVE_FILES``, ``_DESCRIPTIVE_DIRECTORIES``). Every file of a
    ``calls`` directory holds a prompt or a response, and a file this function
    does not know is taken to hold answers.
    """
    if not name.startswith(_CAPTURED):
        return False
    *directories, file = name.removeprefix(_CAPTURED).split("/")
    if "source-bundle" in directories:
        return False
    if "calls" in directories:
        return True
    return not (
        file in _DESCRIPTIVE_FILES
        or (bool(directories) and directories[-1] in _DESCRIPTIVE_DIRECTORIES)
    )


def captured_answer_files(manifest: SetupManifest) -> dict[str, str]:
    """The files of a setup that hold captured answers
    (``holds_captured_answers``), each with the SHA-256 its manifest records."""
    return {
        name: digest
        for name, digest in sorted(manifest.files.items())
        if holds_captured_answers(name)
    }


def omitted_evidence_record(
    manifest_sha256: str, files: Mapping[str, str]
) -> dict[str, object]:
    """What ``omitted-evidence.json`` holds for a copy of the setup whose
    manifest has ``manifest_sha256``: each file the copy leaves out, by its
    path in the setup and the SHA-256 the manifest records for it."""
    return {
        "schema_version": 1,
        "setup_manifest_sha256": manifest_sha256,
        "files": dict(sorted(files.items())),
    }


def omitted_files(package: RunSetup) -> dict[str, str]:
    """The files a loaded copy of a setup leaves out, each with the SHA-256
    its manifest records; empty for a whole setup."""
    return {
        name: digest
        for name, digest in sorted(package.manifest.files.items())
        if name not in package.files
    }


def numerical_platform() -> str:
    """The operating system, machine architecture and C library, such as
    ``Linux-x86_64-glibc2.39``: the parts of a platform that can change a
    result's last bits.

    The kernel release is left out, so the same code on like machines records
    the same identities, and a kernel update changes none.
    """
    return "-".join(
        part
        for part in (
            platform.system(),
            platform.machine(),
            "".join(platform.libc_ver()),
        )
        if part
    )


def setup_code_provenance(code_commit: str | None) -> dict[str, object]:
    """Retain running code and numerical environment without changing the law.

    The environment names the Python implementation and version, the platform
    (``numerical_platform``) and the numerical packages. The interpreter's
    build date and compiler are left out with the kernel release: they say
    when and how a binary was built, and setups, captures and results carry
    this record inside their identities.
    """
    return {
        **asdict(resolve_code_provenance(code_commit)),
        "numerical_environment": {
            "python": (
                f"{platform.python_implementation()} {platform.python_version()}"
            ),
            "platform": numerical_platform(),
            "packages": {
                name: version(name)
                for name in ("catmodel", "numpy", "scipy", "pydantic")
            },
        },
    }


def _required_control_closure(
    spec: RunSetup, required_arms: tuple[str, ...]
) -> frozenset[str]:
    """Follow every selected arm's declared matched control."""
    alternatives = {arm.alternative_id: arm for arm in spec.alternatives}
    closure = set(required_arms)
    pending = list(required_arms)
    while pending:
        arm = pending.pop()
        alternative = alternatives.get(arm)
        if alternative is None:
            if arm != "reference":
                raise ValueError(f"unknown required setup arm {arm!r}")
            continue
        control = alternative.matched_control_id
        if control is not None and control not in closure:
            closure.add(control)
            pending.append(control)
    return frozenset(closure)


def required_state_ids(plan: ExecutionPlan, arm: str, declared_ids) -> tuple[str, ...]:
    declared = tuple(declared_ids)
    selected = plan.required_states(arm, declared)
    if not set(selected) <= set(declared):
        raise ValueError(
            "frozen arm state requirements must name distinct declared joint states"
        )
    return tuple(selected)


type CompiledSink = Callable[[str, str, CompiledIncidentState], None]
"""Takes each state compiled while a setup is checked: the arm, the state ID
and the compiled state."""


def validate_setup(
    package: RunSetup,
    compiled: CompiledSink | None = None,
    *,
    omitted: Mapping[str, str] | None = None,
) -> None:
    """Check a saved package: its files against its manifest, its plan
    against its inputs, and that every state the plan requires of an arm
    compiles. Each compiled state is given to ``compiled``, so a caller that
    needs them does not compile them again. Every alternative's overrides are
    read once, and so validated and checked against the setup: a required
    arm's as its states compile, the others' after.

    ``omitted`` names the files a copy leaves out, each with its SHA-256.
    Only a file that holds captured answers may be left out. It must be in
    the manifest with that digest, and the evidence it carries is then present
    by that digest and not by its bytes."""
    inputs = package
    files = package.files
    manifest = package.manifest
    omitted = dict(omitted or {})
    if not all(map(holds_captured_answers, omitted)):
        raise ValueError(
            "a setup copy may leave out only captured prompts, responses and answers"
        )
    names = set(files) - {MANIFEST}
    arms = (
        set(inputs.overrides.file_names().values())
        if isinstance(inputs.overrides, StoredOverrides)
        else set()
    )
    if {name for name in names if not name.startswith("evidence/")} != _PAYLOADS | arms:
        raise ValueError("setup has missing or unexpected payloads")
    if names & set(omitted) or set(manifest.files) != names | set(omitted):
        raise ValueError("setup manifest does not cover its exact closure")
    for name, digest in manifest.files.items():
        held = omitted[name] if name in omitted else files.sha256(name)
        if held != digest:
            raise ValueError(f"setup checksum mismatch: {name}")
    selected_digests = {
        files.sha256(name) for name in files if name.startswith("evidence/")
    } | set(omitted.values())
    for evidence in package.evidence.records:
        required = {
            evidence.original_sha256,
            evidence.selected_sha256,
            *evidence.raw_attempt_sha256,
        }
        if not required <= selected_digests:
            raise ValueError(
                "incident specification evidence or raw attempts lack their captured original bytes"
            )
    missing_controls = _required_control_closure(
        inputs, package.execution.required_arms
    ) - set(package.execution.required_arms)
    if missing_controls:
        raise ValueError(
            f"execution plan omits required matched controls: {sorted(missing_controls)}"
        )
    for context in package.conditional_contexts:
        if not set(context.required_arms) <= set(package.execution.required_arms):
            raise ValueError("execution plan omits a conditional context arm")
        if not {"welfare", f"lab_loss::{context.lab_id}"} <= set(
            package.execution.requested_views
        ):
            raise ValueError("execution plan omits a conditional peer observable")
        for kernel in context.kernels:
            selected = set(
                required_state_ids(
                    package.execution,
                    kernel.arm_id,
                    (state.state_id for state in package.joint_states),
                )
            )
            if not {row.state_id for row in kernel.components} <= selected:
                raise ValueError("execution plan omits a conditional kernel component")
    alternatives = {row.alternative_id: row for row in inputs.alternatives}
    for arm in package.execution.required_arms:
        selected = required_state_ids(
            package.execution,
            arm,
            (state.state_id for state in inputs.joint_states),
        )
        alternative = alternatives.get(arm)
        weights = None if alternative is None else alternative.state_weights
        positive = {
            state.state_id
            for state in inputs.joint_states
            if (state.weight if weights is None else weights[state.state_id]) > 0
        }
        if not positive <= set(selected):
            raise ValueError(
                "positive arm target weights require frozen conditional state support"
            )
        for state_id in selected:
            state = compile_incident_state(inputs, state_id, arm)
            if compiled is not None:
                compiled(arm, state_id, state)
    inputs.overrides.read_every_arm()


def build_setup(
    *,
    destination: Path,
    setup: RunSetup,
    code_commit: str | None = None,
) -> RunSetup:
    """Freeze one composed setup and its selected original bytes.

    The files are decoded once, checked to reproduce the composed inputs and
    validated before any is written; the decoded setup, attached to the written
    files, is returned. The composed setup is released before it is decoded, so
    a caller that passes it inline, keeping no reference, never holds both in
    memory. Evidence left on disk (``StoredFiles``) is copied from there one
    file at a time, each copy checked against the digest the manifest records.
    """
    destination = destination.expanduser().absolute()
    require_new_evidence_output(destination)
    code = setup_code_provenance(code_commit)
    if setup._package is None:
        text, arms = saved_setup(setup)
        payloads: Mapping[str, bytes] = StoredFiles.merged(
            {
                "setup.json": (text + "\n").encode(),
                "execution-plan.json": (
                    setup.execution.model_dump_json(indent=2) + "\n"
                ).encode(),
            },
            arms,
        )
    else:
        payloads = setup.files.selected(
            lambda name: name != MANIFEST and not name.startswith("evidence/")
        )
    evidence = StoredFiles(setup.evidence.files)
    for name in evidence:
        if normalize_source_bundle_path(name) != name:
            raise ValueError("invalid selected evidence path")
    files = StoredFiles.merged(payloads, evidence.prefixed("evidence/"))
    manifest = SetupManifest(
        setup_id=setup.setup_id,
        code_provenance=code,
        files={name: files.sha256(name) for name in files},
    )
    files = StoredFiles.merged(
        files, {MANIFEST: (manifest.model_dump_json(indent=2) + "\n").encode()}
    )
    # The scientific digest leaves the disclosure prose out, so it is compared
    # beside it. A composed setup whose overrides are stored saves the same
    # arm files, and a decoded package's are always stored: the two are then
    # compared with each arm entered by its file's name, which fixes its
    # bytes, and no arm is read for the comparison.
    stored = isinstance(setup.overrides, StoredOverrides)

    def inputs(row: RunSetup) -> tuple[object, ...]:
        return (
            stored_inputs_sha256(row) if stored else row.scientific_sha256,
            disclosure_prose(row),
            row.execution.digest(),
        )

    composed = inputs(setup)
    del setup
    package = RunSetup._from_package_files(destination, files)
    if inputs(package) != composed:
        raise ValueError("saved package differs from the composed setup inputs")
    validate_setup(package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".setup-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "package"
        stage.mkdir()
        for name in files:
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            files.write(name, path)
        stage.rename(destination)
    # The package reads its files where they were written from now on, not
    # where its evidence came from, and holds none of them in memory.
    package._attach(
        destination,
        StoredFiles.under(
            destination,
            files,
            {**manifest.files, MANIFEST: files.sha256(MANIFEST)},
        ),
        manifest,
    )
    return package


def _recorded_omissions(
    root: Path, manifest: SetupManifest, manifest_bytes: bytes
) -> dict[str, str]:
    """The files the copy at ``root`` records as left out
    (``omitted-evidence.json``), each with its SHA-256; empty without the
    record. The record must list every file of the manifest that holds
    captured answers, and no other, so a copy holds all of them or none."""
    path = root / OMITTED_EVIDENCE
    if path.is_symlink():
        raise ValueError("setup rejects symlinks")
    if not path.is_file():
        return {}
    payload = strict_json_loads(path.read_bytes())
    require_marker(
        payload,
        "schema_version",
        1,
        f"{OMITTED_EVIDENCE} schema version",
        "copy the setup again with the current software",
    )
    expected = captured_answer_files(manifest)
    if not expected or payload != omitted_evidence_record(
        hashlib.sha256(manifest_bytes).hexdigest(), expected
    ):
        raise ValueError(
            f"{OMITTED_EVIDENCE} must name its setup's manifest and list exactly "
            "the captured prompts, responses and answers, each with the SHA-256 "
            "the manifest records"
        )
    return expected


def load_setup(root: Path, *, compiled: CompiledSink | None = None) -> RunSetup:
    """Load and check the setup saved at ``root`` (``validate_setup``, which
    gives ``compiled`` each state it compiles).

    A copy that leaves its captured answers out, as an export's does by
    default, says so in ``omitted-evidence.json`` (``_recorded_omissions``).
    It loads without those files, which must all be absent; every file it
    holds is checked as a whole setup's is, and ``omitted_files`` gives what
    it lacks. A file that is missing without that record is refused."""
    root = root.expanduser().resolve()
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("setup rejects manifest symlinks")
    data = manifest_path.read_bytes()
    manifest = SetupManifest.from_saved(strict_json_loads(data))
    omitted = _recorded_omissions(root, manifest, data)
    files: dict[str, Stored] = {MANIFEST: data}
    for name, digest in manifest.files.items():
        if normalize_source_bundle_path(name) != name or name == MANIFEST:
            raise ValueError("invalid setup path")
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("setup rejects escaping paths and symlinks")
        if name in omitted:
            if path.exists():
                raise ValueError(f"setup holds a file it records as left out: {name}")
            continue
        # Checked here and again whenever it is read; never held.
        entry = DiskFile(path, digest)
        try:
            entry.check()
        except ValueError:
            raise ValueError(f"setup checksum mismatch: {name}") from None
        files[name] = entry
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("setup rejects symlinks")
    tracked = set(files) | ({OMITTED_EVIDENCE} if omitted else set())
    if {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    } != tracked:
        raise ValueError("setup contains untracked files")
    package = RunSetup._from_package_files(root, StoredFiles(files))
    validate_setup(package, compiled, omitted=omitted)
    return package
