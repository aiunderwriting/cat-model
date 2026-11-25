"""The complete authored setup for a run, independent of its storage location."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import cache, cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from catmodel.contracts.capability_evidence import UTCInstant
from catmodel.contracts.components import ContractTerms
from catmodel.contracts.conditional import ConditionalContext
from catmodel.contracts.incident import (
    Alternative,
    CapabilityScale,
    DeploymentBinding,
    Entity,
    Evidence,
    ExecutionPlan,
    JointState,
    Population,
    ReportPopulationBinding,
)
from catmodel.contracts.law_overrides import (
    HeldOverrides,
    LawOverrides,
    StoredOverrides,
)
from catmodel.contracts.primitives import (
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import require_marker, strict_json_loads
from catmodel.stored_files import StoredFiles


class RunScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_id: RequiredIdentifier
    title: RequiredString
    period_start: UTCInstant
    period_end: UTCInstant
    forecast_as_of: UTCInstant
    data_cutoff: UTCInstant
    availability_cutoff: UTCInstant
    information_date: UTCInstant
    capability_scales: tuple[CapabilityScale, ...] = Field(min_length=1)
    deployments: tuple[DeploymentBinding, ...] = ()
    entities: tuple[Entity, ...] = Field(min_length=1)
    populations: tuple[Population, ...] = Field(min_length=1)
    report_population_bindings: tuple[ReportPopulationBinding, ...] = ()


@dataclass(frozen=True, kw_only=True)
class EvidenceSet:
    """Captured observations, elicited judgments, provenance and original bytes.

    ``files`` is held as ``StoredFiles``: a file given on disk stays there and
    is read, and checked against its digest, when it is looked up.
    """

    records: tuple[Evidence, ...] = ()
    files: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        records = TypeAdapter(tuple[Evidence, ...]).validate_python(self.records)
        files = self.files
        if not isinstance(files, StoredFiles):
            files = dict(files)
            if any(
                not isinstance(name, str) or not isinstance(data, bytes)
                for name, data in files.items()
            ):
                raise ValueError(
                    "selected evidence must have string paths and original bytes"
                )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "files", StoredFiles(files))


class AssumptionSet(BaseModel):
    """Standalone independence statements; local rationale stays with its input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    independence: tuple[RequiredString, ...] = ()


class SetupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    setup_id: RequiredIdentifier
    code_provenance: dict[str, object]
    files: dict[str, Sha256Digest]

    @classmethod
    def from_saved(cls, payload: object) -> SetupManifest:
        """The manifest a saved file's decoded JSON holds, which states the
        current schema version."""
        require_marker(
            payload,
            "schema_version",
            1,
            "setup manifest schema version",
            "rebuild the setup with the current software or read it with its "
            "pinned producer software",
        )
        return cls.model_validate(payload)


def _revalidated[T: BaseModel](model: type[T], value: object) -> T:
    """Rebuild a typed section so constructed or copied values cannot skip validation."""
    if not isinstance(value, model):
        raise TypeError(
            f"setup section expected {model.__name__}, got {type(value).__name__}"
        )
    return model.model_validate(value.model_dump(mode="python", warnings=False))


def _revalidated_field(name: str, value: object) -> Any:
    """A RunSetup field that is not a section, rebuilt from its dump so
    copied or constructed nested models revalidate."""
    adapter = field_adapters()[name]
    if isinstance(value, Mapping):
        value = dict(value)
    element = element_adapters().get(name)
    if element is not None and isinstance(value, tuple | list):
        # One element at a time, so a large field is never held as a whole
        # dump beside its validated copy; the field's own validator then
        # checks the tuple of validated models.
        return adapter.validate_python(
            tuple(
                element.validate_python(
                    element.dump_python(row, mode="python", warnings=False)
                )
                for row in value
            )
        )
    return adapter.validate_python(
        adapter.dump_python(value, mode="python", warnings=False)
    )


@dataclass(frozen=True)
class _SetupPackage:
    root: Path
    files: StoredFiles
    manifest: SetupManifest


@dataclass(frozen=True, kw_only=True)
class RunSetup:
    """Scope, evidence, standalone assumptions, executable inputs and execution.

    Joint states, conditional contexts, alternatives and contracts sit directly
    on the setup as resolved executable inputs. The laws an alternative puts in
    place of reference laws are held by alternative in ``overrides``
    (``LawOverrides``): in memory for an authored setup, and in each
    alternative's own file for a saved one, which reads one arm at a time.

    Construction revalidates every input, and nothing validated can change in
    place: every model and dataclass the setup holds is frozen, sequences are
    tuples and mappings are read-only at every depth (``Frozen``). Overrides
    held in memory are revalidated and checked against the other inputs too.
    Stored ones are not read at construction: an arm is validated, and checked
    against the other inputs, whenever it is read, a file is read only while
    it has the digest its name records, and loading or building a setup reads
    every arm (``store.validate_setup``). The scientific digest is therefore
    computed on first use and kept, and a saved package stays attached to
    exactly the inputs decoded from its files.
    """

    scope: RunScope
    evidence: EvidenceSet
    assumptions: AssumptionSet
    execution: ExecutionPlan
    setup_id: RequiredIdentifier
    limitations: tuple[RequiredString, ...]
    joint_states: Annotated[tuple[JointState, ...], Field(min_length=1)]
    conditional_contexts: tuple[ConditionalContext, ...] = ()
    alternatives: tuple[Alternative, ...] = ()
    contracts: tuple[ContractTerms, ...] = ()
    overrides: LawOverrides = field(default_factory=HeldOverrides)
    _package: _SetupPackage | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _revalidated(RunScope, self.scope))
        object.__setattr__(
            self, "assumptions", _revalidated(AssumptionSet, self.assumptions)
        )
        if not isinstance(self.evidence, EvidenceSet):
            raise TypeError("setup evidence must be an EvidenceSet")
        object.__setattr__(
            self,
            "evidence",
            EvidenceSet(
                records=tuple(
                    Evidence.model_validate(
                        row.model_dump(mode="python", warnings=False)
                    )
                    for row in self.evidence.records
                ),
                files=self.evidence.files,
            ),
        )
        object.__setattr__(
            self, "execution", _revalidated(ExecutionPlan, self.execution)
        )
        for name in field_adapters():
            object.__setattr__(
                self, name, _revalidated_field(name, getattr(self, name))
            )
        overrides = self.overrides
        if not isinstance(overrides, LawOverrides):
            raise TypeError("setup overrides must be LawOverrides")
        if isinstance(overrides, HeldOverrides):
            overrides = HeldOverrides.validated(
                {arm: overrides.of(arm) for arm in overrides.arms()}
            )
        self._check_inputs(overrides)

    def _check_inputs(self, overrides: LawOverrides) -> None:
        """Check the validated inputs against each other and put ``overrides``
        under their check."""
        from catmodel.contracts._setup_validation import (
            validate_setup_arms,
            validate_setup_inputs,
        )

        try:
            check = validate_setup_inputs(self)
            # Held overrides are checked here; stored ones whenever an arm
            # is read.
            object.__setattr__(self, "overrides", overrides.checked_by(check))
            validate_setup_arms(self)
        except ValueError as error:
            raise ValidationError.from_exception_data(
                "RunSetup",
                [
                    {
                        "type": "value_error",
                        "loc": (),
                        "input": {
                            "scope_id": self.scope.scope_id,
                            "setup_id": self.setup_id,
                        },
                        "ctx": {"error": error},
                    }
                ],
            ) from error

    def with_joint_states(self, joint_states: tuple[JointState, ...]) -> RunSetup:
        """This setup with other joint states and no saved package.

        Every other input is this setup's own, validated when it was made and
        frozen since, so only the new states are revalidated; the inputs are
        then checked against each other as at construction. A caller that
        makes many setups differing only in their states, as admitting each
        compiled law does (``contextual_node.admit_node``), validates the
        rest once.
        """
        derived = object.__new__(RunSetup)
        for row in fields(RunSetup):
            object.__setattr__(derived, row.name, getattr(self, row.name))
        object.__setattr__(derived, "_package", None)
        object.__setattr__(
            derived, "joint_states", _revalidated_field("joint_states", joint_states)
        )
        derived._check_inputs(self.overrides)
        return derived

    @cached_property
    def scientific_sha256(self) -> str:
        from catmodel.contracts._setup_codec import scientific_sha256

        # The inline encoding of a large setup runs to gigabytes.
        return scientific_sha256(self)

    def _saved_package(self) -> _SetupPackage:
        if self._package is None:
            raise ValueError("setup has no saved package")
        return self._package

    @property
    def root(self) -> Path:
        return self._saved_package().root

    @property
    def files(self) -> StoredFiles:
        """The saved package's files by name. Each stays on disk and is read,
        and checked against its digest, when it is looked up."""
        return self._saved_package().files

    @property
    def manifest(self) -> SetupManifest:
        return self._saved_package().manifest.model_copy(deep=True)

    @property
    def manifest_sha256(self) -> str:
        return self.files.sha256("setup-manifest.json")

    @property
    def execution_plan_sha256(self) -> str:
        if self._package is None:
            return self.execution.digest()
        return canonical_json_sha256(
            strict_json_loads(self.files["execution-plan.json"])
        )

    @classmethod
    def _from_package_files(cls, root: Path, files: Mapping[str, bytes]) -> RunSetup:
        """Decode a saved package's files, once, into a setup attached to them."""
        from catmodel.contracts._setup_codec import decode_saved_setup

        retained = StoredFiles(files)
        manifest = SetupManifest.from_saved(
            strict_json_loads(retained["setup-manifest.json"])
        )
        setup = decode_saved_setup(
            strict_json_loads(retained["setup.json"]),
            ExecutionPlan.model_validate(
                strict_json_loads(retained["execution-plan.json"])
            ),
            setup_id=manifest.setup_id,
            evidence_files=retained.unprefixed("evidence/"),
            files=retained,
        )
        setup._attach(root, retained, manifest)
        return setup

    def _attach(self, root: Path, files: StoredFiles, manifest: SetupManifest) -> None:
        """Attach the setup to its saved package: ``files``, and its evidence
        and stored overrides as they hold them."""
        object.__setattr__(self.evidence, "files", files.unprefixed("evidence/"))
        if isinstance(self.overrides, StoredOverrides):
            object.__setattr__(self, "overrides", self.overrides.reading(files))
        object.__setattr__(
            self,
            "_package",
            _SetupPackage(root=root.resolve(), files=files, manifest=manifest),
        )


_SECTIONS = frozenset({"scope", "evidence", "assumptions", "execution"})
# Fields the adapters below do not validate: the sections, and the overrides,
# whose store validates them (``LawOverrides``).
_NOT_ADAPTED = _SECTIONS | {"overrides"}


@cache
def element_adapters() -> Mapping[str, TypeAdapter[Any]]:
    """Validators for the models of every RunSetup field that is a tuple of
    models, by field."""
    hints = get_type_hints(RunSetup)
    result = {}
    for row in fields(RunSetup):
        if not row.init or row.name in _NOT_ADAPTED:
            continue
        hint = hints[row.name]
        arguments = get_args(hint)
        if (
            get_origin(hint) is tuple
            and len(arguments) == 2
            and arguments[1] is Ellipsis
            and isinstance(arguments[0], type)
            and issubclass(arguments[0], BaseModel)
        ):
            result[row.name] = TypeAdapter(arguments[0])
    return MappingProxyType(result)


@cache
def field_adapters() -> Mapping[str, TypeAdapter[Any]]:
    """Validators for every non-section RunSetup field, from its annotation."""
    hints = get_type_hints(RunSetup, include_extras=True)
    return MappingProxyType(
        {
            row.name: TypeAdapter(hints[row.name])
            for row in fields(RunSetup)
            if row.init and row.name not in _NOT_ADAPTED
        }
    )
