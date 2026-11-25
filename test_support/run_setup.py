"""Run setups with stipulated execution for unit fixtures."""

import dataclasses
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from pydantic import BaseModel

from catmodel.contracts._setup_codec import decode_scientific, encode_scientific
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.law_overrides import LawOverrides
from catmodel.contracts.primitives import FrozenDict
from catmodel.contracts.run_setup import EvidenceSet, RunSetup
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import strict_json_loads


def setup_from_payload(payload: object) -> RunSetup:
    return decode_scientific(
        payload,
        ExecutionPlan(
            plan_id="fixture-execution",
            required_arms=("reference",),
            requested_views=("gross",),
            simulations=1,
            seed=0,
        ),
        setup_id="fixture-setup",
    )


def setup_from_json(payload: str | bytes) -> RunSetup:
    return setup_from_payload(strict_json_loads(payload))


def scientific_digest(setup: RunSetup) -> str:
    """The scientific digest from its definition, for comparison with the
    streamed one: the canonical JSON digest of the whole inline encoding,
    less the title, the limitations and each alternative's description."""
    inline = encode_scientific(setup)
    del inline["title"], inline["limitations"]
    inline["alternatives"] = [
        {key: value for key, value in row.items() if key != "description"}
        for row in inline["alternatives"]
    ]
    return canonical_json_sha256(inline)


def with_execution(
    setup: RunSetup,
    execution: ExecutionPlan,
    *,
    setup_id: str,
    evidence_files: Mapping[str, bytes] | None = None,
) -> RunSetup:
    return replace(
        setup,
        execution=execution,
        setup_id=setup_id,
        evidence=EvidenceSet(
            records=setup.evidence.records,
            files=setup.evidence.files if evidence_files is None else evidence_files,
        ),
    )


def mutable_containers(value: object, path: str = "value") -> list[str]:
    """The path of every dict, list, set or bytearray, and every unfrozen model or
    dataclass, reachable from ``value`` through models, dataclass inputs,
    tuples, read-only maps and a setup's law overrides: everything there that
    can still change."""
    found = []
    if isinstance(value, FrozenDict | MappingProxyType):
        items = list(value.items())
    elif isinstance(value, dict | list | set | bytearray):
        return [path]
    elif isinstance(value, tuple):
        items = list(enumerate(value))
    elif isinstance(value, LawOverrides):
        items = [(arm, value.of(arm)) for arm in value.arms()]
    elif isinstance(value, BaseModel):
        if not type(value).model_config.get("frozen"):
            found.append(f"{path} ({type(value).__name__} is not frozen)")
        items = [(name, getattr(value, name)) for name in type(value).model_fields]
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Every dataclass has ``__dataclass_params__``; typeshed leaves it out.
        if not type(value).__dataclass_params__.frozen:  # ty: ignore[missing-attribute]
            found.append(f"{path} ({type(value).__name__} is not frozen)")
        items = [
            (row.name, getattr(value, row.name))
            for row in dataclasses.fields(value)
            if row.init
        ]
    else:
        return []
    return found + [
        row for key, item in items for row in mutable_containers(item, f"{path}.{key}")
    ]
