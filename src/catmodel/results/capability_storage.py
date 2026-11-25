"""Typed numeric payloads referenced by the single immutable study capture."""

from __future__ import annotations

import hashlib
import io
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmodel.contracts.primitives import Sha256Digest
from catmodel.hashing import canonical_json_sha256
from catmodel.simulation.capability_mixture import (
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def finite_metadata(self):
        canonical_json_sha256(self.model_dump(mode="json"))
        return self


class StoredMean(_Record):
    value: float = Field(ge=0, allow_inf_nan=False)
    estimator: str = Field(min_length=1)
    precision: dict[str, object] | None = None


class StoredView(_Record):
    array: str
    mean_exists: bool
    variance_exists: bool
    native_mean: StoredMean | None = None


class StoredDraw(_Record):
    views: dict[str, StoredView] = Field(min_length=1)
    joint_law_semantics: Literal["physical_joint", "marginal_only", "mixed_or_unknown"]
    evaluator_provenance_by_risk_unit: dict[str, dict[str, object]] | None
    numeric_sha256: Sha256Digest

    @property
    def array_path(self) -> str:
        return f"arrays/{self.numeric_sha256}.npz"


def _numeric_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind not in "fiu":
        raise ValueError(
            "capture accepts real numeric arrays only; no pickle/object data"
        )
    return np.array(values, dtype=float, copy=True)


def encode_conditional_draw(
    draw: ConditionalCapabilityDraws,
) -> tuple[StoredDraw, bytes]:
    """Store each distinct vector once; year masses are rederived on load."""
    draw.__post_init__()
    arrays, references = {}, {}

    def retain(values: np.ndarray) -> str:
        array = _numeric_array(values)
        key = hashlib.sha256(array.tobytes()).hexdigest()
        if key not in references:
            name = f"array_{len(arrays)}"
            references[key] = name
            arrays[name] = array
        return references[key]

    def stored(mean: NativeMean | None) -> StoredMean | None:
        if mean is None:
            return None
        return StoredMean(
            value=mean.value,
            estimator=mean.estimator,
            precision=None if mean.precision is None else dict(mean.precision),
        )

    views = {}
    for name, view in sorted(draw.views.items()):
        views[name] = StoredView(
            array=retain(view.array),
            mean_exists=view.mean_exists,
            variance_exists=view.variance_exists,
            native_mean=stored(view.native_mean),
        )
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    data = output.getvalue()
    return StoredDraw(
        views=views,
        joint_law_semantics=draw.joint_law_semantics,
        evaluator_provenance_by_risk_unit=None
        if draw.evaluator_provenance_by_risk_unit is None
        else {
            key: dict(value)
            for key, value in draw.evaluator_provenance_by_risk_unit.items()
        },
        numeric_sha256=hashlib.sha256(data).hexdigest(),
    ), data


def _restored(mean: StoredMean | None) -> NativeMean | None:
    if mean is None:
        return None
    return NativeMean(
        value=mean.value, estimator=mean.estimator, precision=mean.precision
    )


def decode_conditional_draw(
    record: StoredDraw,
    data: bytes,
    *,
    stratum_sha256: str,
    response_sha256: str,
    seed: int,
) -> ConditionalCapabilityDraws:
    """Restore aligned immutable vectors under their enclosing capture's identity."""
    record = StoredDraw.model_validate(record.model_dump(mode="json"))
    if hashlib.sha256(data).hexdigest() != record.numeric_sha256:
        raise ValueError("conditional numeric array checksum mismatch")
    expected = {view.array for view in record.views.values()}
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != expected or len(archive.files) != len(expected):
            raise ValueError("numeric archive differs from its declared array roster")
        arrays = {name: _numeric_array(archive[name]) for name in archive.files}
    for array in arrays.values():
        array.flags.writeable = False
    return ConditionalCapabilityDraws(
        stratum_sha256=stratum_sha256,
        response_sha256=response_sha256,
        seed=seed,
        views={
            name: ConditionalView(
                array=arrays[view.array],
                mean_exists=view.mean_exists,
                variance_exists=view.variance_exists,
                native_mean=_restored(view.native_mean),
            )
            for name, view in record.views.items()
        },
        joint_law_semantics=record.joint_law_semantics,
        evaluator_provenance_by_risk_unit=record.evaluator_provenance_by_risk_unit,
    )
