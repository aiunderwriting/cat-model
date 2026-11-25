import pytest
from pydantic import BaseModel, ValidationError

from catmodel.contracts.governance import (
    AssumptionRecord,
    AssumptionRegisterBase,
    ExternalSourceSnapshot,
    FileChecksum,
    SourceClassification,
)
from catmodel.contracts.primitives import (
    OptionalString,
    RequiredIdentifier,
    Sha256Digest,
)


class _PrimitivePayload(BaseModel):
    required_id: RequiredIdentifier
    optional_id: OptionalString = None
    digest: Sha256Digest


def _assumption(assumption_id: str = "severity-tail") -> AssumptionRecord:
    return AssumptionRecord(
        assumption_id=assumption_id,
        title="Tail selection",
        source_classification=SourceClassification.EMPIRICAL_FIT,
        status="OK",
        uncertainty_type="epistemic",
        materiality="high",
        rationale="Selected from fitted candidate distributions.",
        empirical_grounding="Calibrated loss observations.",
        sensitivity_range="Adjacent accepted candidates.",
        sensitivity_tested=True,
        sensitivity_evidence_refs=["sensitivity/severity-tail.json"],
    )


def test_string_and_digest_primitives_normalize_without_lowercasing() -> None:
    payload = _PrimitivePayload(
        required_id="  source-a ", optional_id="   ", digest=f" {'a' * 64} "
    )

    assert payload.required_id == "source-a"
    assert payload.optional_id is None
    assert payload.digest == "a" * 64

    with pytest.raises(ValidationError, match="lowercase 64-character"):
        _PrimitivePayload(required_id="source-a", digest="A" * 64)


def test_file_checksum_and_external_snapshot_have_deliberate_hash_rules() -> None:
    checksum = FileChecksum(sha256="b" * 64, role="  calibration input ")
    snapshot = ExternalSourceSnapshot(
        source_id=" kev ",
        uri=" https://example.test/kev ",
        content_hash=" etag:ABC-123 ",
        filters="   ",
        record_count=0,
    )

    assert checksum.role == "calibration input"
    assert snapshot.source_id == "kev"
    assert snapshot.content_hash == "etag:ABC-123"
    assert snapshot.filters is None

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ExternalSourceSnapshot(
            source_id="kev",
            uri="https://example.test",
            content_hash="etag",
            record_count=-1,
        )


def test_external_source_snapshot_validates_availability_and_captured_paths() -> None:
    with pytest.raises(ValidationError, match="ISO calendar date"):
        ExternalSourceSnapshot(
            source_id="kev",
            uri="https://example.test",
            content_hash="etag",
            availability_date="2026-2-03",
        )

    with pytest.raises(ValidationError, match="relative paths"):
        ExternalSourceSnapshot(
            source_id="kev",
            uri="https://example.test",
            content_hash="etag",
            captured_paths=["../outside.csv"],
        )


def test_an_empty_register_states_that_there_are_no_assumptions() -> None:
    assert AssumptionRegisterBase.model_validate({"assumptions": []}).assumptions == []
    # Saying nothing is not saying there are none.
    with pytest.raises(ValidationError, match="assumptions"):
        AssumptionRegisterBase.model_validate({})


def test_assumption_register_detects_duplicates_after_id_normalization() -> None:
    with pytest.raises(ValidationError, match="duplicate assumption_id"):
        AssumptionRegisterBase(
            assumptions=[_assumption("severity-tail"), _assumption(" severity-tail ")]
        )


def test_tested_sensitivity_claim_requires_cited_evidence() -> None:
    data = _assumption().model_dump()
    data["sensitivity_evidence_refs"] = []
    with pytest.raises(ValidationError, match="requires sensitivity_evidence_refs"):
        AssumptionRecord.model_validate(data)
    data["sensitivity_tested"] = False
    AssumptionRecord.model_validate(data)


def test_assumption_record_rejects_blank_list_members() -> None:
    data = _assumption().model_dump()
    data["applies_to"] = ["scenario-a", " "]
    with pytest.raises(ValidationError, match="must not be blank"):
        AssumptionRecord.model_validate(data)


def test_external_source_paths_normalize_and_accept_upstream_evidence() -> None:
    snapshot = ExternalSourceSnapshot(
        source_id="source",
        uri="https://example.test",
        content_hash="etag",
        captured_paths=["./evidence//observations.csv", "assumptions\\study.yaml"],
        transformations=[],
    )
    assert snapshot.captured_paths == [
        "evidence/observations.csv",
        "assumptions/study.yaml",
    ]
    assert snapshot.transformations == []


@pytest.mark.parametrize(
    "paths",
    [
        ["/outside.csv"],
        ["evidence/../../outside.csv"],
        ["evidence\\..\\outside.csv"],
        ["."],
        ["evidence/"],
        ["evidence/file.csv", "./evidence/file.csv"],
    ],
)
def test_external_source_paths_reject_escapes_and_normalized_duplicates(paths):
    with pytest.raises(ValidationError, match="captured_paths"):
        ExternalSourceSnapshot(
            source_id="source",
            uri="https://example.test",
            content_hash="etag",
            captured_paths=paths,
        )


def test_an_external_source_refuses_a_field_its_contract_does_not_define():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExternalSourceSnapshot.model_validate(
            {
                "source_id": "source",
                "uri": "https://example.test",
                "content_hash": "etag",
                "unknown_field": [],
            }
        )
