"""A saved setup.json stores each shared record once and decodes to the same setup."""

import hashlib
import json
import math
import typing
from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from pydantic import BaseModel

from catmodel.contracts import _setup_codec
from catmodel.contracts._setup_codec import (
    SCHEMA_VERSION,
    SEVERITY_PATHS,
    SHARED_RECORDS,
    decode_saved_setup,
    decode_scientific,
    encode_scientific,
    saved_setup,
)
from catmodel.contracts._shared_records import (
    _KERNEL,
    _KERNEL_TEXTS,
    _LAW_TEXTS,
    _TRAJECTORY,
)
from catmodel.contracts.incident import ExecutionPlan, IncidentLaw, MarkKernel
from catmodel.contracts.native_severity import NativeSeverityLaw
from catmodel.contracts.primitives import Frozen, RequiredString
from catmodel.hashing import canonical_json, canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.resources import config_path
from catmodel.setup import store as packages_setup
from catmodel.setup.store import build_setup, load_setup
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_moments import analytic_linear_eals
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import (
    mutable_containers,
    setup_from_json,
    setup_from_payload,
    with_execution,
)
from test_support.study import _financial_study, _study, stipulated_study
from tests.test_public_frontier_comparisons import _compile

PLAN = ExecutionPlan(
    plan_id="storage",
    required_arms=("reference",),
    requested_views=("gross",),
    simulations=8,
    seed=19,
)


@cache
def _public_frontier():
    """Paths, respondents, parameter ranks, public trajectories and every
    default comparison family's arm overrides."""
    return _compile()


PRODUCERS = {
    "stipulated": lambda: with_execution(stipulated_study(), PLAN, setup_id="s"),
    "multi-state": lambda: with_execution(_study(), PLAN, setup_id="s"),
    "financial": lambda: with_execution(_financial_study(), PLAN, setup_id="s"),
    "conditional": lambda: with_execution(
        setup_from_json(
            config_path("setups", "stipulated-conditional", "setup.json").read_bytes()
        ),
        ExecutionPlan.model_validate_json(
            config_path("setups", "stipulated-conditional", "plan.json").read_bytes()
        ),
        setup_id="s",
    ),
    "public-frontier": _public_frontier,
    "golden-poisson": lambda: _golden_poisson(),
}


def _golden_poisson():
    """The golden Poisson example."""
    root = config_path("setups", "golden-poisson")
    payload = strict_json_loads((root / "setup.json").read_bytes())
    return with_execution(
        setup_from_payload(payload),
        ExecutionPlan.model_validate_json((root / "plan.json").read_bytes()),
        setup_id="s",
    )


def _decode(stored, setup, files=None):
    """Decode a saved ``setup.json`` beside ``setup``'s alternative files."""
    return decode_saved_setup(
        stored,
        setup.execution,
        setup_id=setup.setup_id,
        evidence_files=setup.evidence.files,
        files=saved_setup(setup)[1] if files is None else files,
    )


def _classes(annotation, sequence=False):
    """Each class an annotation can hold, and whether it sits in a tuple."""
    origin = typing.get_origin(annotation)
    if origin is None:
        return [(annotation, sequence)] if isinstance(annotation, type) else []
    if origin is typing.Annotated:
        return _classes(typing.get_args(annotation)[0], sequence)
    return [
        row
        for arg in typing.get_args(annotation)
        for row in _classes(arg, sequence or origin is tuple)
    ]


def _model_paths(model, target, prefix=()):
    """Field paths from ``model`` to every field of type ``target``."""
    paths = set()
    for name, info in model.model_fields.items():
        for cls, sequence in _classes(info.annotation):
            if not issubclass(cls, BaseModel):
                continue
            if cls is target:
                paths.add((*prefix, name))
            else:
                paths |= _model_paths(
                    cls, target, (*prefix, name, *(("*",) if sequence else ()))
                )
    return paths


def test_severity_paths_are_every_native_severity_law_of_a_mark_kernel():
    assert set(SEVERITY_PATHS) == _model_paths(MarkKernel, NativeSeverityLaw)
    assert len(SEVERITY_PATHS) == len(set(SEVERITY_PATHS))


@pytest.mark.parametrize("producer", sorted(PRODUCERS))
def test_a_saved_setup_decodes_to_the_same_setup_and_laws(producer):
    setup = PRODUCERS[producer]()
    saved = strict_json_loads(saved_setup(setup)[0])
    assert saved["schema_version"] == SCHEMA_VERSION
    for name in SHARED_RECORDS:
        values = list(saved[name].values())
        assert len({canonical_json(value) for value in values}) == len(values)
    decoded = _decode(saved, setup)
    # Digests first: a large unequal setup is slow for pytest to diff.
    assert decoded.scientific_sha256 == setup.scientific_sha256
    assert decoded == setup
    assert mutable_containers(decoded, "decoded") == []
    assert mutable_containers(setup, "composed") == []
    for arm in setup.execution.required_arms:
        for state in setup.joint_states:
            before = compile_incident_state(setup, state.state_id, arm)
            after = compile_incident_state(decoded, state.state_id, arm)
            assert (after.conditional_key, after.law_sha256) == (
                before.conditional_key,
                before.law_sha256,
            )
            views = setup.execution.requested_views
            assert (
                analytic_linear_eals(after, views).values
                == analytic_linear_eals(before, views).values
            )


def test_public_frontier_records_are_shared_and_the_saved_setup_simulates_alike(
    tmp_path,
):
    setup = _public_frontier()
    text, arms = saved_setup(setup)
    saved = json.loads(text)
    inline = encode_scientific(setup)
    reference = sum(len(state["laws"]) for state in inline["joint_states"])
    overridden = {
        row["alternative_id"]: len(row["overrides"])
        for row in inline["alternatives"]
        if row["overrides"]
    }
    assert reference > 20 and sum(overridden.values()) > 30
    assert all(saved[name] for name in SHARED_RECORDS)
    # Each distinct kernel, severity law and trajectory is stored once in its
    # file, and each arm that overrides laws has its own file, named by its
    # digest.
    assert len(saved["mark_kernels"]) < reference / 4
    named = {
        row["alternative_id"]: row["overrides"]
        for row in saved["alternatives"]
        if row["overrides"] is not None
    }
    assert set(named) == set(overridden) and set(named.values()) == set(arms)
    for alternative_id, name in named.items():
        assert name == f"alternatives/{hashlib.sha256(arms[name]).hexdigest()}.json"
        arm = json.loads(arms[name])
        assert arm["alternative_id"] == alternative_id
        assert len(arm["overrides"]) == overridden[alternative_id]
        assert len(arm["mark_kernels"]) < overridden[alternative_id] / 4
    assert len(text) + sum(len(arms[name]) for name in arms) < 0.4 * len(
        canonical_json(inline)
    )
    package = build_setup(
        destination=tmp_path / "setup", setup=setup, code_commit="fixture"
    )
    written = (tmp_path / "setup" / "setup.json").read_bytes()
    assert written == (text + "\n").encode() and b"\n " not in written
    loaded = load_setup(tmp_path / "setup")
    assert loaded.scientific_sha256 == setup.scientific_sha256
    assert package == loaded == setup
    state = setup.joint_states[-1].state_id
    for arm in setup.execution.required_arms[:3]:
        draws = [
            simulate_incident_state(
                setup.execution.for_arm(arm),
                compile_incident_state(source, state, arm),
                response_sha256="a" * 64,
            )[0]
            for source in (setup, loaded)
        ]
        for view, row in draws[0].views.items():
            np.testing.assert_array_equal(row.array, draws[1].views[view].array)


def test_build_and_load_decode_the_saved_file_once(tmp_path, monkeypatch):
    setup = PRODUCERS["conditional"]()
    calls = {"_decode": 0, "_encode": 0, "scientific_sha256": 0}
    for name in calls:

        def counted(*args, _original=getattr(_setup_codec, name), _name=name, **kw):
            calls[_name] += 1
            return _original(*args, **kw)

        monkeypatch.setattr(_setup_codec, name, counted)
    built = build_setup(
        destination=tmp_path / "setup", setup=setup, code_commit="fixture"
    )
    # One encoding saves the file; the composed and decoded digests are compared.
    assert calls == {"_decode": 1, "_encode": 1, "scientific_sha256": 2}
    calls.update(_decode=0, _encode=0, scientific_sha256=0)
    loaded = load_setup(tmp_path / "setup")
    for _ in range(3):
        assert loaded.files and loaded.manifest and loaded.execution_plan_sha256
        assert loaded.scientific_sha256 == built.scientific_sha256
    assert calls == {"_decode": 1, "_encode": 0, "scientific_sha256": 1}
    assert loaded == built == setup


def test_building_the_same_inputs_again_writes_the_same_bytes(tmp_path):
    # A setup records no build time, so its manifest, and the identity that
    # its digest is, change only with the setup ID, code and saved files.
    setup = PRODUCERS["stipulated"]()
    first, second = (
        build_setup(destination=tmp_path / name, setup=setup, code_commit="fixture")
        for name in ("first", "second")
    )
    assert set(first.files) == set(second.files)
    for name in first.files:
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()
    assert first.manifest_sha256 == second.manifest_sha256


@pytest.mark.parametrize("file", ["setup.json", "execution-plan.json"])
def test_build_refuses_a_saved_file_that_does_not_reproduce_its_inputs(
    tmp_path, monkeypatch, file
):
    setup = PRODUCERS["stipulated"]()
    if file == "setup.json":
        other = replace(setup, limitations=(*setup.limitations, "Another limitation."))
        monkeypatch.setattr(packages_setup, "saved_setup", lambda _: saved_setup(other))
    else:
        # A plan that differs only in its seed leaves the scientific digest alone.
        text = setup.execution.model_copy(update={"seed": 20}).model_dump_json(indent=2)
        monkeypatch.setattr(ExecutionPlan, "model_dump_json", lambda *_, **__: text)
    with pytest.raises(ValueError, match="differs from the composed setup inputs"):
        build_setup(destination=tmp_path / "setup", setup=setup, code_commit="fixture")
    assert not (tmp_path / "setup").exists()


def _saved():
    return strict_json_loads(saved_setup(_public_frontier())[0])


def _copy(value):
    return json.loads(json.dumps(value))


def test_near_duplicate_severity_laws_stay_separate_entries():
    """Laws one unit in the last place apart, or apart only in the sign of a
    zero, are different laws: each keeps its own entry and decodes exactly."""
    setup = _public_frontier()
    inline = encode_scientific(setup)
    kernel = inline["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    components = kernel["member_outcomes"][0]["components"]
    base = components[0]["severity"]
    assert components[1]["severity"] == components[2]["severity"] == base
    assert components[3]["severity"] == base
    ulp = _copy(base)
    ulp["components"][0]["p50"] = math.nextafter(ulp["components"][0]["p50"], math.inf)
    # An exponential tail, written once with a positive and once with a
    # negative zero shape.
    zero = _copy(base)
    zero["components"][0]["gpd_shape"] = 0.0
    negative = _copy(zero)
    negative["components"][0]["gpd_shape"] = -0.0
    components[1]["severity"] = ulp
    components[2]["severity"], components[3]["severity"] = zero, negative
    changed = decode_scientific(inline, setup.execution, setup_id=setup.setup_id)
    saved = strict_json_loads(saved_setup(changed)[0])
    assert len(saved["severity_laws"]) == len(_saved()["severity_laws"]) + 3
    decoded = encode_scientific(_decode(saved, changed))
    rows = decoded["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"]
    assert rows[1]["severity"]["components"][0]["p50"].hex() == (
        ulp["components"][0]["p50"].hex()
    )
    assert math.copysign(1, rows[3]["severity"]["components"][0]["gpd_shape"]) == -1
    assert math.copysign(1, rows[2]["severity"]["components"][0]["gpd_shape"]) == 1
    assert canonical_json_sha256(decoded) == canonical_json_sha256(
        encode_scientific(changed)
    )


def _model_field(model, path):
    """The annotation at ``path`` of field names from ``model``."""
    annotation = None
    for name in path:
        assert isinstance(model, type) and issubclass(model, BaseModel), path
        assert name in model.model_fields, path
        annotation = model.model_fields[name].annotation
        models = [cls for cls, _ in _classes(annotation) if issubclass(cls, BaseModel)]
        model = models[0] if models else None
    return annotation


def test_every_shared_position_names_a_field_of_its_record():
    """A renamed field fails here instead of silently dropping out of a table."""
    assert {(*path,) for path in _KERNEL} == _model_paths(IncidentLaw, MarkKernel)
    for path in _LAW_TEXTS:
        assert _model_field(IncidentLaw, path) is RequiredString
    for path in _KERNEL_TEXTS:
        assert _model_field(MarkKernel, path) is RequiredString
    ((context, key),) = _TRAJECTORY
    assert _model_field(IncidentLaw, (context,)) == Frozen[dict[str, object]]
    assert key == "public_trajectory"
    law = _public_frontier().joint_states[0].laws[0]
    assert key in law.conditioning_context


@pytest.mark.parametrize("table", SHARED_RECORDS)
def test_an_entry_no_law_uses_is_refused_in_every_table(table):
    saved = _saved()
    extra = "unused prose" if table == "texts" else {"unused": table}
    saved[table] = {**saved[table], canonical_json_sha256(extra): extra}
    with pytest.raises(ValueError, match=f"no law uses.*{table}"):
        _decode(saved, _public_frontier())


@pytest.mark.parametrize("table", SHARED_RECORDS)
def test_an_entry_unlike_its_digest_is_refused_in_every_table(table):
    saved = _saved()
    key = next(iter(saved[table]))
    value = saved[table][key]
    saved[table] = {
        **saved[table],
        key: value + " " if isinstance(value, str) else {**value, "changed": 1},
    }
    with pytest.raises(ValueError, match=f"{table} keys differ from their digests"):
        _decode(saved, _public_frontier())


def test_strict_decoding_refuses_every_inconsistent_file():
    setup = _public_frontier()
    saved = _saved()
    severity = next(iter(saved["severity_laws"]))
    kernel_key = saved["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    without = _copy(saved)
    without["severity_laws"].pop(severity)
    cases = {
        f"schema version {SCHEMA_VERSION + 1} is not": {
            **saved,
            "schema_version": SCHEMA_VERSION + 1,
        },
        "schema version True is not": {**saved, "schema_version": True},
        f"schema version {float(SCHEMA_VERSION)} is not": {
            **saved,
            "schema_version": float(SCHEMA_VERSION),
        },
        "schema version is not stated": {
            key: value for key, value in saved.items() if key != "schema_version"
        },
        "lacks its texts table": {
            key: value for key, value in saved.items() if key != "texts"
        },
        "refers to no severity_laws entry": without,
    }
    # A shared position holding the value itself instead of its key.
    inline_kernel = _copy(saved)
    law = inline_kernel["joint_states"][0]["laws"][0]
    law["segments"][0]["kernel"] = saved["mark_kernels"][kernel_key]
    cases["refers to no mark_kernels entry"] = inline_kernel
    inline_text = _copy(saved)
    law = inline_text["joint_states"][0]["laws"][0]
    law["qualification_assumption"] = saved["texts"][law["qualification_assumption"]]
    cases["refers to no texts entry"] = inline_text
    for message, payload in cases.items():
        with pytest.raises(ValueError, match=message):
            _decode(payload, setup)
    # The inline layout of authored inputs is not a saved setup.json.
    with pytest.raises(ValueError, match="schema version"):
        _decode(encode_scientific(setup), setup)


def _malformed(change):
    saved = _saved()
    change(saved)
    return saved


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda saved: saved["joint_states"][0].pop("laws"), "joint state laws"),
        (lambda saved: saved.update(joint_states={}), "joint states"),
        (
            lambda saved: saved["alternatives"][0].update(overrides={}),
            "alternative overrides",
        ),
        (lambda saved: saved["joint_states"][0]["laws"].append(3), "joint state laws"),
        (
            lambda saved: saved["joint_states"][0]["laws"][0].update(segments={}),
            "where a list is required",
        ),
    ],
    ids=[
        "state-without-laws",
        "states-object",
        "overrides-object",
        "law-number",
        "segments-object",
    ],
)
def test_a_malformed_saved_file_is_refused_with_a_value_error(change, message):
    with pytest.raises(ValueError, match=message):
        _decode(_malformed(change), _public_frontier())


def test_a_saved_file_is_not_an_authored_input():
    setup = _public_frontier()
    with pytest.raises(ValueError, match=r"this is a saved setup\.json"):
        decode_scientific(_saved(), setup.execution, setup_id=setup.setup_id)
