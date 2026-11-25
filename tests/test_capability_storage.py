"""The capture codec preserves native statistics without duplicate derived arrays."""

import hashlib
import io
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from catmodel.results.capability_storage import (
    decode_conditional_draw,
    encode_conditional_draw,
)
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
)


def _draw(path="e" * 64):
    return ConditionalCapabilityDraws(
        stratum_sha256=path,
        response_sha256="d" * 64,
        seed=7,
        views={
            "welfare": ConditionalView(
                np.array([0.0, 1, 4, 20]),
                variance_exists=False,
                native_mean=NativeMean(7, "native_component_mean"),
            ),
            "financial": ConditionalView(
                np.array([0.0, 1, 2, 5]),
                native_mean=NativeMean(2, "native_analytic_mean"),
            ),
        },
    )


def _decode(original, record, data):
    return decode_conditional_draw(
        record,
        data,
        stratum_sha256=original.stratum_sha256,
        response_sha256=original.response_sha256,
        seed=original.seed,
    )


def test_roundtrip_retains_native_aligned_arrays_and_statistics():
    original = _draw()
    record, data = encode_conditional_draw(original)
    loaded = _decode(original, record, data)
    assert {name: view.native_mean for name, view in loaded.views.items()} == {
        name: view.native_mean for name, view in original.views.items()
    }
    assert {name: view.variance_exists for name, view in loaded.views.items()} == {
        name: view.variance_exists for name, view in original.views.items()
    }
    assert loaded.seed == 7
    np.testing.assert_array_equal(
        loaded.normalized_weights(), original.normalized_weights()
    )
    for view in original.annual_losses:
        np.testing.assert_array_equal(
            loaded.annual_losses[view], original.annual_losses[view]
        )
        assert not loaded.annual_losses[view].flags.writeable


def test_grouped_draws_keep_the_existing_stored_view_schema():
    record, _ = encode_conditional_draw(_draw())
    financial = record.model_dump(mode="json")["views"]["financial"]
    assert financial == {
        "array": record.views["financial"].array,
        "mean_exists": True,
        "variance_exists": True,
        "native_mean": {
            "value": 2.0,
            "estimator": "native_analytic_mean",
            "precision": None,
        },
    }
    assert set(record.model_dump()) == {
        "views",
        "joint_law_semantics",
        "evaluator_provenance_by_risk_unit",
        "numeric_sha256",
    }


def test_each_native_mean_keeps_its_own_missing_precision_through_storage():
    original = _draw()
    original = replace(
        original,
        views={
            **original.views,
            "financial": replace(
                original.views["financial"],
                native_mean=NativeMean(
                    2,
                    "native_analytic_mean",
                    {
                        "numerical_standard_error": 0,
                        "precision_basis": "deterministic_integration",
                    },
                ),
            ),
        },
    )
    loaded = _decode(original, *encode_conditional_draw(original))
    mixture = CapabilityMixture(
        {loaded.stratum_sha256: loaded}, {loaded.stratum_sha256: 1}
    )
    financial = cast(dict[str, object], mixture.metrics("financial")["eal_precision"])
    welfare = cast(dict[str, object], mixture.metrics("welfare")["eal_precision"])
    assert financial["standard_error_upper_bound"] == 0
    assert welfare["standard_error_upper_bound"] is None


def test_identical_views_share_one_vector_and_no_normalized_copy():
    original = _draw()
    shared = original.views["welfare"].array
    original = replace(
        original,
        views={
            name: replace(view, array=shared) for name, view in original.views.items()
        },
    )
    record, data = encode_conditional_draw(original)
    assert record.views["welfare"].array == record.views["financial"].array
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        assert archive.files == ["array_0"]
        np.testing.assert_array_equal(archive["array_0"], shared)
    loaded = _decode(original, record, data)
    assert loaded.annual_losses["welfare"] is loaded.annual_losses["financial"]


def test_stored_estimators_reweight_without_repooling_or_simulation():
    first, second = (
        _draw(),
        replace(
            _draw("f" * 64),
            views={
                name: replace(view, native_mean=replace(view.native_mean, value=value))
                for (name, view), value in zip(
                    _draw().views.items(), (17, 12), strict=True
                )
            },
        ),
    )
    draws = {
        original.stratum_sha256: _decode(original, *encode_conditional_draw(original))
        for original in (first, second)
    }
    mixture = CapabilityMixture(
        draws,
        {first.stratum_sha256: 0.9, second.stratum_sha256: 0.1},
    )
    assert mixture.metrics("financial")["expected_annual_loss"] == 3
    reweighted = replace(
        mixture, stratum_weights={first.stratum_sha256: 0.2, second.stratum_sha256: 0.8}
    )
    assert reweighted.metrics("financial")["expected_annual_loss"] == pytest.approx(10)
    assert reweighted.draws is mixture.draws


def test_year_masses_need_no_stored_weight_array():
    original = _draw()
    record, data = encode_conditional_draw(original)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        assert len(archive.files) == 2
    np.testing.assert_array_equal(
        _decode(original, record, data).normalized_weights(), np.full(4, 0.25)
    )


def test_numeric_checksum_and_declared_array_roster_are_enforced():
    original = _draw()
    record, data = encode_conditional_draw(original)
    with pytest.raises(ValueError, match="checksum"):
        _decode(original, record, data + b"corruption")
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = dict(source)
    arrays["untracked"] = np.zeros(4)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    data = output.getvalue()
    record = record.model_copy(
        update={"numeric_sha256": hashlib.sha256(data).hexdigest()}
    )
    with pytest.raises(ValueError, match="roster"):
        _decode(original, record, data)


def test_codec_never_loads_pickle_even_with_restamped_numeric_hash():
    original = _draw()
    record, data = encode_conditional_draw(original)
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = dict(source)
    arrays[record.views["financial"].array] = np.array([object()], dtype=object)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    data = output.getvalue()
    record = record.model_copy(
        update={"numeric_sha256": hashlib.sha256(data).hexdigest()}
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        _decode(original, record, data)


def test_draw_vectors_cannot_be_changed_through_inputs_or_write_flags():
    original = _draw()
    losses = np.array([1.0, 2.0, 3.0])
    draw = replace(
        original,
        views={
            name: replace(view, array=losses) for name, view in original.views.items()
        },
    )
    losses[0] = 999
    np.testing.assert_array_equal(draw.annual_losses["financial"], [1, 2, 3])
    for vector in draw.annual_losses.values():
        with pytest.raises(ValueError):
            vector.flags.writeable = True
    with pytest.raises(TypeError):
        draw.annual_losses["new"] = losses
    with pytest.raises(TypeError):
        draw.views["new"] = ConditionalView(losses)
