"""Reduction and replay share their intermediates and spread work over
processes without changing a result.

Each view's statistics share one sample, one sort and one batching; replay
checks each distinct captured conditional against its law once, in worker
processes, which also reduce the arms' views.
"""

import errno
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.hashing import canonical_json_sha256
from catmodel.probability.percentiles import SortedSample
from catmodel.results import execution, reduction
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.reduction import mixture_view_metrics
from catmodel.results.results_export import export_results
from catmodel.results.results_file import diagnostics_path
from catmodel.simulation import tail_gate
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
    tail_parts,
    weighted_var_tvar,
)
from catmodel.simulation.tail_gate import TailGateControls, view_gate
from test_support.results import available_metric, reduced_arm
from tests.test_parallel_simulation import (
    _BOTH_GATE,
    _GATE,
    _without_execution_context,
)
from tests.test_percentiles import exact_left_inverse, exact_steps
from tests.test_tail_gate import _arms_package

LEVELS = (0.5, 0.9, 0.99, 0.995)


def _percentile(values, weights, q):
    """The weighted left-inverse percentile in rational arithmetic, sorted
    afresh: the reference the shared sort must reproduce exactly."""
    return exact_left_inverse(exact_steps(values, weights), Fraction(str(q)) / 100)


@pytest.mark.parametrize("uniform", [True, False])
def test_one_sort_gives_every_weighted_percentile(uniform):
    rng = np.random.default_rng(5)
    # Ties and an atom at zero, as annual losses have.
    values = np.round(rng.pareto(1.5, 3001), 1) * (rng.random(3001) < 0.4)
    weights = np.full(3001, 1 / 3001) if uniform else rng.random(3001)
    ordered = SortedSample(values, weights)
    for q in (0.0, 12.5, 50.0, 90.0, 99.0, 99.9, 100.0):
        assert ordered.percentile(q) == _percentile(values, weights, q)
    for level in (0.5, 0.99):
        assert tail_parts(values, weights, level, ordered=ordered) == tail_parts(
            values, weights, level
        )
        assert weighted_var_tvar(
            values, weights, level, ordered=ordered
        ) == weighted_var_tvar(values, weights, level)


def _native_mean(losses):
    return NativeMean(
        float(np.mean(losses)) * 1.05,
        "native_analytic_mean",
        {
            "precision_basis": "deterministic_integration",
            "integration_error_bound": 0.0,
            "numerical_standard_error": 0.0,
            "standard_error_upper_bound": 0.0,
        },
    )


def _mixture(states=3, years=640):
    """Three states with two views, so a shared intermediate that ignored
    the view would mix them."""
    rng = np.random.default_rng(11)
    draws = {}
    for index in range(states):
        gross = rng.pareto(1.8, years) * (rng.random(years) < 0.3) * 10**index
        welfare = gross * rng.random(years)
        draws[f"s{index}"] = ConditionalCapabilityDraws(
            stratum_sha256=f"s{index}",
            response_sha256="response",
            seed=index,
            views={
                view: ConditionalView(losses, native_mean=_native_mean(losses))
                for view, losses in (("gross", gross), ("welfare", welfare))
            },
        )
    masses = [0.5, 0.3, 0.2][:states]
    return CapabilityMixture(draws, dict(zip(draws, masses, strict=True)))


def _unshared(monkeypatch):
    """Compute every intermediate afresh: no mixture memo and no shared sort."""
    from catmodel.simulation import capability_mixture, tail_gate

    monkeypatch.setattr(
        CapabilityMixture, "derived", lambda self, key, compute: compute()
    )
    for module in (capability_mixture, tail_gate):
        sort_afresh = module.weighted_var_tvar

        def unordered(losses, masses, level, *, ordered=None, _f=sort_afresh, **kw):
            return _f(losses, masses, level, **kw)

        monkeypatch.setattr(module, "weighted_var_tvar", unordered)


def test_shared_intermediates_leave_every_view_statistic_unchanged(monkeypatch):
    controls = TailGateControls(batches=8)
    shared = _mixture()
    # Another batch count first, so a shared intermediate that ignored it
    # would leak into the view's statistics.
    four = TailGateControls(batches=4)
    view_gate(shared, "gross", {}, four, mean_exists=True)
    first = {
        view: mixture_view_metrics(shared, view, LEVELS, controls)
        for view in ("welfare", "gross")
    }
    again = mixture_view_metrics(shared, "gross", LEVELS, controls)
    assert (
        first["gross"]["tail_diagnostics"]["convergence"]["unavailable_reason"] is None
    )
    _unshared(monkeypatch)
    fresh = {
        view: mixture_view_metrics(_mixture(), view, LEVELS, controls)
        for view in ("welfare", "gross")
    }
    assert repr(first) == repr(fresh)
    assert repr(again) == repr(fresh["gross"])
    assert repr(view_gate(_mixture(), "gross", {}, four, mean_exists=True)) == repr(
        view_gate(shared, "gross", {}, four, mean_exists=True)
    )


def test_a_mixture_shares_read_only_intermediates():
    mixture = _mixture()
    losses, masses = mixture.samples("gross")
    assert mixture.samples("gross")[0] is losses
    assert not losses.flags.writeable and not masses.flags.writeable
    (draw, _), *_ = mixture.positive_strata()
    with pytest.raises(ValueError):
        draw.normalized_weights()[0] = 0.0
    batched = tail_gate._batched(mixture, "gross", 8)
    assert not isinstance(batched, str)
    assert tail_gate._batched(mixture, "gross", 8) is batched
    losses, masses, parts = batched
    for array in (losses, masses, *(value for part in parts for value in part)):
        assert not array.flags.writeable


def test_replay_checks_each_distinct_captured_conditional_once(tmp_path, monkeypatch):
    # The follower holds the reference arm's record of the shared state, so
    # only the reference arm's state and the comparison arm's two are checked.
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    checked = []
    check = execution._check_law

    def counted(task):
        checked.append(task.compiled.state_id)
        return check(task)

    monkeypatch.setattr(execution, "_check_law", counted)
    replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert sorted(checked) == ["high", "high", "low"]


def test_replay_workers_refuse_a_tampered_law_with_the_serial_message(tmp_path):
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    record = json.loads((capture / "run-capture.json").read_text())
    for row in record["arms"]["both"].values():
        row["execution"]["work_limits"]["vectorized_initiators"] += 1
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    (capture / "run-capture.json").write_text(json.dumps(record))
    messages = set()
    for workers in (1, 2):
        with pytest.raises(ValueError) as refusal:
            replay_capture(
                setup_path=package.root,
                capture_directory=capture,
                output=tmp_path / f"replay-{workers}.json",
                code_commit="fixture",
                workers=workers,
            )
        messages.add(str(refusal.value))
    assert messages == {
        "capture work counts or limits differ from the plan's work allowances per "
        "year times the captured years"
    }


def test_replay_refuses_a_capture_whose_draws_were_addressed_differently(tmp_path):
    """A capture records how its draws were addressed, the parameter-rank
    stream included, and replay re-derives that record: one drawn under
    another addressing, here one without the rank stream, does not replay."""
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    record = json.loads((capture / "run-capture.json").read_text())
    designs = [
        row["execution"]["stream_design"]
        for rows in record["arms"].values()
        for row in rows.values()
    ]
    assert designs and all(
        design["rank_stream_id"] == design["state_id"] for design in designs
    )
    for design in designs:
        del design["rank_stream_id"], design["parameter_rank_scope"]
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    (capture / "run-capture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="capture stream design differs"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


def test_replay_takes_and_records_the_global_workers(tmp_path):
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    reports = {}
    for workers in ("1", "2"):
        output = tmp_path / f"replay-{workers}"
        result = CliRunner().invoke(
            cli,
            [
                "--workers",
                workers,
                "results",
                "replay",
                "--setup",
                str(package.root),
                "--capture",
                str(tmp_path / "run" / "capture"),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        reports[workers] = json.loads((output / "results.json").read_bytes())
        assert reports[workers]["execution_context"] == {"workers": int(workers)}
    assert _without_execution_context(reports["2"]) == _without_execution_context(
        reports["1"]
    )


def test_reduction_maps_each_available_view_in_order(monkeypatch):
    mapped = []

    def recording(function, tasks):
        tasks = list(tasks)
        mapped.append([task.view for task in tasks])
        return [function(task) for task in tasks]

    mixture = _mixture()
    report = reduction.reduce_captures(
        scientific_sha256="0" * 64,
        plan_sha256="0" * 64,
        draws_by_arm={"reference": dict(mixture.draws)},
        state_weights=dict(mixture.stratum_weights),
        required_arms=("reference",),
        reference_arm="reference",
        views=("welfare", "gross"),
        reducer_provenance={"commit": "fixture"},
        view_map=recording,
    )
    assert mapped == [["welfare", "gross"]]
    # Each view's statistics are its own, whatever order the map ran them in.
    metrics = reduced_arm(report["arms"]["reference"])["metrics"]
    for view in ("welfare", "gross"):
        direct = reduction.mixture_view_metrics(
            reduction._view_mixture(mixture.draws, mixture.stratum_weights, view),
            view,
            reduction.DEFAULT_TAIL_LEVELS,
            TailGateControls(),
        )
        metric = available_metric(metrics[view])
        assert metric["expected_annual_loss"] == direct["expected_annual_loss"]
        # The reported tail records add the batch-means intervals of the
        # view's tail diagnostics.
        tails = direct["tail_diagnostics"]["tails"]
        assert set(metric["tail_metrics"]) == set(direct["tail_metrics"])
        for level, row in metric["tail_metrics"].items():
            intervals = ("var_numerical_interval_95", "tvar_numerical_interval_95")
            assert {
                key: value for key, value in row.items() if key not in intervals
            } == direct["tail_metrics"][level]
            for field in intervals:
                assert row[field] == tails[level][field]


def _restamp(capture, change):
    record = json.loads((capture / "run-capture.json").read_text())
    change(record)
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    (capture / "run-capture.json").write_text(json.dumps(record))


def test_a_tampered_record_is_checked_even_beside_an_untampered_copy(tmp_path):
    # The comparison arm reaches the reference arm's years on the shared state,
    # so its record of that state equals the reference arm's. A work limit
    # tampered in the reference and follower records, which stay equal to each
    # other, differs only in the execution record from that untampered copy.
    gate = {
        "batches": 4,
        "max_reference_state_years": 400,
        "max_comparison_state_years": 800,
        "levels": [0.999],
    }
    package = _arms_package(tmp_path, gate)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    record = json.loads((capture / "run-capture.json").read_text())
    (high,) = record["arms"]["reference"]
    assert record["arms"]["both"][high] == record["arms"]["reference"][high]

    def tamper(record):
        for arm in ("reference", "follower"):
            record["arms"][arm][high]["execution"]["work_limits"][
                "vectorized_initiators"
            ] += 1

    _restamp(capture, tamper)
    with pytest.raises(ValueError, match="work allowances per year times"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


def test_a_follower_capture_is_checked_against_the_selected_setups_laws(tmp_path):
    # Captured under a setup where the follower only reweights, replayed under
    # one that gives it its own law: its records no longer match its law.
    package = _arms_package(tmp_path, _GATE)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    changed = _arms_package(tmp_path, _GATE, follower_rate=0.77, name="changed")
    with pytest.raises(ValueError, match="capture conditional law"):
        replay_capture(
            setup_path=changed.root,
            capture_directory=tmp_path / "run" / "capture",
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


def test_reproduction_allows_last_bit_differences_and_nothing_else():
    from catmodel.results.results_export import _reproduces

    record = {
        "status": "available",
        "years": 4096,
        "converged": True,
        "value": 1.0e12,
        "reference": 3.6e12,
        "alternative": 3.7e12,
        "probability": 0.001,
        # A near-zero difference of large operands, and its ratio to one.
        "absolute_difference": 2.5,
        "half_width": 0.5,
        "relative_change": 0.0,
        "interval": [0.9e12, 1.1e12],
    }
    close = {
        **record,
        "value": 1.0e12 * (1 + 1e-12),
        # Last bits of the operands move the difference by about 1e-12 of
        # them, far more than 1e-9 of the difference itself.
        "absolute_difference": 2.5 + 3.7e12 * 1e-12,
        "half_width": 0.5 + 3.7e12 * 1e-12,
        "relative_change": 1e-12,
        "interval": [0.9e12 * (1 + 1e-12), 1.1e12],
    }
    assert _reproduces(record, close)
    for key, value in (
        ("value", 1.0e12 * (1 + 1e-6)),
        # Only differences, half-widths and ratios are scaled to more than
        # themselves.
        ("probability", 0.001 + 1e-6),
        ("absolute_difference", 2.5 + 3.7e12 * 1e-6),
        ("half_width", 0.5 + 3.7e12 * 1e-6),
        ("relative_change", 1e-6),
        ("status", "withheld"),
        ("years", 4097),
        ("converged", 1),
        ("interval", [0.9e12]),
    ):
        assert not _reproduces(record, {**record, key: value}), key
    assert not _reproduces(record, {**record, "extra": None})


@pytest.mark.parametrize(
    ("record", "key", "last_bits", "forged"),
    [
        # A difference of near-equal operands beside a large integer count and
        # a variance in squared units, which must not widen its tolerance.
        (
            {
                "reference": 371.0,
                "alternative": 372.0,
                "absolute_difference": 1.0,
                "half_width": 0.0,
                "years_per_state": 3000,
                "annual_loss_variance": 86016.0,
            },
            "absolute_difference",
            1.0 + 372.0 * 1e-12,
            1.0 + 5e-5,
        ),
        # Forgeries: a half-width from zero beside 3,000 years ...
        (
            {
                "reference": 0.5,
                "alternative": 0.5,
                "absolute_difference": 0.0,
                "half_width": 0.0,
                "relative_change": 0.0,
                "years_per_state": 3000,
            },
            "half_width",
            0.0,
            2.9e-6,
        ),
        # ... the EAL's reconciliation with the empirical mean beside its
        # variance ...
        (
            {
                "expected_annual_loss": 371.2,
                "normalized_empirical_mean": 371.0,
                "eal_reconciliation_difference": 0.2,
                "annual_loss_variance": 86016.0,
                "annual_probability_positive": 1.0,
            },
            "eal_reconciliation_difference",
            0.2 + 371.2 * 1e-12,
            0.2 + 5e-5,
        ),
        # ... and forged to zero beside a variance of 10^17.
        (
            {
                "expected_annual_loss": 1e7,
                "normalized_empirical_mean": 1e7 - 5e4,
                "eal_reconciliation_difference": 5e4,
                "annual_loss_variance": 1e17,
            },
            "eal_reconciliation_difference",
            5e4 + 1e7 * 1e-12,
            0.0,
        ),
        # A tornado row's difference of near-equal EALs.
        (
            {
                "control_eal": 3.6e12,
                "alternative_eal": 3.6e12 + 2.5,
                "absolute_difference": 2.5,
                "relative_change": 7e-13,
            },
            "absolute_difference",
            2.5 + 3.6e12 * 1e-12,
            2.5 + 3.6e12 * 1e-6,
        ),
        # A status's half-width is relative to its estimate, the half-width
        # over the relative half-width.
        (
            {"half_width": 10.0, "relative_half_width": 1e-5, "status": "converged"},
            "half_width",
            10.0 + 1e6 * 1e-12,
            10.0 * 1.001,
        ),
        # A standard error is relative to its estimate.
        (
            {
                "tvar": 1e12,
                "tvar_numerical_standard_error": 1e9,
                "positive_year_count": 3000,
            },
            "tvar_numerical_standard_error",
            1e9 + 2.0,
            1e9 + 1e4,
        ),
        # A respondent-sampling standard error of a level is relative to its
        # EAL, of a change to the change's two EALs, and its z to those EALs
        # over the standard error, times one plus |z|.
        (
            {
                "expected_annual_loss": 2e10,
                "sampling_standard_error": 1.0,
                "relative_sampling_standard_error": 5e-11,
                "groups": 7,
            },
            "sampling_standard_error",
            1.0 + 2e10 * 1e-12,
            1.0 + 2e10 * 1e-6,
        ),
        (
            {
                "control_eal": 3.6e12,
                "alternative_eal": 3.6e12 + 2.5,
                "absolute_difference": 2.5,
                "sampling_standard_error": 1.0,
                "sampling_z": 2.5,
            },
            "sampling_standard_error",
            1.0 + 3.6e12 * 1e-12,
            1.0 + 3.6e12 * 1e-6,
        ),
        (
            {
                "control_eal": 1e12,
                "alternative_eal": 1.1e12,
                "absolute_difference": 1e11,
                "sampling_standard_error": 4e10,
                "sampling_z": 2.5,
            },
            "sampling_z",
            2.5 + 3.5 * 25 * 1e-12,
            2.5 + 1e-5,
        ),
        (
            {
                "expected_annual_loss": 4e10,
                "embedded_rule_expected_annual_loss": 4e10 + 1.0,
                "absolute_difference": -1.0,
                "sampling_standard_error": 1.0,
                "sampling_z": -1.0,
            },
            "sampling_z",
            -1.0 - 2 * 4e10 * 1e-12,
            -1.0 - 2 * 4e10 * 1e-6,
        ),
        # A quadrature node's difference contribution is relative to its two
        # contributions, and its standard deviation to its mean.
        (
            {
                "full_contribution": 1e10,
                "embedded_contribution": 1e10 - 5.0,
                "difference_contribution": 5.0,
                "samples": 5,
            },
            "difference_contribution",
            5.0 + 1e10 * 1e-12,
            5.0 + 1e10 * 1e-6,
        ),
        (
            {"state_eal_mean": 3e10, "state_eal_sd": 1.0, "samples": 5},
            "state_eal_sd",
            1.0 + 3e10 * 1e-12,
            1.0 + 3e10 * 1e-6,
        ),
    ],
)
def test_reproduction_scales_a_number_by_its_own_operands_only(
    record, key, last_bits, forged
):
    from catmodel.results.results_export import _reproduces

    assert _reproduces({**record, key: last_bits}, record)
    assert not _reproduces({**record, key: forged}, record)


def test_the_replayed_record_sets_the_scale():
    from catmodel.results.results_export import _reproduces

    # A relative half-width of 1e-9 may move by 1e-9; reported at 1e-18 it
    # would make the estimate 10^15 and let a forged half-width through.
    replayed = {"half_width": 1e-3, "relative_half_width": 1e-9, "status": "converged"}
    assert not _reproduces(
        {**replayed, "half_width": 1.001, "relative_half_width": 1e-18}, replayed
    )


def test_reproduction_scales_interval_endpoints_by_the_interval():
    from catmodel.results.results_export import _reproduces

    # A lower endpoint near zero moves with the estimate's last bits.
    record = {"tvar": 1.05e12, "tvar_numerical_interval_95": [1e8, 2.1e12]}
    nudged = {**record, "tvar_numerical_interval_95": [1e8 + 1.0, 2.1e12]}
    assert _reproduces(nudged, record)
    forged = {**record, "tvar_numerical_interval_95": [1e8 + 1e4, 2.1e12]}
    assert not _reproduces(forged, record)


def _nudged(value):
    """Every float moved by one part in 10^12, as another CPU's BLAS might."""
    if isinstance(value, dict):
        return {key: _nudged(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nudged(item) for item in value]
    if isinstance(value, float) and not isinstance(value, bool):
        return value * (1 + 1e-12)
    return value


def test_export_accepts_another_machines_last_bits_but_not_a_changed_result(tmp_path):
    package = _arms_package(tmp_path, _GATE)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    results = tmp_path / "run" / "results.json"
    report = json.loads(results.read_bytes())
    nudged = tmp_path / "nudged.json"
    # The analysis identity records inputs, which another machine reads alike.
    nudged.write_text(
        json.dumps({**_nudged(report), "analysis_spec": report["analysis_spec"]})
    )
    # The exhibits, in the diagnostics file beside the results, move with it.
    diagnostics = json.loads(diagnostics_path(results).read_bytes())
    diagnostics_path(nudged).write_text(json.dumps(_nudged(diagnostics)))
    export_results(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        results_path=nudged,
        output=tmp_path / "export",
        code_commit="fixture",
        workers=2,
    )
    # An EAL changed by 1e-6, and the reconciliation of the EAL (120) with the
    # empirical mean changed by 5e-6: 40 times its tolerance relative to its
    # operands, though within 1e-9 of the record's variance (22,690).
    for index, (key, change) in enumerate(
        (
            ("expected_annual_loss", lambda value: value * (1 + 1e-6)),
            ("eal_reconciliation_difference", lambda value: value + 5e-6),
        )
    ):
        changed = json.loads(json.dumps(report))
        metric = changed["arms"]["reference"]["metrics"]["gross"]
        metric[key] = change(metric[key])
        forged = tmp_path / f"changed-{index}.json"
        forged.write_text(json.dumps(changed))
        diagnostics_path(forged).write_text(json.dumps(diagnostics))
        with pytest.raises(ValueError, match="do not reproduce"):
            export_results(
                setup_path=package.root,
                capture_directory=tmp_path / "run" / "capture",
                results_path=forged,
                output=tmp_path / f"refused-{index}",
                code_commit="fixture",
            )
    # An exhibit, which the diagnostics file holds, is checked like the
    # headline: a changed count of simulated years does not reproduce.
    forged = tmp_path / "changed-exhibit.json"
    forged.write_text(json.dumps(report))
    changed = json.loads(json.dumps(diagnostics))
    changed["workload"]["by_arm"]["reference"]["simulated_conditional_years"] += 1
    diagnostics_path(forged).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="do not reproduce"):
        export_results(
            setup_path=package.root,
            capture_directory=tmp_path / "run" / "capture",
            results_path=forged,
            output=tmp_path / "refused-exhibit",
            code_commit="fixture",
        )
    # The view diagnostics are not compared, and the export holds none: it
    # carries the exhibits as reported and nothing its check does not cover.
    exported = tmp_path / "export" / "native" / "results.json"
    stored = json.loads(diagnostics_path(exported).read_bytes())
    assert diagnostics["arms"] and stored["arms"] == {}
    assert exported.read_bytes() == nudged.read_bytes()
    assert stored["workload"] == _nudged(diagnostics)["workload"]


@pytest.mark.parametrize("failing", ["reduce_capture", "tornado_svg"])
def test_a_run_whose_results_fail_keeps_its_capture_for_replay(
    tmp_path, monkeypatch, failing
):
    # The reduction fails, or the last result file cannot be written after
    # results.json and its diagnostics file were, as on a full disk.
    package = _arms_package(tmp_path, _GATE)
    original = getattr(execution, failing)

    def fail(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(execution, failing, fail)
    output = tmp_path / "run"
    with pytest.raises(execution.IncompleteRun) as raised:
        run_setup(setup_path=package.root, output=output, code_commit="fixture")
    message = str(raised.value)
    assert "OSError: No space left on device" in message
    assert f"saved in {output / 'capture'}" in message
    assert f"catmodel results replay --setup {package.root}" in message
    assert (output / "capture" / "run-capture.json").exists()
    assert sorted(path.name for path in output.iterdir()) == [
        "INCOMPLETE.txt",
        "capture",
    ]
    note = (output / execution.INCOMPLETE_RUN_NOTE).read_text()
    assert "results replay" in note and "No space left on device" in note
    assert not list(tmp_path.glob(".run-*"))
    monkeypatch.setattr(execution, failing, original)
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=output / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert "gross" in reduced_arm(replay["arms"]["reference"])["metrics"]


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("taken", ["directory", "file"])
def test_a_finished_run_whose_output_was_taken_is_kept_whole_beside_it(
    tmp_path, monkeypatch, taken
):
    package = _arms_package(tmp_path, _GATE)
    run_setup(setup_path=package.root, output=tmp_path / "clean", code_commit="fixture")
    output = tmp_path / "run"
    tornado_svg = execution.tornado_svg

    def racing(summary):
        # Another process creates the output while the run derives its results.
        if taken == "file":
            output.write_text("theirs")
        else:
            (output / "theirs").mkdir(parents=True)
        return tornado_svg(summary)

    monkeypatch.setattr(execution, "tornado_svg", racing)
    with pytest.raises(execution.IncompleteRun) as raised:
        run_setup(setup_path=package.root, output=output, code_commit="fixture")
    [saved] = tmp_path.glob("run.unpublished-*")
    message = str(raised.value)
    assert f"could not be moved to {output}" in message
    assert f"saved whole in {saved}" in message
    replay = f"results replay --setup {package.root} --capture {saved / 'capture'}"
    assert replay in message
    # The run is whole, as the same run publishes it at an output that is free.
    assert _files(saved) == _files(tmp_path / "clean")
    # The other process's output is as it left it, and nothing stays staged.
    if taken == "file":
        assert output.read_text() == "theirs"
    else:
        assert list(output.iterdir()) == [output / "theirs"]
    assert not list(tmp_path.glob(".run-*"))


def test_a_run_whose_results_fail_keeps_its_capture_beside_a_taken_output(
    tmp_path, monkeypatch
):
    package = _arms_package(tmp_path, _GATE)
    output = tmp_path / "run"

    def fail(*args, **kwargs):
        (output / "theirs").mkdir(parents=True)
        raise OSError("No space left on device")

    monkeypatch.setattr(execution, "reduce_capture", fail)
    with pytest.raises(execution.IncompleteRun) as raised:
        run_setup(setup_path=package.root, output=output, code_commit="fixture")
    [saved] = tmp_path.glob("run.unpublished-*")
    message = str(raised.value)
    assert "OSError: No space left on device" in message
    assert f"saved in {saved / 'capture'}" in message
    assert sorted(path.name for path in saved.iterdir()) == [
        "INCOMPLETE.txt",
        "capture",
    ]
    note = (saved / execution.INCOMPLETE_RUN_NOTE).read_text()
    assert f"--capture {saved / 'capture'}" in note
    assert list(output.iterdir()) == [output / "theirs"]
    assert not list(tmp_path.glob(".run-*"))


def test_a_finished_run_that_cannot_be_moved_stays_where_it_is_staged(
    tmp_path, monkeypatch
):
    package = _arms_package(tmp_path, _GATE)
    output = tmp_path / "run"
    rename = Path.rename

    def refused(self, target):
        # Neither the output nor a directory beside it may take the run.
        if self.parent.name.startswith(".run-"):
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", refused)
    with pytest.raises(execution.IncompleteRun) as raised:
        run_setup(setup_path=package.root, output=output, code_commit="fixture")
    [staged] = tmp_path.glob(".run-*/run")
    message = str(raised.value)
    assert "PermissionError" in message and f"saved whole in {staged}" in message
    assert not output.exists() and not list(tmp_path.glob("run.unpublished-*"))
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=staged / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert "gross" in reduced_arm(replay["arms"]["reference"])["metrics"]
