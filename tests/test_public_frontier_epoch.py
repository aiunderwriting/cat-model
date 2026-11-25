"""Epoch ECI release capture, pooled discrepancy estimate and the captured data."""

import csv
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import catmodel.forecast.public_frontier as public_frontier_calibration
import catmodel.forecast.public_frontier_quadrature as quadrature_module
from catmodel.contracts.governance import SourceScaleAnchor
from catmodel.contracts.public_frontier import (
    YEAR_SECONDS,
    PublicFrontierForecast,
    forecast_times,
    progress_elapsed_years,
)
from catmodel.forecast.public_frontier import (
    PACE_CAP_ALTERNATIVES,
    compare_public_frontier,
    estimate_trend_discrepancy,
    fit_public_trend,
    lab_frontier_records,
    observed_frontier,
    one_year_before,
    pace_quantile,
    record_residuals,
)
from catmodel.forecast.public_frontier_epoch import (
    EPOCH_ATTRIBUTION,
    EPOCH_ECI_SCORES_PATH,
    EPOCH_INTERVAL_Z,
    epoch_eci_releases,
)
from catmodel.forecast.public_frontier_quadrature import (
    _laws,
    public_frontier_quadrature,
)
from catmodel.setup.public_study import prepare_public_forecast_file
from catmodel.yaml_io import safe_load
from test_support.capability_sources import authored_source_files, write_source_bundle
from test_support.public_frontier import synthetic_forecast, synthetic_release

DATA = Path(__file__).resolve().parents[1] / "data" / "capability"
PROVENANCE = safe_load((DATA / "epoch_eci_scores.provenance.yaml").read_text())
ANCHORS = tuple(
    SourceScaleAnchor.model_validate(row) for row in PROVENANCE["scale_anchors"]
)
HEADER = (
    "Model,Display name,eci,eci_ci_low,eci_ci_high,date,Organization,"
    "Country (of organization),Model accessibility,Accessibility group,"
    "model_versions\n"
)
AS_OF = datetime(2026, 10, 29, tzinfo=UTC)
REF = "evidence_file:evidence/epoch_eci_scores.csv#sha256=" + "a" * 64


def _row(model, eci, low, high, date, org, access="API access"):
    return f'{model},{model},{eci},{low},{high},{date},"{org}",US,{access},Closed,\n'


def test_epoch_rows_map_labs_eligibility_dates_and_interval_errors():
    text = HEADER + "".join(
        (
            _row("Model A", 150, 147, 153.29, "2026-01-02", "OpenAI"),
            _row("Gemini X", 149, 148, 150, "2026-02-01", "Google DeepMind,Google"),
            _row("Claude 3.5 Sonnet", 130, "", "", "2024-06-20", "Anthropic"),
            _row("Hidden", 160, 159, 161, "2026-03-01", "OpenAI", "Unreleased"),
            _row("Unknown access", 161, 160, 162, "2026-03-01", "xAI", "Unknown"),
            _row("Other lab", 170, 169, 171, "2026-03-01", "DeepSeek"),
            _row("Future", 171, 170, 172, "2026-11-01", "OpenAI"),
            _row(
                "Open weights",
                140,
                139,
                141,
                "2026-04-01",
                "Meta AI",
                "Open weights (unrestricted)",
            ),
        )
    )
    releases = epoch_eci_releases(
        text,
        evidence_ref=REF,
        labs=("openai", "anthropic", "google", "meta", "xai"),
        as_of=AS_OF,
        scale_anchors=ANCHORS,
    )
    by_id = {row.release_id: row for row in releases}
    assert set(by_id) == {"model-a", "gemini-x", "claude-3-5-sonnet", "open-weights"}
    assert by_id["gemini-x"].lab_id == "google"
    assert by_id["model-a"].available_at == datetime(2026, 1, 2, tzinfo=UTC)
    assert by_id["model-a"].benchmark_error_sd == pytest.approx(
        6.29 / (2 * EPOCH_INTERVAL_Z)
    )
    assert by_id["claude-3-5-sonnet"].benchmark_error_sd == 0
    assert all(row.availability_basis == "release_date_proxy" for row in releases)
    assert all(row.evidence_refs == (REF,) for row in releases)


@pytest.mark.parametrize(
    "text,match",
    [
        ("Model,eci\nA,1\n", "lacks columns"),
        (HEADER + _row("A", "", 1, 2, "2026-01-01", "OpenAI"), "score or date"),
        (HEADER + _row("A", 150, 149, "", "2026-01-01", "OpenAI"), "one-sided"),
    ],
)
def test_malformed_selected_rows_are_rejected(text, match):
    with pytest.raises(ValueError, match=match):
        epoch_eci_releases(
            text,
            evidence_ref=REF,
            labs=("openai",),
            as_of=AS_OF,
            scale_anchors=ANCHORS,
        )


@pytest.mark.parametrize(
    ("model", "score", "organization", "accepted"),
    [
        ("Claude 3.5 Sonnet", 130, "Anthropic", True),
        ("GPT-5", 150, "OpenAI", True),
        # A missing interval on any other model, or on an anchor's name at a
        # score or organization other than its anchored one, is not zero
        # benchmark error.
        ("GPT-5.1", 152, "OpenAI", False),
        ("GPT-5", 151, "OpenAI", False),
        ("GPT-5", 150, "Anthropic", False),
        ("Claude 3.5 Sonnet (October 2024)", 130, "Anthropic", False),
    ],
)
def test_only_the_provenance_scale_anchors_may_lack_an_interval(
    model, score, organization, accepted
):
    text = HEADER + _row(model, score, "", "", "2025-08-07", organization)
    labs = ("openai", "anthropic")
    if accepted:
        (release,) = epoch_eci_releases(
            text, evidence_ref=REF, labs=labs, as_of=AS_OF, scale_anchors=ANCHORS
        )
        assert release.benchmark_error_sd == 0
        assert "fixed scale anchor" in release.rationale
        return
    with pytest.raises(ValueError, match="not one of the scale anchors"):
        epoch_eci_releases(
            text, evidence_ref=REF, labs=labs, as_of=AS_OF, scale_anchors=ANCHORS
        )
    # Without anchors in the provenance, no row may lack an interval.
    with pytest.raises(ValueError, match="not one of the scale anchors"):
        epoch_eci_releases(
            HEADER + _row("GPT-5", 150, "", "", "2025-08-07", "OpenAI"),
            evidence_ref=REF,
            labs=labs,
            as_of=AS_OF,
            scale_anchors=(),
        )


def test_labs_without_an_organization_mapping_are_rejected():
    with pytest.raises(ValueError, match="no Epoch organization mapping"):
        epoch_eci_releases(
            HEADER,
            evidence_ref=REF,
            labs=("lab_a",),
            as_of=AS_OF,
            scale_anchors=ANCHORS,
        )


def _bundle(tmp_path, csv_bytes, *, provenance=True, drop=None, correspondence=False):
    files = {**authored_source_files(), EPOCH_ECI_SCORES_PATH: csv_bytes}
    source = {
        "source_id": "epoch-eci-scores",
        "provider": "Epoch AI",
        "uri": "https://epoch.ai/data/eci_scores.csv",
        "content_hash": hashlib.sha256(csv_bytes).hexdigest(),
        "pulled_at": "2026-10-29T23:15:06Z",
        "license_access_basis": "CC BY 4.0; credit Epoch AI",
        "captured_paths": [EPOCH_ECI_SCORES_PATH],
        "scale_anchors": PROVENANCE["scale_anchors"],
        "scale_anchor_basis": PROVENANCE["scale_anchor_basis"],
    }
    if correspondence:
        source["scale_correspondence"] = PROVENANCE["scale_correspondence"]
        source["scale_correspondence_basis"] = PROVENANCE["scale_correspondence_basis"]
    if drop == "scale-anchors":
        source.pop("scale_anchors")
        source.pop("scale_anchor_basis")
    elif drop == "not-cc-by":
        source["license_access_basis"] = "Internal use only"
    elif drop is not None:
        source.pop(drop)
    sources = [source] if provenance else []
    return write_source_bundle(tmp_path / "bundle", files=files, extra_sources=sources)


def test_committed_provenance_file_is_complete_and_credits_epoch():
    from catmodel.contracts.governance import ExternalSourceSnapshot
    from catmodel.yaml_io import safe_load

    metadata = safe_load((DATA / "epoch_eci_scores.provenance.yaml").read_text())
    source = ExternalSourceSnapshot.model_validate(
        {
            **metadata,
            "content_hash": hashlib.sha256(
                (DATA / "epoch_eci_scores.csv").read_bytes()
            ).hexdigest(),
        }
    )
    assert source.uri == "https://epoch.ai/data/eci_scores.csv"
    assert source.pulled_at and "CC BY 4.0" in (source.license_access_basis or "")
    assert "Epoch AI" in (source.license_access_basis or "")
    assert source.record_count == 273
    # The anchors are the table's only interval-less rows, at their scores.
    assert source.scale_anchors is not None and "https://epoch.ai/eci" in (
        source.scale_anchor_basis or ""
    )
    rows = csv.DictReader(io.StringIO((DATA / "epoch_eci_scores.csv").read_text()))
    assert {
        (row["Model"], row["Organization"], float(row["eci"]))
        for row in rows
        if not row["eci_ci_low"]
    } == {(row.model, row.organization, row.value) for row in source.scale_anchors}
    with pytest.raises(ValueError, match="given together"):
        ExternalSourceSnapshot.model_validate(
            {**metadata, "content_hash": "a", "scale_anchor_basis": None}
        )
    # What Epoch reports a gain in points to have corresponded to, with its
    # citation; a record without either leaves both out of what it stores.
    correspondence = source.scale_correspondence
    assert correspondence is not None and correspondence.points_per_doubling == 5.0
    assert "METR's time horizon" in correspondence.measure
    assert "https://epoch.ai/eci" in (source.scale_correspondence_basis or "")
    with pytest.raises(ValueError, match="given together"):
        ExternalSourceSnapshot.model_validate(
            {**metadata, "content_hash": "a", "scale_correspondence_basis": None}
        )
    bare = {
        key: value
        for key, value in metadata.items()
        if not key.startswith("scale_correspondence")
    }
    stored = ExternalSourceSnapshot.model_validate(
        {**bare, "content_hash": "a"}
    ).model_dump(mode="json")
    assert "scale_anchors" in stored
    assert not any(key.startswith("scale_correspondence") for key in stored)


def test_prepare_public_refuses_an_interval_less_row_without_provenance_anchors(
    tmp_path,
):
    with pytest.raises(ValueError, match="not one of the scale anchors"):
        prepare_public_forecast_file(
            source_bundle=_bundle(
                tmp_path,
                (DATA / "epoch_eci_scores.csv").read_bytes(),
                drop="scale-anchors",
            ),
            template=DATA / "public-frontier-forecast-template.json",
            output=tmp_path / "forecast.json",
        )


@pytest.mark.parametrize("missing", ["pulled_at", "license_access_basis", "not-cc-by"])
def test_prepare_public_requires_uri_retrieval_time_and_cc_by(tmp_path, missing):
    csv_bytes = (DATA / "epoch_eci_scores.csv").read_bytes()
    bundle = _bundle(tmp_path, csv_bytes, drop=missing)
    with pytest.raises(ValueError, match="CC BY license basis"):
        prepare_public_forecast_file(
            source_bundle=bundle,
            template=DATA / "public-frontier-forecast-template.json",
            output=tmp_path / "forecast.json",
        )


def test_prepare_public_requires_captured_provenance_and_cites_the_bundle(tmp_path):
    csv_bytes = (DATA / "epoch_eci_scores.csv").read_bytes()
    template = DATA / "public-frontier-forecast-template.json"
    with pytest.raises(ValueError, match="provenance"):
        prepare_public_forecast_file(
            source_bundle=_bundle(tmp_path / "bare", csv_bytes, provenance=False),
            template=template,
            output=tmp_path / "bare.json",
        )
    prepared = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path, csv_bytes),
        template=template,
        output=tmp_path / "forecast.json",
    )
    forecast = PublicFrontierForecast.model_validate_json(
        (tmp_path / "forecast.json").read_text()
    )
    assert forecast == prepared
    digest = hashlib.sha256(csv_bytes).hexdigest()
    assert all(
        row.evidence_refs == (f"evidence_file:{EPOCH_ECI_SCORES_PATH}#sha256={digest}",)
        for row in forecast.releases
    )
    assert forecast.trend_discrepancy_sd is None
    # The CC BY credit travels with the forecast into ensembles and prompts.
    assert EPOCH_ATTRIBUTION in forecast.limitations
    assert all("CC BY 4.0" in row.rationale for row in forecast.releases)
    payload = json.loads(template.read_text())
    payload["releases"] = []
    bad = tmp_path / "bad-template.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="without releases"):
        prepare_public_forecast_file(
            source_bundle=tmp_path / "bundle",
            template=bad,
            output=tmp_path / "bad.json",
        )


def test_prepare_public_carries_the_provenance_scale_correspondence(tmp_path):
    """The forecast carries the correspondence its table's provenance records,
    and one prepared from a provenance without it is the forecast prepared
    before the field existed: the same record and the same digest."""
    csv_bytes = (DATA / "epoch_eci_scores.csv").read_bytes()
    template = DATA / "public-frontier-forecast-template.json"
    plain = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path / "plain", csv_bytes),
        template=template,
        output=tmp_path / "plain.json",
    )
    assert plain.scale_correspondence is None
    assert "scale_correspondence" not in json.loads(
        (tmp_path / "plain.json").read_text()
    )
    stated = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path / "stated", csv_bytes, correspondence=True),
        template=template,
        output=tmp_path / "stated.json",
    )
    assert stated.scale_correspondence is not None
    assert (
        stated.scale_correspondence.model_dump(mode="json")
        == PROVENANCE["scale_correspondence"]
    )
    assert json.loads((tmp_path / "stated.json").read_text())[
        "scale_correspondence"
    ] == {**PROVENANCE["scale_correspondence"]}
    assert stated.digest() != plain.digest()
    assert stated.model_copy(update={"scale_correspondence": None}) == plain


def test_captured_epoch_data_reproduces_the_documented_fit(tmp_path):
    """The figures in methodology.md and assumptions.md come from this fit."""
    forecast = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path, (DATA / "epoch_eci_scores.csv").read_bytes()),
        template=DATA / "public-frontier-forecast-template.json",
        output=tmp_path / "forecast.json",
    )
    posterior = fit_public_trend(forecast)
    assert len(forecast.releases) == 141
    assert sum(lab.observation_count for lab in posterior.labs.values()) == 44
    assert posterior.trend_discrepancy_basis == "pooled_reml_estimate"
    assert posterior.trend_discrepancy_sd == pytest.approx(1.64, abs=0.01)
    assert posterior.trend_discrepancy_interval == pytest.approx((1.12, 2.27), abs=0.01)
    assert posterior.rate_location == pytest.approx(16.15, abs=0.01)
    assert posterior.rate_variance**0.5 == pytest.approx(0.63, abs=0.01)
    assert max(lab.as_of_value for lab in posterior.labs.values()) == 168.92
    residuals = record_residuals(forecast).values()
    pooled = np.sqrt(
        sum(row["count"] * row["leave_one_out_rms"] ** 2 for row in residuals)
        / sum(row["count"] for row in residuals)
    )
    assert pooled == pytest.approx(1.04, abs=0.01)


def test_captured_epoch_data_reproduces_the_documented_basis_of_the_pace_cap(tmp_path):
    """The record behind the default pace cap of 4 (docs/capability-frontiers.md
    and assumptions.md, OBS-PUBLIC-JOINT): each roster lab's fastest twelve months
    and fastest six months, as multiples of the pooled trend rate. A lab's
    frontier at a date is its highest release score available by then."""
    forecast = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path, (DATA / "epoch_eci_scores.csv").read_bytes()),
        template=DATA / "public-frontier-forecast-template.json",
        output=tmp_path / "forecast.json",
    )
    assert "pace_cap" not in json.loads(
        (DATA / "public-frontier-forecast-template.json").read_text()
    )
    assert forecast.pace_cap == 4
    rate = fit_public_trend(forecast).rate_location
    assert rate == pytest.approx(16.15, abs=0.01)

    def fastest(years):
        """Each lab's largest annualized frontier gain over a span of ``years``
        ending at one of its records."""
        span = timedelta(seconds=years * YEAR_SECONDS)
        gains = {}
        for lab, rows in lab_frontier_records(forecast).items():
            gains[lab] = max(
                (end.value - earlier[-1].value) / years
                for end in rows
                if (
                    earlier := [
                        row
                        for row in rows
                        if row.available_at <= end.available_at - span
                    ]
                )
            )
        return gains

    year, half = fastest(1.0), fastest(0.5)
    assert {lab: round(gain / rate, 2) for lab, gain in year.items()} == {
        "openai": 1.30,
        "anthropic": 1.52,
        "google": 1.66,
        "meta": 1.83,
        "xai": 1.34,
    }
    # Meta's twelve months to 2026-10-12 and six months to 2026-09-02, both
    # after a year without a frontier-raising release.
    assert year["meta"] == pytest.approx(29.62, abs=0.01)
    assert max(half, key=lambda lab: half[lab]) == "meta"
    assert half["meta"] / rate == pytest.approx(3.09, abs=0.01)
    assert sorted(round(gain / rate, 2) for gain in half.values())[:-1] == [
        1.58,
        1.69,
        1.81,
        1.96,
    ]
    # The cap is about twice the fastest lab-year and above the fastest
    # half-year; the lower-cap alternative is about the fastest lab-year.
    assert forecast.pace_cap / (year["meta"] / rate) == pytest.approx(2.18, abs=0.01)
    assert PACE_CAP_ALTERNATIVES == {"lower_pace_cap": 2.0}
    assert PACE_CAP_ALTERNATIVES["lower_pace_cap"] / (year["meta"] / rate) == (
        pytest.approx(1.09, abs=0.01)
    )
    # What the cap removes of the log-logistic law at shapes 5 and 3, and the
    # capped median at shape 5.
    assert [1 / (1 + 4**shape) for shape in (5, 3)] == pytest.approx(
        [0.00098, 0.0154], abs=0.0001
    )
    assert pace_quantile(5, np.array([0.5]), 4)[0] == pytest.approx(0.9996, abs=5e-5)


def test_captured_epoch_data_reproduces_the_documented_scale_reference_and_tail(
    tmp_path,
):
    """The scale reference every question quotes, and the figures in
    docs/limitations.md on the 15-node rule's three fastest trajectories."""
    forecast = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path, (DATA / "epoch_eci_scores.csv").read_bytes()),
        template=DATA / "public-frontier-forecast-template.json",
        output=tmp_path / "forecast.json",
    )
    assert forecast.scale_anchors == ANCHORS
    assert one_year_before(forecast.as_of) == datetime(2025, 10, 29, tzinfo=UTC)
    assert observed_frontier(forecast, one_year_before(forecast.as_of)) == 150.66
    quadrature = public_frontier_quadrature(
        forecast,
        period_start=datetime(2027, 1, 1, tzinfo=UTC),
        period_end=datetime(2028, 1, 1, tzinfo=UTC),
        nodes=15,
    )
    laws = {law.label: law for law in quadrature.alternatives}
    wider, lower_cap = laws["lower_pace_shape"], laws["lower_pace_cap"]
    assert wider.pace_shape == 3 and forecast.pace_shape == 5
    assert (wider.pace_cap, lower_cap.pace_cap, forecast.pace_cap) == (4, 2, 4)
    fastest = quadrature.nodes[-3:]
    assert [max(node.values[-1]) for node in fastest] == pytest.approx(
        [233.9, 239.7, 247.1], abs=0.1
    )
    # Their gain over the as-of frontier, against the year before as-of.
    assert [max(node.values[-1]) - 168.92 for node in fastest] == pytest.approx(
        [65.0, 70.8, 78.2], abs=0.1
    )
    # The as-of frontier of 168.92 stands 18.26 points above 150.66.
    assert observed_frontier(forecast, forecast.as_of) == 168.92
    assert quadrature.base.weights[-3:] == pytest.approx(
        [7.66e-3, 1.57e-3, 2.20e-4], rel=0.01
    )
    assert wider.weights[-3:] == pytest.approx([4.59e-2, 1.26e-2, 1.99e-3], rel=0.01)
    # The lower cap reaches none of the three, and the fourth only at rates far
    # above the pooled one.
    assert lower_cap.weights[-3:] == (0, 0, 0)
    assert 0 < lower_cap.weights[-4] < 1e-12
    # No node asks about a lab faster than the cap.
    assert max(max(node.pace_multipliers) for node in quadrature.nodes) == (
        pytest.approx(3.72, abs=0.01)
    )
    # A loss growing like a power of frontier progress: the share of its EAL
    # that the three nodes supply.
    progress = np.asarray([node.progress for node in quadrature.nodes])

    def share(weights, power):
        terms = np.asarray(weights) * progress**power
        return terms[-3:].sum() / terms.sum()

    assert [share(quadrature.base.weights, power) for power in (2, 3, 4)] == (
        pytest.approx([0.054, 0.109, 0.195], abs=0.002)
    )
    assert [share(wider.weights, power) for power in (2, 3, 4)] == pytest.approx(
        [0.219, 0.335, 0.457], abs=0.002
    )
    # The seven-node rule is nested in this one: its fastest node is the second.
    seven = public_frontier_quadrature(
        forecast,
        period_start=datetime(2027, 1, 1, tzinfo=UTC),
        period_end=datetime(2028, 1, 1, tzinfo=UTC),
        nodes=7,
    )
    assert seven.nodes[-1].values == fastest[1].values
    # An untruncated pace has a power-law tail, and the rule's fastest node
    # follows it; a cap far beyond every node stands for no cap.
    untruncated = public_frontier_quadrature(
        forecast.model_copy(update={"pace_cap": 1e9}),
        period_start=datetime(2027, 1, 1, tzinfo=UTC),
        period_end=datetime(2028, 1, 1, tzinfo=UTC),
        nodes=15,
    )
    assert max(untruncated.nodes[-1].values[-1]) == pytest.approx(584.7, abs=0.1)
    # Five of the sixteen nodes carry under half a percent of base weight each.
    light = [weight for weight in quadrature.base.weights if weight < 0.005]
    assert len(light) == 5 and sum(light) == pytest.approx(0.0045, abs=1e-4)
    # The period-end leader of each node's path, by base weight.
    leaders: dict[str, float] = {}
    for node, weight in zip(quadrature.nodes[1:], quadrature.base.weights[1:]):
        lab = quadrature.coordinates[int(np.argmax(node.values[-1]))].lab_id
        leaders[lab] = leaders.get(lab, 0) + weight
    assert leaders == pytest.approx(
        {"openai": 0.499, "anthropic": 0.493, "meta": 0.0077, "google": 0.0001},
        abs=1e-3,
    )


def test_captured_epoch_data_reproduces_the_documented_rule_accuracy(
    tmp_path, monkeypatch
):
    """The accuracy figures in methodology.md ("Trajectory quadrature"): what
    the node rules recover of each pace alternative's effect on the mean of
    the frontier progress and of its powers, against each law integrated on a
    fine grid, and how far a threshold-like loss errs."""
    forecast = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path, (DATA / "epoch_eci_scores.csv").read_bytes()),
        template=DATA / "public-frontier-forecast-template.json",
        output=tmp_path / "forecast.json",
    )
    period = datetime(2027, 1, 1, tzinfo=UTC), datetime(2028, 1, 1, tzinfo=UTC)
    base, alternatives, _ = _laws(
        forecast,
        fit_public_trend(forecast),
        progress_elapsed_years(forecast.as_of, forecast_times(*period)),
    )
    grid = np.exp(np.linspace(np.log(1e-2), np.log(100.0), 6_001))
    density = {
        law.label: law.density(grid)
        for law in (base, *alternatives)
        if law.label in {"baseline", "lower_pace_shape", "lower_pace_cap"}
    }

    def exact(label, loss):
        return np.trapezoid(grid * density[label] * loss(grid), np.log(grid))

    for label in density:
        assert exact(label, np.ones_like) == pytest.approx(1, abs=1e-6)

    def rule(nodes):
        quadrature = public_frontier_quadrature(
            forecast, period_start=period[0], period_end=period[1], nodes=nodes
        )
        progress = np.asarray([node.progress for node in quadrature.nodes])
        laws = {law.label: law for law in (quadrature.base, *quadrature.alternatives)}

        def integrate(label, loss, embedded=False):
            law = laws[label]
            return np.asarray(law.embedded_weights if embedded else law.weights) @ (
                loss(progress)
            )

        return integrate, quadrature.median_progress

    def power(order):
        return lambda value: value**order

    def effects(integrate, label):
        return [
            integrate(label, power(order)) / integrate("baseline", power(order)) - 1
            for order in (1, 2)
        ]

    def exact_effects(label):
        return [
            exact(label, power(order)) / exact("baseline", power(order)) - 1
            for order in (1, 2)
        ]

    approx = pytest.approx
    assert exact_effects("lower_pace_shape") == approx([0.230, 0.609], abs=0.002)
    assert exact_effects("lower_pace_cap") == approx([-0.079, -0.190], abs=0.002)
    seven, median = rule(7)
    # Seven stretched nodes: the wider shape's effect, the base moments and
    # the lower cap, whose law falls away between two nodes.
    assert effects(seven, "lower_pace_shape") == approx([0.228, 0.612], abs=0.002)
    assert effects(seven, "lower_pace_cap") == approx([-0.063, -0.156], abs=0.002)
    for order, error in ((1, 0.0009), (2, 0.0011), (4, 0.0054)):
        assert seven("baseline", power(order)) / exact(
            "baseline", power(order)
        ) - 1 == (approx(error, abs=0.0005))
    assert seven("lower_pace_shape", power(4)) / exact(
        "lower_pace_shape", power(4)
    ) - 1 == approx(0.038, abs=0.002)
    fifteen, _ = rule(15)
    assert effects(fifteen, "lower_pace_shape") == approx([0.230, 0.608], abs=0.002)
    assert effects(fifteen, "lower_pace_cap") == approx([-0.084, -0.201], abs=0.002)

    # A logistic step of width 0.1 times the median progress, by its location:
    # the full rule's error on the base EAL and the embedded-rule indicator.
    def step(location):
        def loss(value):
            return 1 / (1 + np.exp(-(value - location * median) / (0.1 * median)))

        full = seven("baseline", loss)
        return (
            full / exact("baseline", loss) - 1,
            (full - seven("baseline", loss, embedded=True)) / full,
        )

    assert step(1.4) == approx((0.085, 0.323), abs=0.005)
    assert step(1.7) == approx((-0.172, -0.767), abs=0.005)
    assert abs(step(1.5)[0]) < 0.01

    # Seven nodes on the covering law's own probability scale, without the
    # stretch, end at its 3.8% and 96.2% points.
    monkeypatch.setattr(quadrature_module, "TAIL_STRETCH", 1.0)
    plain, _ = rule(7)
    assert effects(plain, "lower_pace_shape") == approx([0.231, 0.585], abs=0.002)
    for order, error in ((2, -0.0072), (4, -0.0396)):
        assert plain("baseline", power(order)) / exact(
            "baseline", power(order)
        ) - 1 == (approx(error, abs=0.0005))


def test_the_frontier_a_year_earlier_reads_only_releases_available_then():
    forecast = synthetic_forecast()
    first = min(row.available_at for row in forecast.releases)
    assert observed_frontier(forecast, first - timedelta(seconds=1)) is None
    assert observed_frontier(forecast, first) == max(
        row.value for row in forecast.releases if row.available_at == first
    )
    assert observed_frontier(forecast, forecast.as_of) == max(
        row.value for row in forecast.releases
    )
    leap = datetime(2028, 2, 29, 12, tzinfo=UTC)
    assert one_year_before(leap) == datetime(2027, 2, 28, 12, tzinfo=UTC)


def _synthetic_records(discrepancy_sd, *, labs=80, seed=11):
    rng = np.random.default_rng(seed)
    payload = synthetic_forecast().model_dump(mode="json")
    template = payload["coordinates"][0]
    payload.update(
        coordinates=[{**template, "lab_id": f"lab_{lab}"} for lab in range(labs)],
        trend_window_start="2024-01-01T00:00:00Z",
        trend_discrepancy_sd=None,
        releases=[],
    )
    as_of = datetime(2026, 10, 1, tzinfo=UTC)
    for lab in range(labs):
        for step in range(5):
            elapsed = -2.5 + 0.5 * step
            at = (as_of + timedelta(seconds=elapsed * YEAR_SECONDS)).isoformat()
            # Trend discrepancy plus the declared benchmark error (SD 0.5).
            value = (
                100
                + 3 * lab
                + 15 * elapsed
                + rng.normal(0, discrepancy_sd)
                + rng.normal(0, 0.5)
            )
            payload["releases"].append(
                synthetic_release(f"lab_{lab}", f"r{step}", at, value, sd=0.5)
            )
    return PublicFrontierForecast.model_validate(payload)


def test_pooled_reml_recovers_the_discrepancy_and_brackets_it():
    results = [
        estimate_trend_discrepancy(_synthetic_records(2.0, labs=30, seed=seed))
        for seed in range(20)
    ]
    assert np.mean([estimate for estimate, _ in results]) == pytest.approx(
        2.0, rel=0.05
    )
    # A 90% profile interval: 18 of 20 expected; 15 or more passes.
    assert sum(lower < 2.0 < upper for _, (lower, upper) in results) >= 15
    forecast = _synthetic_records(2.0)
    estimate, _ = estimate_trend_discrepancy(forecast)
    posterior = fit_public_trend(forecast)
    assert posterior.trend_discrepancy_basis == "pooled_reml_estimate"
    assert posterior.trend_discrepancy_sd == pytest.approx(estimate)
    authored = fit_public_trend(forecast.model_copy(update={"trend_discrepancy_sd": 3}))
    assert (authored.trend_discrepancy_basis, authored.trend_discrepancy_sd) == (
        "authored",
        3,
    )
    assert authored.trend_discrepancy_interval is None


def _complex_step_restricted_score(forecast, discrepancy_sd):
    """d/d(log τ) of the restricted negative log-likelihood, by complex step.

    An independent dense implementation, differentiated to roundoff without
    a finite difference.
    """
    records = [
        (lab, row)
        for lab, rows in lab_frontier_records(forecast).items()
        for row in rows
        if row.available_at >= forecast.trend_window_start
    ]
    labs = sorted({lab for lab, _ in records})
    design = np.array(
        [
            [
                *(lab == other for other in labs),
                (row.available_at - forecast.as_of).total_seconds() / YEAR_SECONDS,
            ]
            for lab, row in records
        ]
    )
    values = np.array([row.value for _, row in records])
    benchmark = np.array([row.benchmark_error_sd**2 for _, row in records])
    step = 1e-30
    variance = np.exp(2 * (np.log(discrepancy_sd) + 1j * step)) + benchmark
    inverse = np.diag(1 / variance)
    information = design.T @ inverse @ design
    projection = inverse - inverse @ design @ np.linalg.solve(
        information, design.T @ inverse
    )
    restricted = 0.5 * (
        np.sum(np.log(variance))
        + np.log(np.linalg.det(information))
        + values @ projection @ values
    )
    return float(restricted.imag / step)


def test_reml_discrepancy_is_the_score_root_whatever_the_search_path(monkeypatch):
    # Import refits a frozen forecast and compares the estimate at a relative
    # 1e-8, so the bounded search's stopping point must not leak into it.
    forecast = _synthetic_records(2.0, labs=30, seed=3)
    estimate, interval = estimate_trend_discrepancy(forecast)
    # The independent score changes sign within 1e-9 of the estimate in log τ.
    assert (
        _complex_step_restricted_score(forecast, estimate * np.exp(-1e-9))
        < 0
        < _complex_step_restricted_score(forecast, estimate * np.exp(1e-9))
    )

    monkeypatch.setattr(
        public_frontier_calibration, "DISCREPANCY_SEARCH_RANGE", (0.02, 40.0)
    )
    moved, moved_interval = estimate_trend_discrepancy(forecast)
    assert moved == pytest.approx(estimate, rel=1e-12)
    assert moved_interval == pytest.approx(interval, rel=1e-12)


def test_benchmark_error_alone_gives_an_interval_reaching_zero():
    estimate, (lower, _) = estimate_trend_discrepancy(_synthetic_records(1e-6))
    assert estimate < 0.3
    assert lower == 0


def test_too_few_records_require_an_authored_discrepancy():
    source = synthetic_forecast().model_copy(update={"trend_discrepancy_sd": None})
    with pytest.raises(ValueError, match="author trend_discrepancy_sd"):
        fit_public_trend(source)


def test_comparison_brackets_an_estimated_discrepancy_with_its_interval():
    forecast = _synthetic_records(2.0, labs=10)
    result = compare_public_frontier(
        forecast,
        period_start=datetime(2027, 1, 1, tzinfo=UTC),
        period_end=datetime(2028, 1, 1, tzinfo=UTC),
        n_draws=200,
        seed=1,
    )
    interval = result["trend_discrepancy"]["interval_90"]
    sensitivity = result["trend_discrepancy_sensitivity"]
    assert sensitivity["lower_trend_discrepancy"][
        "trend_discrepancy_sd"
    ] == pytest.approx(interval[0])
    assert sensitivity["upper_trend_discrepancy"][
        "trend_discrepancy_sd"
    ] == pytest.approx(interval[1])


def test_captured_epoch_backtest_reproduces_the_documented_figures(tmp_path):
    """The backtest figures in assumptions.md (OBS-PUBLIC-JOINT) come from here."""
    from catmodel.forecast.public_frontier_backtest import backtest_public_frontier

    forecast = prepare_public_forecast_file(
        source_bundle=_bundle(tmp_path, (DATA / "epoch_eci_scores.csv").read_bytes()),
        template=DATA / "public-frontier-forecast-template.json",
        output=tmp_path / "forecast.json",
    )
    assert forecast.pace_shape == 5
    result = backtest_public_frontier(forecast, n_draws=4000, seed=0)
    assert (len(result.origins), result.scored_cases) == (16, 36)
    assert result.lowest_loss_candidate == "pace_shape_8"
    rows = {row.candidate_id: row for row in result.candidates}
    approx = pytest.approx
    shape3, shape5, shape8 = (
        rows["pace_shape_3"],
        rows["pace_shape_5"],
        rows["pace_shape_8"],
    )
    # Every candidate is truncated at the forecast's cap of 4, which removes
    # 1.5% of shape 3's law, 0.1% of shape 5's and almost none of shape 8's.
    assert result.pace_cap == forecast.pace_cap == 4
    # Truncation moves shape 5's loss by under 0.005 and shape 8's hardly at
    # all; a cap far beyond every draw stands for no cap.
    untruncated = {
        row.candidate_id: row.mean_pinball_loss
        for row in backtest_public_frontier(
            forecast.model_copy(update={"pace_cap": 1e9}), n_draws=4000, seed=0
        ).candidates
    }
    assert 0.004 < untruncated["pace_shape_5"] - shape5.mean_pinball_loss < 0.005
    assert abs(untruncated["pace_shape_8"] - shape8.mean_pinball_loss) < 2e-4
    assert shape8.mean_pinball_loss == approx(0.461, abs=0.002)
    assert (
        shape5.loss_difference_vs_lowest,
        shape5.loss_difference_se,
    ) == approx((0.042, 0.016), abs=0.002)
    assert (
        shape3.loss_difference_vs_lowest,
        shape3.loss_difference_se,
    ) == approx((0.328, 0.043), abs=0.002)
    assert rows["trend_only"].loss_difference_vs_lowest == approx(0.183, abs=0.002)
    assert (
        shape5.mean_upper_tail_pinball_loss,
        shape8.mean_upper_tail_pinball_loss,
    ) == approx((0.325, 0.236), abs=0.002)
    assert shape5.upper_exceedance_rates["p90"] == approx(2 / 36)
    assert shape5.upper_exceedance_rates["p95"] == 0
    assert shape8.upper_exceedance_rates["p95"] == approx(5 / 36)
    # Per horizon: 5 has the lowest loss of all candidates at 3 months but
    # loses to 8 at 6 and 12 months; shape 8's p95 misses are all at 3 and 6
    # months, where its central 90% interval covers 10 of 16 and 11 of 13.
    assert [shape5.mean_pinball_loss_by_horizon[h] for h in ("3", "6", "12")] == (
        approx([0.425, 0.502, 0.684], abs=0.002)
    )
    assert [shape8.mean_pinball_loss_by_horizon[h] for h in ("3", "6", "12")] == (
        approx([0.462, 0.483, 0.420], abs=0.002)
    )
    assert min(rows, key=lambda row: rows[row].mean_pinball_loss_by_horizon["3"]) == (
        "pace_shape_5"
    )
    assert shape8.upper_exceedance_rates_by_horizon["3"]["p95"] == approx(4 / 16)
    assert shape8.upper_exceedance_rates_by_horizon["6"]["p95"] == approx(1 / 13)
    assert shape8.upper_exceedance_rates_by_horizon["12"]["p95"] == 0
    assert shape8.central_90_coverage_by_horizon["3"] == approx(10 / 16)
    assert shape5.central_90_coverage_by_horizon["3"] == approx(14 / 16)
    assert shape5.central_90_coverage_by_horizon["12"] == 1
