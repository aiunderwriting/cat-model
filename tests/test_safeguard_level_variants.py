"""Safeguard-level variants: separately collected studies of authored safeguard postures."""

import json
import re
from typing import cast

import pytest
from click.testing import CliRunner

import catmodel.setup.public_frontier as frontier
from catmodel.cli import cli
from catmodel.collection.contextual_collection import run_study_collection
from catmodel.config import load_model_profile
from catmodel.contracts.contextual_risk import (
    SAFEGUARD_LEVEL_FIELDS,
    ContextualRiskInstrument,
)
from catmodel.contracts.public_frontier import (
    NO_PROGRESS_NODE_ID,
    PublicFrontierQuadrature,
    quadrature_node_ids,
)
from catmodel.llm.transport import ProviderResponse
from catmodel.resources import config_path, data_path
from catmodel.results.execution import load_run_capture, run_setup
from catmodel.setup.contextual_inputs import build_contextual_instrument
from catmodel.setup.public_frontier_comparisons import (
    CONTEXT_VARIANT_FAMILIES,
    SAFEGUARD_CONTROL,
)
from catmodel.setup.public_study import (
    import_public_frontier_study,
    prepare_public_forecast_file,
    public_study_templates,
    write_public_frontier_study,
)
from catmodel.setup.sensitivity_catalog import SAFEGUARD_LEVELS
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
)
from test_support.results import reduced_arm
from tests.test_exposure_scale_variants import QUANTILES, USERS, _template
from tests.test_exposure_scale_variants import (
    _variant_study as _exposure_variant_study,
)
from tests.test_hypothetical_frontier_lab import (
    _epoch_bundle,
    _run_prompt,
    _shared_context,
)
from tests.test_public_frontier_study import (
    _collect,
    _study,
)

STUDY = config_path("studies", "hypothetical-frontier-lab")
SAFEGUARD = CONTEXT_VARIANT_FAMILIES["safeguard_level"]
# Synthetic postures: the fixture provider reads its marker word.
POSTURES = {
    "weaker": ("permissive", 2.0),
    "stronger": ("restrictive", 0.5),
}
BLIND_WORDS = (
    "weaker",
    "stronger",
    "safeguard_level",
    "level",
    "levels",
    "variants",
    "headline",
    "baseline",
    "variant",
    "sensitivity",
    "arm",
)


def _get(record, path):
    for name in path.split("."):
        record = record[name]
    return record


def _put(record, path, value):
    *parents, name = path.split(".")
    for parent in parents:
        record = record[parent]
    record[name] = value


class _PostureProvider:
    """A synthetic annual rate: users / 100, times a posture's factor."""

    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    async def complete(self, *, user_prompt, **_kwargs):
        found = USERS.search(user_prompt)
        assert found is not None
        rate = float(found.group(1).replace(",", "")) / 100
        for marker, factor in POSTURES.values():
            if marker in user_prompt:
                rate *= factor
        answer = contextual_answer()
        assert answer.rate is not None
        answer = answer.model_copy(
            update={
                "rate": answer.rate.model_copy(update=dict.fromkeys(QUANTILES, rate))
            }
        )
        return ProviderResponse(
            text=fixture_response(user_prompt, answer),
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=5,
            output_tokens=10,
        )


def _posture(level, status):
    """Synthetic safeguard texts of one level for one source status."""
    marker = POSTURES[level][0]
    texts = {
        "fixed_controls": f"Synthetic {marker} controls.",
        "deployment_binding.tools_and_permissions": f"Synthetic {marker} tools.",
        "deployment_binding.human_oversight": f"Synthetic {marker} oversight.",
        "deployment.definition": f"Synthetic {marker} internal use",
        "deployment_binding.use_population": f"About {{users}} synthetic users, {marker} scope",
    }
    return {path: texts[path] for path in SAFEGUARD_LEVEL_FIELDS[status]}


def _variant_study(
    tmp_path, bundle, level, *, name=None, edits=None, no_progress_edits=None
):
    """A safeguard variant on the headline's forecast, at one sample, on the
    embedded rule's interior nodes and the no-progress node; ``edits`` replace
    other instrument fields, and ``no_progress_edits`` those of the
    no-progress node's planned instruments only."""
    name = name or level
    templates = {}
    for status in ("public", "not_yet_public"):
        payload = json.loads((tmp_path / f"{status}.json").read_text())
        for path, text in _posture(level, status).items():
            _put(payload, path, text)
        payload["safeguard_level"] = level
        payload["respondents"] = payload["respondents"][:1]
        payload.update(edits or {})
        templates[status] = tmp_path / f"{name}-{status}.json"
        templates[status].write_text(
            ContextualRiskInstrument.model_validate(payload).model_dump_json()
        )
    folder = tmp_path / f"safeguard-{name}"
    write_public_frontier_study(
        forecast_path=tmp_path / "forecast-input.json",
        public_instrument=templates["public"],
        not_yet_public_instrument=templates["not_yet_public"],
        assumptions_path=tmp_path / "assumptions.json",
        output=folder,
        nodes=3,
        seed=42,
    )
    plan = json.loads((folder / "public-study.json").read_text())
    for row in plan["contexts"]:
        if no_progress_edits and row["draw_id"] == "node-00":
            path = folder / row["instrument"]
            payload = json.loads(path.read_text())
            payload.update(no_progress_edits)
            path.write_text(
                ContextualRiskInstrument.model_validate(payload).model_dump_json()
            )
    run_study_collection(
        study=folder / "public-study.json",
        source_bundle=bundle,
        profile="test",
    )
    return folder / "public-study.json"


def _headline(tmp_path, monkeypatch):
    folder, bundle, _ = _study(
        tmp_path,
        monkeypatch,
        nodes=7,
        template=_template(2),
        providers=[_PostureProvider()],
    )
    _collect(folder, bundle)
    return folder / "public-study.json", bundle


def _importer(tmp_path, headline):
    def imported(name, studies, **options):
        return import_public_frontier_study(
            source=headline,
            destination=tmp_path / name,
            setup_id="safeguards",
            years_per_state=8,
            max_reference_state_years=22 * 4096,
            max_comparison_state_years=22 * 1024,
            seed=5,
            variant_studies=studies,
            code_commit="fixture",
            **options,
        )

    return imported


def test_safeguard_variants_run_against_the_embedded_rule_control(
    tmp_path, monkeypatch
):
    headline, bundle = _headline(tmp_path, monkeypatch)
    weaker = _variant_study(tmp_path, bundle, "weaker")
    stronger = _variant_study(tmp_path, bundle, "stronger")
    imported = _importer(tmp_path, headline)

    package = imported("package", (weaker, stronger))
    arms = {row.alternative_id: row for row in package.alternatives}
    variants = ("safeguard_level::weaker", "safeguard_level::stronger")
    assert {SAFEGUARD_CONTROL, *variants} <= set(package.execution.required_arms)
    assert {arms[arm].matched_control_id for arm in variants} == {SAFEGUARD_CONTROL}
    assert arms["safeguard_level::weaker"].selection_provenance == {
        "family_id": "safeguard_level",
        "base_support_only": False,
        "safeguard_level": "weaker",
    }
    assert {
        row.status
        for row in package.execution.comparison_requirements
        if row.family_id == "safeguard_level"
    } == {"required"}
    # Each variant's three embedded-rule nodes and its no-progress node, by one
    # sample, join at zero reference weight; the control is the headline
    # reweighted to the embedded rule, like exposure's.
    quadrature = PublicFrontierQuadrature.model_validate_json(
        (headline.parent / "public-frontier.json").read_text()
    )
    embedded = {
        node.draw_id: weight
        for node, weight in zip(
            quadrature.nodes, quadrature.base.embedded_weights, strict=True
        )
    }
    weights = {state.state_id: state.weight for state in package.joint_states}
    streams = {state.state_id: state.rank_stream_id for state in package.joint_states}
    for level in SAFEGUARD_LEVELS:
        prefix = SAFEGUARD.state_id(level, "")
        own = {key for key in weights if key.startswith(prefix)}
        assert len(own) == 4 and {weights[key] for key in own} == {0.0}
        no_progress = {key for key in own if key.startswith(f"{prefix}node-00:")}
        assert len(no_progress) == 1
        # The arm averages the embedded-rule nodes only, at the embedded rule's
        # weights: the no-progress node's states carry none.
        variant_weights = arms[f"safeguard_level::{level}"].state_weights
        assert variant_weights is not None
        assert {key for key, value in variant_weights.items() if value} == (
            own - no_progress
        )
        by_node: dict[str, float] = {}
        for key in own:
            node = key.removeprefix(prefix).split(":", 1)[0]
            by_node[node] = by_node.get(node, 0.0) + variant_weights[key]
        assert by_node == pytest.approx(
            {node: embedded[node] for node in by_node}, abs=1e-12
        )
        assert by_node["node-00"] == 0.0
        # Each variant state reads the parameter ranks of the headline state
        # at its node and respondent selection, so the arm stays paired with
        # the control, whose states keep their own streams.
        assert {key: streams[key] for key in own} == {
            key: key.removeprefix(prefix) for key in own
        }
        assert all(streams[key.removeprefix(prefix)] is None for key in own)
    control = arms[SAFEGUARD_CONTROL].state_weights
    assert control is not None
    for key, value in control.items():
        assert value == pytest.approx(embedded.get(key.split(":", 1)[0], 0.0) / 2)
    assert any("Safeguard-level variants" in row for row in package.limitations)
    assert any(
        row.evidence_id.startswith("safeguard_level_weaker_")
        and "not shown to respondents" in row.conditioning
        for row in package.evidence.records
    )

    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    ranked = {
        row["arm"]: row for row in report["sensitivity_summary"]["tornado"]["rows"]
    }
    assert SAFEGUARD_CONTROL not in ranked
    assert {ranked[arm]["control_arm"] for arm in variants} == {SAFEGUARD_CONTROL}
    # The synthetic rate doubles under the weaker posture and halves under the
    # stronger one, at both source statuses.
    for level, (_, factor) in POSTURES.items():
        assert ranked[f"safeguard_level::{level}"]["relative_change"] == pytest.approx(
            factor - 1, rel=1e-9
        )
    comparisons = cast(dict, report["comparisons"])
    for arm in variants:
        (lab,) = (view for view in comparisons[arm] if view.startswith("lab_loss::"))
        # EAL by view and source status, against the matched control; the
        # comparison with the reference arm is reported separately.
        assert report["comparison_reference_arms"][arm] == SAFEGUARD_CONTROL
        assert arm in report["reference_comparisons"]
        for view in (
            "welfare",
            lab,
            "source_status::public::welfare",
            "source_status::not_yet_public::welfare",
        ):
            effect = comparisons[arm][view]["expected_annual_loss"]
            assert effect["absolute_difference"] is not None, (arm, view)
    family = next(
        row
        for row in report["coverage"]["sensitivity_families"]
        if row["family_id"] == "safeguard_level"
    )
    assert family["status"] == "complete"

    without = imported("without", ())
    # The family is unavailable without variant studies, and adding variants leaves
    # every headline state, every other arm and every other family unchanged.
    assert {
        row.reason
        for row in without.execution.comparison_requirements
        if row.family_id == "safeguard_level"
    } == {
        "No safeguard-level variant studies were imported; collect each variant as a public study on the embedded rule's nodes and the no-progress node and import it with --variant-study."
    }
    headline_states = {state.state_id: state for state in without.joint_states}
    assert {
        state.state_id: state
        for state in package.joint_states
        if state.state_id in headline_states
    } == headline_states
    assert len(package.joint_states) == len(headline_states) + 8
    for arm in without.alternatives:
        other = arms[arm.alternative_id]
        assert package.overrides.of(other.alternative_id) == (
            without.overrides.of(arm.alternative_id)
        )
        if arm.state_weights is not None:
            assert other.state_weights is not None
            assert {
                key: value
                for key, value in other.state_weights.items()
                if key in headline_states
            } == arm.state_weights
    assert [
        row
        for row in package.execution.comparison_requirements
        if row.family_id != "safeguard_level"
    ] == [
        row
        for row in without.execution.comparison_requirements
        if row.family_id != "safeguard_level"
    ]

    with pytest.raises(ValueError, match="require the safeguard_level comparison"):
        imported("unselected", (weaker,), comparison_families=("tail_shape",))
    with pytest.raises(ValueError, match="imported twice"):
        imported("twice", (weaker, weaker))
    # A study whose instruments carry no variant, here the headline itself,
    # has no family to be imported under.
    with pytest.raises(ValueError, match="is not a variant study"):
        imported("no-variant", (headline,))
    # One list takes both families: each study enters the family of the
    # variant its instruments carry, and the order given does not reach the
    # setup.
    half = _exposure_variant_study(tmp_path, bundle, "half", 0.5)
    families = ("exposure_scale", "safeguard_level")
    mixed = imported("mixed", (weaker, half), comparison_families=families)
    assert {"exposure_scale::half", "safeguard_level::weaker"} <= set(
        mixed.execution.required_arms
    )
    assert (
        imported(
            "mixed-reordered", (half, weaker), comparison_families=families
        ).manifest_sha256
        == mixed.manifest_sha256
    )
    other = _variant_study(
        tmp_path,
        bundle,
        "weaker",
        name="other-adoption",
        edits={"fixed_adoption": "Synthetic adoption that differs."},
    )
    with pytest.raises(ValueError, match="only in its safeguard texts"):
        imported("other-adoption", (other,))
    with pytest.raises(ValueError, match="cannot carry a safeguard level"):
        import_public_frontier_study(
            source=weaker,
            destination=tmp_path / "variant-as-headline",
            setup_id="safeguards",
            code_commit="fixture",
        )


def test_one_safeguard_level_imports_without_the_other(tmp_path, monkeypatch):
    """A setup can hold one level's variant study: its arm is compared with the
    control, and the level that was not collected is recorded unavailable."""
    headline, bundle = _headline(tmp_path, monkeypatch)
    stronger = _variant_study(tmp_path, bundle, "stronger")

    package = _importer(tmp_path, headline)("package", (stronger,))

    arms = {row.alternative_id: row for row in package.alternatives}
    assert "safeguard_level::stronger" in arms
    assert "safeguard_level::weaker" not in arms
    assert arms["safeguard_level::stronger"].matched_control_id == SAFEGUARD_CONTROL
    assert {SAFEGUARD_CONTROL, "safeguard_level::stronger"} <= set(
        package.execution.required_arms
    )
    rows = [
        row
        for row in package.execution.comparison_requirements
        if row.family_id == "safeguard_level"
    ]
    assert {row.status for row in rows} == {"required", "unavailable"}
    assert [row.reason for row in rows if row.status == "unavailable"] == [
        "weaker: no variant study was imported."
    ]
    assert not any(
        key.startswith(SAFEGUARD.state_id("weaker", ""))
        for key in (state.state_id for state in package.joint_states)
    )


def test_variant_no_progress_states_are_simulated_without_moving_any_arm(
    tmp_path, monkeypatch
):
    """Each variant arm simulates its no-progress states at weight zero, so they
    have draws under the variant's own law, and every arm's EAL in every view is
    what it is when they are not simulated."""
    headline, bundle = _headline(tmp_path, monkeypatch)
    variants = tuple(
        _variant_study(tmp_path, bundle, level) for level in SAFEGUARD_LEVELS
    )
    imported = _importer(tmp_path, headline)

    def run(name):
        package = imported(name, variants)
        report = run_setup(
            setup_path=package.root,
            output=tmp_path / f"{name}-run",
            code_commit="fixture",
        )
        return (
            package,
            report,
            load_run_capture(tmp_path / f"{name}-run" / "capture")[1],
        )

    package, report, capture = run("simulated")
    for level in SAFEGUARD_LEVELS:
        arm = f"safeguard_level::{level}"
        prefix = SAFEGUARD.state_id(level, "")
        zero = {
            state.state_id
            for state in package.joint_states
            if state.state_id.startswith(f"{prefix}node-00:")
        }
        assert len(zero) == 1
        assert zero <= set(package.execution.required_state_ids_by_arm[arm])
        simulated = {capture.states[key].state_id for key in capture.arms[arm]}
        assert zero <= simulated
        weights = package.execution.required_state_ids_by_arm
        assert not zero & set(weights["reference"])
    with monkeypatch.context() as patch:
        patch.setattr(frontier, "variant_no_progress_requirements", lambda _cells: {})
        _, without, bare = run("unsimulated")
    assert not any(
        capture.states[key].state_id.startswith(
            f"{SAFEGUARD.state_id(level, '')}node-00:"
        )
        for level in SAFEGUARD_LEVELS
        for key in bare.arms[f"safeguard_level::{level}"]
    )

    def eals(result):
        return {
            arm: {
                view: metric["expected_annual_loss"]
                for view, metric in reduced_arm(row)["metrics"].items()
                if metric["status"] == "available"
            }
            for arm, row in result["arms"].items()
        }

    assert eals(report) == eals(without)
    assert report["tail_gate"]["by_arm"].keys() == without["tail_gate"]["by_arm"].keys()
    for arm, row in report["tail_gate"]["by_arm"].items():
        assert (
            row["years_per_state"]
            == without["tail_gate"]["by_arm"][arm]["years_per_state"]
        ), arm


def test_safeguard_variant_intake_requires_the_headline_no_progress_node(
    tmp_path, monkeypatch
):
    headline, bundle = _headline(tmp_path, monkeypatch)
    imported = _importer(tmp_path, headline)
    # A no-progress node whose stipulation departs from the headline's is
    # refused, here an adoption text changed at that node only.
    departed = _variant_study(
        tmp_path,
        bundle,
        "weaker",
        name="departed",
        no_progress_edits={"fixed_adoption": "Synthetic adoption that differs."},
    )
    with pytest.raises(
        ValueError, match=r"cannot change across public paths|only in its safeguard"
    ):
        imported("departed", (departed,))
    # So is one weighted as anything but zero.
    import catmodel.setup.public_study as service

    weaker = _variant_study(tmp_path, bundle, "weaker")
    base = service._base_weights

    def weighted(quadrature):
        weights = base(quadrature)
        if quadrature.interior_nodes == 3:  # the variant's rule, not the headline's
            moved = weights["node-1-2"] / 2
            weights["node-1-2"] -= moved
            weights["node-00"] += moved
        return weights

    with monkeypatch.context() as patch:
        patch.setattr(service, "_base_weights", weighted)
        with pytest.raises(ValueError, match="no-progress node at weight zero"):
            imported("weighted", (weaker,))
    # And a variant that did not collect it.
    folder = weaker.parent
    for status in ("public", "not_yet_public"):
        (folder / "collections" / f"node-00-{status}" / "contextual-risk.json").unlink()
    with pytest.raises(ValueError, match="must collect its no-progress node node-00"):
        imported("uncollected", (weaker,))


def _cli(*arguments):
    result = CliRunner().invoke(cli, [str(argument) for argument in arguments])
    assert result.exit_code == 0, result.output
    return result.output


def test_committed_safeguard_levels_vary_only_their_stipulated_texts(tmp_path):
    bundle = _epoch_bundle(tmp_path / "source")
    context = json.loads((STUDY / "context.json").read_text())
    forecast = prepare_public_forecast_file(
        source_bundle=bundle,
        template=data_path("capability", "public-frontier-forecast-template.json"),
        output=tmp_path / "forecast.json",
    )
    templates = {
        (level, status): template.model_dump(mode="json")
        for level in (None, *SAFEGUARD_LEVELS)
        for status, template in public_study_templates(
            context,
            forecast=forecast,
            source_bundle=bundle,
            respondents=contextual_instrument().respondents,
            scenario_ids=("all",),
            variant=level,
        ).items()
    }
    for level in SAFEGUARD_LEVELS:
        authored = context["safeguard_levels"][level]
        for status in ("public", "not_yet_public"):
            base = templates[None, status]
            variant = json.loads(json.dumps(templates[level, status]))
            assert "safeguard_level" not in base
            assert variant.pop("safeguard_level") == level
            for path in SAFEGUARD_LEVEL_FIELDS[status]:
                record, _, field = path.rpartition(".")
                text = (
                    authored["fixed_controls"]
                    if not record
                    else authored["source_statuses"][status][record][field]
                )
                assert _get(variant, path) == text != _get(base, path), path
                # Respondents react to content, not length: each text is within
                # 10% of the headline's word count.
                assert abs(len(text.split()) - len(_get(base, path).split())) <= (
                    0.1 * len(_get(base, path).split())
                ), (level, path)
                _put(variant, path, _get(base, path))
            # Every other byte of the template is the headline's.
            assert json.dumps(variant, indent=2) == json.dumps(base, indent=2)

    # The question at one trajectory node is the headline's question with the
    # level's texts in place of the headline's, and names no level or variant.
    # The headline study has seven interior nodes and each level's study the
    # embedded three-node rule's, each with the no-progress node. The headline
    # asks each model of the profile its samples_per_question and a level's
    # study three samples.
    profile = load_model_profile("test")
    assert profile.settings.samples_per_question != 3
    studies = {}
    for level in (None, *SAFEGUARD_LEVELS):
        name = level or "headline"
        _cli(
            "--profile",
            "test",
            "capability-forecast",
            "public-study",
            "--source-bundle",
            bundle,
            "--forecast",
            data_path("capability", "public-frontier-forecast-template.json"),
            "--context",
            STUDY / "context.json",
            "--scenario",
            "all",
            *(("--variant", level) if level else ()),
            "--assumptions",
            STUDY / "public-assembly-assumptions.json",
            "--output",
            tmp_path / name,
        )
        studies[level] = json.loads((tmp_path / name / "public-study.json").read_text())
        draws = {row["draw_id"] for row in studies[level]["contexts"]}
        assert draws == {
            NO_PROGRESS_NODE_ID,
            *quadrature_node_ids(3 if level else 7),
        }
        samples = 3 if level else profile.settings.samples_per_question
        for row in studies[level]["contexts"]:
            planned = json.loads((tmp_path / name / row["instrument"]).read_bytes())
            assert planned.get("safeguard_level") == level
            assert [
                (respondent["model_id"], respondent["sample_index"])
                for respondent in planned["respondents"]
            ] == [
                (spec.model_id, sample)
                for spec in profile.models
                for sample in range(samples)
            ]

    def question(level, status):
        row = next(
            row
            for row in studies[level]["contexts"]
            if row["draw_id"] == "node-1-2" and row["source_model_status"] == status
        )
        instrument = ContextualRiskInstrument.model_validate_json(
            (tmp_path / (level or "headline") / row["instrument"]).read_text()
        )
        return _run_prompt(instrument), instrument.model_dump(mode="json")

    for status in ("public", "not_yet_public"):
        base_prompt, base = question(None, status)
        for level in SAFEGUARD_LEVELS:
            variant_prompt, variant = question(level, status)
            restored = variant_prompt
            for path in SAFEGUARD_LEVEL_FIELDS[status]:
                encoded = json.dumps(_get(variant, path))[1:-1]
                assert encoded in restored, (level, path)
                restored = restored.replace(encoded, json.dumps(_get(base, path))[1:-1])
            assert restored == base_prompt, (level, status)
            # The level's own texts name no level, variant or comparison.
            for path in SAFEGUARD_LEVEL_FIELDS[status]:
                for word in BLIND_WORDS:
                    pattern = rf"(?<![\w.-]){re.escape(word)}(?!\w)"
                    assert not re.search(pattern, _get(variant, path), re.IGNORECASE), (
                        level,
                        path,
                        word,
                    )


REVIEW_PASSAGES = (
    "deployment_binding.tools_and_permissions",
    "deployment_binding.human_oversight",
)
REVIEW_STATEMENTS = {
    ("public", "deployment_binding.tools_and_permissions"): (
        "with user confirmation before purchases and other consequential actions",
    ),
    ("public", "deployment_binding.human_oversight"): (
        "Consumer users review outputs and confirm consequential agent actions.",
        "Enterprise customers set their own oversight, from human review of every "
        "output to autonomous pipelines.",
    ),
    ("not_yet_public", "deployment_binding.human_oversight"): (
        "Staff review agent work at the granularity each team sets;",
    ),
}
REVIEW_WORDS = re.compile(r"\b(?:review|approv|confirm)\w*", re.IGNORECASE)


def test_no_safeguard_level_changes_who_reviews_an_output_or_action():
    """The taxonomy assigns an incident to a scenario by whether a person
    considered the output or action, so a level that required or removed that
    review would move incidents between scenarios (``POP-SAFEGUARD-LEVEL``).
    Each level keeps the stipulated statements of review word for word and
    says nothing else about review, approval or confirmation."""
    context = json.loads((STUDY / "context.json").read_text())
    for status in ("public", "not_yet_public"):
        for path in REVIEW_PASSAGES:
            statements = REVIEW_STATEMENTS.get((status, path), ())
            stipulated = _get(context["source_statuses"][status], path)
            texts = {
                level: _get(
                    context["safeguard_levels"][level]["source_statuses"][status], path
                )
                for level in SAFEGUARD_LEVELS
            }
            for name, text in ((None, stipulated), *texts.items()):
                for statement in statements:
                    assert statement in text, (name, status, path)
                    text = text.replace(statement, "")
                # The listed statements are every one the stipulated text
                # makes, and a level adds none.
                assert not REVIEW_WORDS.search(text), (name, status, path)


def test_build_instrument_applies_one_authored_safeguard_level(tmp_path):
    bundle = write_source_bundle(tmp_path / "source")
    context = _shared_context()
    levels = {}
    for level in SAFEGUARD_LEVELS:
        statuses = {}
        for status in ("public", "not_yet_public"):
            fields = {}
            for path, text in _posture(level, status).items():
                if path == "fixed_controls":
                    continue
                record, _, name = path.partition(".")
                if name == "use_population":
                    text = (
                        context["source_statuses"][status]["deployment_binding"][
                            "use_population"
                        ]
                        + f" ({level} scope)"
                    )
                fields.setdefault(record, {})[name] = text
            statuses[status] = fields
        levels[level] = {
            "rationale": f"Synthetic {level} rationale.",
            "fixed_controls": f"Synthetic {level} controls.",
            "source_statuses": statuses,
        }
    authored = {**context, "safeguard_levels": levels}

    def build(payload, status="not_yet_public", **variants):
        return build_contextual_instrument(
            payload,
            source_bundle=bundle,
            respondents=contextual_instrument().respondents,
            scenario_ids=("output_reliance_legal",),
            source_status=status,
            **variants,
        )

    # The authored levels never reach the study's own templates.
    assert build(authored) == build(context)
    variant = build(authored, safeguard_level="stronger")
    assert variant.safeguard_level == "stronger"
    assert variant.fixed_controls == "Synthetic stronger controls."
    assert variant.deployment.definition == "Synthetic restrictive internal use"
    assert variant.deployment_binding.use_population.endswith("(stronger scope)")
    assert (
        build(authored, "public", safeguard_level="weaker").deployment
        == build(context, "public").deployment
    )
    with pytest.raises(ValueError, match="safeguard level must be one of"):
        build(authored, safeguard_level="strongest")
    with pytest.raises(ValueError, match="requires a context with safeguard_levels"):
        build(context, safeguard_level="weaker")
    with pytest.raises(ValueError, match="at most one context variant"):
        build(authored, safeguard_level="weaker", exposure_variant="half")
    missing = json.loads(json.dumps(authored))
    del missing["safeguard_levels"]["weaker"]["source_statuses"]["public"][
        "deployment_binding"
    ]["human_oversight"]
    with pytest.raises(ValueError, match="must replace exactly"):
        build(missing)
    rescaled = json.loads(json.dumps(authored))
    rescaled["safeguard_levels"]["stronger"]["source_statuses"]["not_yet_public"][
        "deployment_binding"
    ]["use_population"] = "About {lab_staff} synthetic staff"
    with pytest.raises(ValueError, match="keeps the study's scale placeholders"):
        build(rescaled, safeguard_level="stronger")
