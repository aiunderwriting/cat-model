"""The conditioning pilot (scripts/conditioning_pilot.py): its arms change only
what they say they change, and its analysis recovers a known response."""

import csv
import math
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from catmodel.config import load_model_profile
from catmodel.contracts.contextual_risk import (
    ContextualRiskInstrument,
    UnansweredCase,
)
from catmodel.elicitation.contextual_frequency import ContextualChainPrompt
from catmodel.elicitation.contextual_risk import ContextualPrompt
from catmodel.evidence import source_inputs
from catmodel.evidence.source_inputs import build_source_bundle
from catmodel.json_io import strict_json_loads
from catmodel.llm.transport import ProviderResponse
from catmodel.resources import config_path, scenarios_path
from catmodel.results.respondent_sampling import NOISE_Z
from catmodel.scenarios.loader import load_scenarios
from catmodel.worktree import CodeProvenance
from scripts.conditioning_pilot import (
    ARMS,
    FIELDS,
    QUANTITIES,
    RESOLVED_CHANGE,
    RULES,
    Estimate,
    analyse_pilot,
    combine,
    fixed_scale_instrument,
    main,
    never_falls,
    noise_floor,
    paired_weights,
    placebo_instrument,
    read_pilot,
    slope_weights,
    verdict,
    write_analysis,
)
from scripts.run_hypothetical_lab_study import stage_authored_source
from test_support.collection_trace import request_kind
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_artifact,
    fixture_response,
    scenario_node_answer,
)

STUDY = config_path("studies", "hypothetical-frontier-lab")
SCENARIOS = ("compromise_critical_infrastructure", "output_reliance_legal")
STATUSES = ("public", "not_yet_public")
NODES = ("node-00", "node-1-4", "node-1-2", "node-3-4")
"""The three-node rule's nodes in order of frontier progress."""
PROFILE = "test"
NOISES = (-0.05, 0.05)
"""The log noise of one question's successive calls, one per sample: every
group's mean log answer is its expected log exactly."""
SAMPLES = len(NOISES)


@pytest.fixture(scope="module")
def planned(tmp_path_factory) -> Path:
    """A pilot planned on two scenarios and the three-node rule from the
    committed hypothetical-lab inputs, and never collected."""
    root = tmp_path_factory.mktemp("conditioning-pilot")
    bundle = root / "source-bundle"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            source_inputs,
            "resolve_code_provenance",
            lambda _: CodeProvenance("a" * 40, True, "clean", None),
        )
        build_source_bundle(stage_authored_source(root / "authored-source"), bundle)
    arguments = ["--source-bundle", str(bundle), "--profile", PROFILE]
    arguments += ["--context", str(STUDY / "context.json"), "--samples", str(SAMPLES)]
    for scenario in SCENARIOS:
        arguments += ["--scenario", scenario]
    assert main(["plan", str(root / "pilot"), *arguments]) == 0
    return root / "pilot"


def _instruments(pilot: Path) -> dict[tuple[str, str, str], ContextualRiskInstrument]:
    """The pilot's instruments by (arm, trajectory node, source status)."""
    return {
        (row["arm"], row["draw_id"], row["source_model_status"]): (
            ContextualRiskInstrument.model_validate(
                strict_json_loads((pilot / row["instrument"]).read_bytes())
            )
        )
        for row in read_pilot(pilot)["contexts"]
    }


def _questions(instrument: ContextualRiskInstrument) -> list[list[str]]:
    """The lines of every scenario's direct and chain question."""
    respondent, score = instrument.respondents[0], instrument.anchor_scores[0]
    return [
        prompt(instrument, node, respondent, score).render().split("\n")
        for node in instrument.collectible_nodes
        for prompt in (ContextualPrompt, ContextualChainPrompt)
    ]


def _differing(left: ContextualRiskInstrument, right: ContextualRiskInstrument):
    """Every pair of differing lines between two instruments' questions."""
    pairs = []
    for ours, theirs in zip(_questions(left), _questions(right), strict=True):
        assert len(ours) == len(theirs)
        pairs += [(a, b) for a, b in zip(ours, theirs, strict=True) if a != b]
    return pairs


_TABLE_LINE = re.compile(r"(date|\d{4}-\d{2}-\d{2})  ")


def test_the_plan_lists_three_arms_on_the_three_node_rule(planned):
    pilot = read_pilot(planned)
    assert pilot["arms"] == list(ARMS) and pilot["skipped_arms"] == {}
    assert pilot["scenarios"] == list(SCENARIOS)
    assert sorted(pilot["nodes"], key=pilot["nodes"].get) == list(NODES)
    assert pilot["nodes"]["node-00"] == 0
    assert {
        (row["arm"], row["draw_id"], row["source_model_status"])
        for row in pilot["contexts"]
    } == {(arm, node, status) for arm in ARMS for node in NODES for status in STATUSES}
    assert len({row["capture"] for row in pilot["contexts"]}) == len(pilot["contexts"])
    respondents = SAMPLES * len(load_model_profile(profile=PROFILE).models)
    assert pilot["samples_per_model"] * len(pilot["models"]) == respondents
    # A direct and a chain question per scenario, respondent and collection.
    assert pilot["initial_calls"] == (
        2 * len(SCENARIOS) * respondents * len(pilot["contexts"])
    )
    # The reference arm asks the planned study's own instruments.
    assert all(
        row["instrument"].startswith("study/instruments/")
        for row in pilot["contexts"]
        if row["arm"] == "reference"
    )
    # The plan is never assembled, and says so where a study gives rationales.
    study = strict_json_loads((planned / "study" / "public-study.json").read_bytes())
    assert study["exposure_rationale"].startswith("Not assembled")


def test_the_default_pilot_s_cost_and_resolution(planned):
    """The default rule at the example profile asks 720 initial questions per
    scenario for the three arms, 4,320 for six scenarios, and its fitted
    change over the range has a standard error of about 0.35 noise floors."""
    plan = read_pilot(planned)
    example = load_model_profile(profile="example")
    respondents = len(example.models) * example.settings.samples_per_question
    asked = plan["initial_calls"] / (len(SCENARIOS) * SAMPLES * len(plan["models"]))
    assert asked * respondents == 720 and asked * respondents * 6 == 4320
    # With one noise in every group: the slope's standard error times the
    # range of progress, and the placebo's over every node and respondent.
    x = np.array(sorted(plan["nodes"].values())) / 10
    fitted = (x[-1] - x[0]) / math.sqrt(respondents * ((x - x.mean()) ** 2).sum())
    assert fitted == pytest.approx(0.35, abs=0.005)
    assert math.sqrt(2 / (respondents * len(x))) == pytest.approx(0.18, abs=0.005)
    assert NOISE_Z * fitted == pytest.approx(0.7, abs=0.01)
    assert math.log(RESOLVED_CHANGE) / NOISE_Z / fitted == pytest.approx(0.3, abs=0.03)


def test_the_fixed_scale_arm_states_one_scale_and_keeps_every_path(planned):
    instruments = _instruments(planned)
    for status in STATUSES:
        fixed = [instruments["fixed_scale", node, status] for node in NODES]
        reference = [instruments["reference", node, status] for node in NODES]
        assert len({row.deployment_binding.use_population for row in fixed}) == 1
        assert len({row.deployment_binding.use_population for row in reference}) == 4
        # Every scale is at its study-year level on the median path.
        exposure = fixed[0].path_linked_exposure
        assert exposure is not None
        assert {tuple(row.scale_levels().items()) for row in fixed} == {
            tuple(
                (scale.scale_id, float(f"{scale.study_year_median:.2g}"))
                for scale in exposure.scales
            )
        }
        for ours, theirs in zip(fixed, reference, strict=True):
            assert ours.public_trajectory == theirs.public_trajectory
            a, b = ours.model_dump(mode="json"), theirs.model_dump(mode="json")
            assert {key for key in a if a[key] != b[key]} == {
                "context_id",
                "path_linked_exposure",
                "deployment_binding",
            }
            # Against the reference question only stated numbers differ: the
            # use line and a chain's base count derived from a scale.
            changed = _differing(theirs, ours)
            assert changed
            for old, new in changed:
                assert old.split(": ")[0] in (
                    "Use",
                    "Exposure base",
                    "opportunity_rate",
                )
                assert re.sub(r"[\d.,]+", "#", old) == re.sub(r"[\d.,]+", "#", new)
        # Between nodes a fixed-scale question differs only in its path: the
        # table's dated rows and the paragraph above them, which names a
        # change of leader and the doublings the frontier's rise would be.
        changed = _differing(fixed[0], fixed[-1])
        assert changed
        assert all(
            _TABLE_LINE.match(old)
            or "The lead changes" in old + new
            or "such doublings" in old + new
            for old, new in changed
        )
        # The reference arm also changes its stated use between nodes.
        assert any(
            old.startswith("Use: ")
            for old, _ in _differing(reference[0], reference[-1])
        )


def test_the_placebo_arm_only_reverses_the_table_columns(planned):
    instruments = _instruments(planned)
    for node in NODES:
        for status in STATUSES:
            placebo = instruments["placebo", node, status]
            reference = instruments["reference", node, status]
            a, b = placebo.model_dump(mode="json"), reference.model_dump(mode="json")
            assert {key for key in a if a[key] != b[key]} == {
                "context_id",
                "public_trajectory",
            }
            ours, theirs = placebo.public_trajectory, reference.public_trajectory
            assert ours is not None and theirs is not None
            assert ours.target_path() == theirs.target_path()
            assert ours.trajectory_sha256 == theirs.trajectory_sha256
            assert [row.lab_id for row in ours.coordinates] == [
                row.lab_id for row in theirs.coordinates
            ][::-1]
            # Every line of the table, its header included, and no other line.
            changed = _differing(reference, placebo)
            table = [
                line
                for lines in _questions(reference)
                for line in lines
                if _TABLE_LINE.match(line)
            ]
            assert [old for old, _ in changed] == table and table
            for old, new in changed:
                date, *labs, frontier = old.split()
                assert new.split() == [date, *labs[::-1], frontier]


def test_an_arm_that_changes_nothing_is_not_built():
    """A context without path-linked scales has no fixed-scale arm, and one
    without a trajectory no placebo."""
    instrument = contextual_instrument()
    assert instrument.path_linked_exposure is None
    assert fixed_scale_instrument(instrument) is None
    assert instrument.public_trajectory is None
    assert placebo_instrument(instrument) is None


def test_plan_refuses_the_catalog_an_existing_output_and_an_even_rule(
    planned, tmp_path, capsys
):
    bundle = read_pilot(planned)["source_bundle"]
    arguments = ["--source-bundle", bundle, "--context", str(STUDY / "context.json")]
    new = str(tmp_path / "new")
    catalog = [row.id for row in load_scenarios(scenarios_path())]
    for selection in (["all"], catalog):
        named = [
            option for scenario in selection for option in ("--scenario", scenario)
        ]
        assert main(["plan", new, *arguments, *named]) == 2
        assert "name the pilot's scenarios" in capsys.readouterr().err
    assert main(["plan", str(planned), *arguments, "--scenario", SCENARIOS[0]]) == 2
    assert "already exists" in capsys.readouterr().err
    assert (
        main(["plan", new, *arguments, "--scenario", SCENARIOS[0], "--nodes", "4"]) == 2
    )
    assert "odd number of at least 3 nodes" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# A respondent with a known response function


_DATED_ROW = re.compile(r"^\d{4}-\d{2}-\d{2}  .+$", re.MULTILINE)
_USE = re.compile(
    r"^Use: .*?about (\d[\d.,]*\d|\d)(?: (million|billion|trillion))?", re.MULTILINE
)
_SCENARIO = re.compile(r"^## Scenario ([^\s:]+)", re.MULTILINE)
_WORDS = {None: 1.0, "million": 1e6, "billion": 1e9, "trillion": 1e12}


@dataclass(frozen=True)
class Shown:
    """What a question shows of its trajectory and its use: the frontier at
    the end of the study year, the first lab column's value there, and the
    first deployment scale its use line states."""

    frontier: float
    first_lab: float
    scale: float


def _shown(user_prompt: str) -> Shown:
    """Read a question as a respondent does, from its text alone."""
    last = _DATED_ROW.findall(user_prompt)[-1].split()
    use = _USE.search(user_prompt)
    assert use is not None
    return Shown(
        frontier=float(last[-1]),
        first_lab=float(last[1]),
        scale=float(use.group(1).replace(",", "")) * _WORDS[use.group(2)],
    )


@dataclass(frozen=True)
class Law:
    """An expected count of ``0.001 * exp(rate * (level - 170)) * scale **
    elasticity``, where the level is the frontier at the end of the study
    year or, for a respondent that reads the table by position, the first lab
    column there."""

    rate: float = 0.0
    elasticity: float = 0.0
    by_position: bool = False

    def log_count(self, shown: Shown) -> float:
        level = shown.first_lab if self.by_position else shown.frontier
        return (
            math.log(0.001)
            + self.rate * (level - 170)
            + self.elasticity * math.log(shown.scale)
        )


LAWS = {
    # Responds to the path and to the stated scale.
    (SCENARIOS[0], "public"): Law(rate=0.02, elasticity=0.5),
    # Responds to the path, read by column position: placebo-sensitive.
    (SCENARIOS[0], "not_yet_public"): Law(rate=0.02, by_position=True),
    # Responds to the stated scale only.
    (SCENARIOS[1], "public"): Law(elasticity=0.5),
    # Ignores the trajectory and the scale.
    (SCENARIOS[1], "not_yet_public"): Law(),
}
EXPECTED_VERDICTS = {
    (SCENARIOS[0], "public"): (
        "responds; with scales fixed: responds; moves with stated scale"
    ),
    # Whatever one lab's path does across the nodes, both arms show the same.
    (SCENARIOS[0], "not_yet_public"): (
        r"(responds|responds against the expected sign|flat|noisy); "
        r"with scales fixed: \1; placebo-sensitive"
    ),
    (SCENARIOS[1], "public"): (
        "responds; with scales fixed: flat; moves with stated scale"
    ),
    (SCENARIOS[1], "not_yet_public"): "flat; with scales fixed: flat",
}
"""Each law's verdict, as a pattern."""


class _Respondent:
    """A fake provider whose expected count follows ``laws`` from what its
    question shows, times a noise factor that cycles over the calls of one
    question. A question names no respondent, so each sample of a model is
    one more call of the same text."""

    def __init__(self, spec, nodes, laws, noise=(0.0,)):
        self.model_id = spec.model_id
        self.provider_name = spec.provider
        self.reasoning_effort = spec.reasoning_effort
        self.nodes, self.laws, self.noise = nodes, laws, noise
        self.asked: Counter[str] = Counter()

    async def complete(self, *, user_prompt, **_kwargs):
        heading = _SCENARIO.search(user_prompt)
        assert heading is not None
        scenario = heading.group(1)
        internal = "is not yet publicly available when" in user_prompt
        law = self.laws[scenario, "not_yet_public" if internal else "public"]
        noise = self.noise[self.asked[user_prompt] % len(self.noise)]
        self.asked[user_prompt] += 1
        count = math.exp(law.log_count(_shown(user_prompt)) + noise)
        answer = scenario_node_answer(self.nodes[scenario], count=count)
        return ProviderResponse(
            text=fixture_response(user_prompt, answer),
            model_id=self.model_id,
            provider_name=self.provider_name,
        )


def _respondents(pilot: Path, monkeypatch, laws, noise=(0.0,)) -> list[_Respondent]:
    """Stand one fake respondent in for each model of the pilot's profile."""
    template = ContextualRiskInstrument.model_validate(
        strict_json_loads((pilot / "templates" / "public.json").read_bytes())
    )
    nodes = {node.node_id: node for node in template.collectible_nodes}
    providers = [
        _Respondent(spec, nodes, laws, noise)
        for spec in load_model_profile(profile=PROFILE).models
    ]
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: providers
    )
    return providers


def _copy(planned: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(planned, tmp_path / "pilot"))


def _shown_by_contract(pilot: Path) -> dict[tuple[str, str, str], Shown]:
    """What each (arm, node, source status) shows, from its instrument's
    contract values and not from its question's text."""
    shown = {}
    for key, instrument in _instruments(pilot).items():
        condition, exposure = (
            instrument.public_trajectory,
            instrument.path_linked_exposure,
        )
        assert condition is not None and exposure is not None
        final, template = condition.values[-1], exposure.use_population_template
        assert template is not None
        scales = instrument.scale_levels()
        first = min(
            (scale for scale in scales if "{" + scale + "}" in template),
            key=lambda scale: template.index("{" + scale + "}"),
        )
        shown[key] = Shown(
            frontier=float(f"{max(final):.1f}"),
            first_lab=float(f"{final[0]:.1f}"),
            scale=scales[first],
        )
    return shown


def _written(analysis: Path) -> list[dict[str, str]]:
    with (analysis / "pilot.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _index(rows):
    return {
        (*(row[name] for name in FIELDS[:5]), row["node"] or ""): row for row in rows
    }


def test_collect_needs_one_mode_and_analyse_needs_answers(planned, tmp_path, capsys):
    pilot = _copy(planned, tmp_path)
    for flags in ([], ["--dry-run", "--live"]):
        assert main(["collect", str(pilot), *flags]) == 2
        assert "exactly one of --dry-run" in capsys.readouterr().err
    assert not (pilot / "arms" / "reference").exists()
    assert main(["collect", str(pilot), "--dry-run", "--arm", "placebo"]) == 0
    plan = read_pilot(pilot)
    frozen = list((pilot / "arms").rglob("request.json"))
    assert len(frozen) == plan["initial_calls"] // len(ARMS)
    assert {path.parents[2] for path in frozen} == {
        pilot / row["capture"] for row in plan["contexts"] if row["arm"] == "placebo"
    }
    assert not list(pilot.rglob("contextual-risk.json"))
    assert main(["analyse", str(pilot), "--output", str(tmp_path / "analysis")]) == 2
    assert "no arm of the pilot has been collected" in capsys.readouterr().err
    assert not (tmp_path / "analysis").exists()


def _check_statistics(index, shown, x, models: int) -> None:
    """Every statistic of a fully collected pilot is the one ``LAWS`` and the
    noise imply. ``shown`` is what each (arm, node, source status) shows and
    ``x`` the nodes' frontier progress in tens of ECI points."""
    spread = float(((x - x.mean()) ** 2).sum())
    # The standard error of one group's mean log answer.
    group = math.sqrt(statistics.variance(NOISES) / SAMPLES)

    def close(row, value, error):
        assert row["value"] == pytest.approx(value, abs=1e-9)
        assert row["standard_error"] == pytest.approx(error, rel=1e-9)
        # A node's mean log answer is a level and carries no z.
        assert row["z"] == (
            None if row["node"] else pytest.approx(row["value"] / row["standard_error"])
        )

    for (scenario, status), law in LAWS.items():
        means = {
            arm: np.array([law.log_count(shown[arm, node, status]) for node in NODES])
            for arm in ARMS
        }
        slopes = {arm: float(np.polyfit(x, y, 1)[0]) for arm, y in means.items()}
        ranges = {arm: float(y[-1] - y[0]) for arm, y in means.items()}
        for quantity in QUANTITIES:
            key = (scenario, status, quantity)
            # A direct EAL is a constant multiple of its count in every answer.
            offset = math.log(2.7e6) if quantity == "direct_eal" else 0.0
            for arm in ARMS:
                slope = index[*key, arm, "log_slope_per_10_eci", ""]
                close(slope, slopes[arm], group / math.sqrt(models * spread))
                step = index[*key, arm, "fastest_to_slowest_log_ratio", ""]
                close(step, ranges[arm], group * math.sqrt(2 / models))
                floor = index[*key, arm, "noise_floor_log_sd", ""]
                assert floor["value"] == pytest.approx(statistics.stdev(NOISES))
                used = index[*key, arm, "answers_used", ""]
                assert used["value"] == SAMPLES * models * len(NODES)
                assert not any(
                    row[4].startswith("excluded_")
                    for row in index
                    if row[:4] == (*key, arm)
                )
                for node, mean in zip(NODES, means[arm], strict=True):
                    row = index[*key, arm, "node_mean_log", node]
                    close(row, mean + offset, group / math.sqrt(models))
            # The reference slope less the fixed-scale slope is the part that
            # follows the stated scale, and nothing else.
            scale = index[
                *key, "reference_minus_fixed_scale", "log_slope_per_10_eci", ""
            ]
            close(
                scale,
                slopes["reference"] - slopes["fixed_scale"],
                group * math.sqrt(2 / (models * spread)),
            )
            step = index[
                *key, "reference_minus_fixed_scale", "fastest_to_slowest_log_ratio", ""
            ]
            close(
                step,
                ranges["reference"] - ranges["fixed_scale"],
                group * math.sqrt(4 / models),
            )
            placebo = index[*key, "placebo_minus_reference", "mean_log_ratio", ""]
            close(
                placebo,
                float((means["placebo"] - means["reference"]).mean()),
                group * math.sqrt(2 / (models * len(NODES))),
            )
            if law.elasticity:
                assert scale["value"] > 0 and scale["z"] > NOISE_Z
            else:
                assert scale["value"] == pytest.approx(0, abs=1e-9)
            if law.by_position:
                assert abs(placebo["z"]) > NOISE_Z
            else:
                # A ratio of one: the reordered table moves no answer.
                assert placebo["value"] == pytest.approx(0, abs=1e-9)
            note = index[*key, "", "verdict", ""]["note"]
            assert re.fullmatch(EXPECTED_VERDICTS[scenario, status], note)
        # With scales fixed the response is the path's part alone, and the
        # difference between the arms is the stated scale's part alone.
        slow, fast = (
            {arm: shown[arm, node, status] for arm in ARMS}
            for node in (NODES[0], NODES[-1])
        )
        level = "first_lab" if law.by_position else "frontier"
        assert ranges["fixed_scale"] == pytest.approx(
            law.rate
            * (
                getattr(fast["fixed_scale"], level)
                - getattr(slow["fixed_scale"], level)
            )
        )
        assert ranges["reference"] - ranges["fixed_scale"] == pytest.approx(
            law.elasticity * math.log(fast["reference"].scale / slow["reference"].scale)
        )
        assert fast["fixed_scale"].scale == slow["fixed_scale"].scale
        assert fast["reference"].scale > slow["reference"].scale


def test_the_analysis_recovers_a_known_response(planned, tmp_path, monkeypatch, capsys):
    """Dry-run, collect and analyse a pilot whose respondents answer from
    ``LAWS``: the slopes, their split into path and stated scale, the placebo
    ratio and every standard error are the ones the laws imply."""
    pilot = _copy(planned, tmp_path)
    plan = read_pilot(pilot)
    providers = _respondents(pilot, monkeypatch, LAWS, NOISES)
    models = len(providers)

    # The reference arm alone, once a dry run has frozen its questions: the
    # other arms are named as not collected and a verdict rests on its slope.
    assert main(["collect", str(pilot), "--dry-run", "--arm", "reference"]) == 0
    assert len(list(pilot.rglob("request.json"))) == plan["initial_calls"] // len(ARMS)
    assert not any(provider.asked for provider in providers)
    assert main(["collect", str(pilot), "--live", "--arm", "reference"]) == 0
    capsys.readouterr()
    partial = tmp_path / "partial"
    assert main(["analyse", str(pilot), "--output", str(partial)]) == 0
    printed = capsys.readouterr().out
    assert (
        "Arms analysed: reference. Arm fixed_scale is left out: not collected. "
        "Arm placebo is left out: not collected.\n"
    ) in (partial / "pilot.md").read_text(encoding="utf-8")
    written = _written(partial)
    assert {row["arm"] for row in written} == {"reference", ""}
    for key, expected in EXPECTED_VERDICTS.items():
        note = _index(written)[*key, "direct_count", "", "verdict", ""]["note"]
        assert re.fullmatch(expected.split(";")[0], note)
        assert "  ".join((*key, "direct_count", note)) + "\n" in printed
    # An arm collected in part stops the analysis.
    capture = next(
        row["capture"] for row in plan["contexts"] if row["arm"] == "reference"
    )
    (pilot / capture / "contextual-risk.json").unlink()
    with pytest.raises(ValueError, match="1 of 8 collections are incomplete"):
        analyse_pilot(pilot)

    # Every arm. The interrupted collection resumes from its stored calls.
    assert main(["collect", str(pilot), "--live"]) == 0
    for provider in providers:
        # Every initial question was asked once per sample, no arm's question
        # repeats another's, and no answer drew a retry or a cross-check.
        assert set(provider.asked.values()) == {SAMPLES}
        assert sum(provider.asked.values()) == plan["initial_calls"] // models
        assert {request_kind(text) for text in provider.asked} == {"direct", "chain"}
    output = tmp_path / "analysis"
    rows = write_analysis(pilot, output)
    index = _index(rows)
    x = np.array([plan["nodes"][node] for node in NODES]) / 10
    _check_statistics(index, _shown_by_contract(pilot), x, models)
    # Each step of the first scenario's public response is far above its
    # noise, so every respondent's answers rise from node to node.
    share = index[
        *next(iter(LAWS)), "direct_count", "reference", "never_falls_share", ""
    ]
    assert (share["value"], share["answers"]) == (1.0, SAMPLES * models)

    written = _written(output)
    assert tuple(written[0]) == FIELDS and len(written) == len(rows)
    for ours, theirs in zip(written, rows, strict=True):
        assert [ours[name] for name in FIELDS[:5]] == [
            theirs[name] for name in FIELDS[:5]
        ]
        assert ours["note"] == (theirs["note"] or "")
        if theirs["value"] is not None:
            assert float(ours["value"]) == pytest.approx(theirs["value"], rel=1e-8)
    report = (output / "pilot.md").read_text(encoding="utf-8")
    assert RULES in report
    assert all(f"- `{arm}`: {text}" in report for arm, text in ARMS.items())
    assert "Arms analysed: reference, fixed_scale, placebo.\n" in report
    line = next(
        line
        for line in report.split("\n")
        if line.startswith(f"| {SCENARIOS[1]} | public | direct_count | ")
    )
    assert line.endswith(f"| {EXPECTED_VERDICTS[SCENARIOS[1], 'public']} |")
    # Eleven cells: a figure reads as its change, standard error and z.
    cells = line.strip("|").split(" | ")
    assert len(cells) == 11 and "-0.0" not in line
    assert re.fullmatch(r"\+\d+\.\d% ± \d+\.\d% \(z \d+\.\d\)", cells[3].strip())
    assert re.fullmatch(r"\+0\.0% ± \d+\.\d% \(z 0\.0\)", cells[4].strip())
    assert re.fullmatch(r"\+0\.00 ± 0\.\d\d \(z 0\.0\)", cells[9].strip())
    used = SAMPLES * models * len(NODES) * len(LAWS)
    assert f"| direct_count | placebo | {used} | 0 | 0 | 0 |" in report
    # The analysis directory is new each time.
    assert main(["analyse", str(pilot), "--output", str(output)]) == 2


def test_answers_left_out_are_counted_and_leave_their_own_group_only(
    planned, tmp_path, monkeypatch
):
    """An unavailable answer, an exhausted question and a zero count are each
    counted and leave one node and model; the other answers still enter."""
    pilot = _copy(planned, tmp_path)
    plan = read_pilot(pilot)
    instruments = _instruments(pilot)
    captures = {
        pilot / row["capture"]: instruments[
            row["arm"], row["draw_id"], row["source_model_status"]
        ]
        for row in plan["contexts"]
        if row["arm"] == "reference"
    }
    first = plan["models"][0]
    unavailable = contextual_answer().model_copy(
        update={
            "status": "unavailable",
            "unavailable_reason": "Declined in this fixture",
            "rate": None,
            "branches": (),
            "direct_eal": None,
            "lab_compensation": None,
            "lab_annual_business_loss": None,
        }
    )

    def answer(prompt):
        """A count of 12 everywhere, but for three answers of the first
        scenario's public questions, each of a different respondent."""
        who = (prompt.respondent.model_id, prompt.respondent.sample_index)
        where = (prompt.instrument.context_id, prompt.node.node_id)
        if where == ("node-00-public", SCENARIOS[0]) and who == (first, 0):
            return unavailable
        zero = where == ("node-1-4-public", SCENARIOS[0]) and who == (first, 1)
        return scenario_node_answer(prompt.node, count=0 if zero else 12)

    def collected(capture):
        instrument = captures[capture]
        artifact = fixture_artifact(instrument, answer)
        if instrument.context_id == "node-1-2-public":
            # A direct question that exhausted its attempts leaves no case.
            lost = next(
                case
                for case in artifact.cases
                if case.node_id == SCENARIOS[0] and case.respondent.model_id != first
            )
            artifact = artifact.model_copy(
                update={
                    "cases": tuple(case for case in artifact.cases if case != lost),
                    "unanswered_cases": (
                        UnansweredCase(
                            node_id=lost.node_id,
                            anchor_score=lost.anchor_score,
                            respondent=lost.respondent,
                            prompt_sha256=lost.prompt_sha256,
                            chain_answer=lost.chain_answer,
                            chain_prompt_sha256=lost.chain_prompt_sha256,
                        ),
                    ),
                }
            )
        return instrument, artifact, None

    monkeypatch.setattr(
        "scripts.conditioning_pilot.load_contextual_collection", collected
    )
    for capture in captures:
        capture.mkdir(parents=True)
        (capture / "contextual-risk.json").write_text("{}")
    rows = write_analysis(pilot, tmp_path / "analysis")
    index = _index(rows)
    respondents = SAMPLES * len(plan["models"])
    for quantity in QUANTITIES:
        key = (SCENARIOS[0], "public", quantity, "reference")
        assert index[*key, "answers_used", ""]["value"] == respondents * len(NODES) - 3
        for reason in ("unavailable", "exhausted", "no_log"):
            assert index[*key, f"excluded_{reason}", ""]["value"] == 1
        # Three respondents lack an answer on one node each.
        share = index[*key, "never_falls_share", ""]
        assert (share["value"], share["answers"]) == (1.0, respondents - 3)
        # A group left with one answer has no variance, so no standard error.
        slope = index[*key, "log_slope_per_10_eci", ""]
        assert slope["value"] == pytest.approx(0, abs=1e-9)
        assert slope["standard_error"] is None and slope["z"] is None
        verdicts = {
            status: index[SCENARIOS[0], status, quantity, "", "verdict", ""]["note"]
            for status in STATUSES
        }
        assert verdicts == {"public": "not estimable", "not_yet_public": "flat"}
        other = (SCENARIOS[1], "public", quantity, "reference")
        assert index[*other, "answers_used", ""]["value"] == respondents * len(NODES)
        assert not any(
            row[4].startswith("excluded_") for row in index if row[:4] == other
        )
    report = (tmp_path / "analysis" / "pilot.md").read_text(encoding="utf-8")
    used = respondents * len(NODES) * len(LAWS) - 3
    assert f"| direct_count | reference | {used} | 1 | 1 | 1 |" in report
    assert f"| direct_eal | reference | {used} | 1 | 1 | 1 |" in report
    assert "+0.0% (no standard error)" in report


# The statistics on their own


def test_a_combination_carries_the_respondent_sampling_error():
    """sum(w * mean) with variance sum(w^2 * s^2 / n) over independent groups."""
    groups = {
        ("reference", "slow", "a"): [1.0, 2.0, 3.0],
        ("reference", "fast", "a"): [0.0, 4.0],
        ("reference", "fast", "b"): [3.0],
    }
    slow, fast, alone = groups
    estimate = combine({slow: 1.0, fast: -0.5}, groups)
    assert estimate is not None
    assert estimate.value == pytest.approx(2.0 - 0.5 * 2.0)
    error = math.sqrt(1.0 / 3 + 0.25 * 8.0 / 2)
    assert estimate.error == pytest.approx(error)
    assert (estimate.answers, estimate.z) == (5, pytest.approx(1.0 / error))
    # One answer gives a mean and no variance.
    single = combine({alone: 1.0}, groups)
    assert single is not None
    assert (single.value, single.error, single.z) == (3.0, None, None)
    assert combine({}, groups) is None
    # A pair enters only when both its groups hold answers.
    absent = ("reference", "slow", "b")
    assert paired_weights(groups, [(fast, slow), (alone, absent)]) == {
        fast: 1.0,
        slow: -1.0,
    }
    assert noise_floor(groups.values()) == pytest.approx(math.sqrt((2 * 1.0 + 8.0) / 3))
    assert noise_floor([[1.0]]) is None


def test_z_at_a_zero_standard_error_is_zero_or_infinite():
    assert Estimate(0.0, 0.0, 4).z == 0.0
    assert Estimate(1e-12, 0.0, 4).z == 0.0
    assert Estimate(0.3, 0.0, 4).z == math.inf
    assert Estimate(-0.3, 0.0, 4).z == -math.inf
    assert Estimate(0.3, 0.1, 4).z == pytest.approx(3.0)
    assert Estimate(0.3, None, 4).z is None


def test_the_slope_averages_each_model_s_own_fit():
    progress = {"slow": 0.0, "middle": 1.0, "fast": 3.0}
    groups = {
        ("reference", "slow", "a"): [1.0, 1.0],
        ("reference", "middle", "a"): [2.0, 2.0],
        ("reference", "fast", "a"): [4.0, 4.0],
        # Model b has no answer on the fastest node: its slope uses two nodes.
        ("reference", "slow", "b"): [5.0, 7.0],
        ("reference", "middle", "b"): [6.0, 6.0],
        # Another arm's answers, and a model with one node, do not enter.
        ("placebo", "fast", "b"): [9.0, 9.0],
        ("reference", "slow", "c"): [1.0, 2.0],
    }
    weights = slope_weights(groups, "reference", progress)
    assert {group[2] for group in weights} == {"a", "b"}
    slope = combine(weights, groups)
    assert slope is not None
    # Model a rises one per unit of progress exactly; model b's two node
    # means are equal, and only its slowest group has any spread.
    assert slope.value == pytest.approx((1.0 + 0.0) / 2)
    assert slope.error == pytest.approx(math.sqrt(2.0 / 2) / 2)
    assert slope_weights(groups, "fixed_scale", progress) == {}


def test_never_falls_counts_the_respondents_with_an_answer_on_every_node():
    nodes = ["slow", "middle", "fast"]
    answers = {
        # Sample 0 of model a never falls; its sample 1 falls on the last node.
        ("reference", "slow", "a"): {0: 1.0, 1: 1.0},
        ("reference", "middle", "a"): {0: 1.0, 1: 2.0},
        ("reference", "fast", "a"): {0: 3.0, 1: 1.5},
        # Model b's sample has no answer on the middle node.
        ("reference", "slow", "b"): {0: 1.0},
        ("reference", "fast", "b"): {0: 2.0},
        ("placebo", "slow", "a"): {0: 5.0},
    }
    assert never_falls(answers, "reference", nodes) == (0.5, 2)
    assert never_falls(answers, "placebo", nodes) is None


@pytest.mark.parametrize(
    ("reference", "fixed", "scale", "placebo", "expected"),
    [
        ((0.01, 0.004), None, None, None, "responds"),
        ((-0.01, 0.004), None, None, None, "responds against the expected sign"),
        ((0.001, 0.001), None, None, None, "flat"),
        # The same z with a standard error too wide to see a 25% change.
        ((0.004, 0.004), None, None, None, "noisy"),
        ((0.004, None), None, None, None, "not estimable"),
        (None, None, None, None, "not estimable"),
        (
            (0.01, 0.004),
            (0.001, 0.001),
            (0.009, 0.004),
            (0.02, 0.03),
            "responds; with scales fixed: flat; moves with stated scale",
        ),
        (
            (0.001, 0.001),
            (0.001, 0.001),
            (0.0, 0.002),
            (0.3, 0.1),
            "flat; with scales fixed: flat; placebo-sensitive",
        ),
        ((0.001, 0.001), None, None, (0.1, 0.2), "flat; placebo noisy"),
    ],
)
def test_each_verdict_follows_its_stated_rule(
    reference, fixed, scale, placebo, expected
):
    span = 34.0
    # The rule for flat, on the fitted change over the plan's range.
    assert 0.001 * span < math.log(RESOLVED_CHANGE) / NOISE_Z < 0.004 * span
    estimates = (
        None if pair is None else Estimate(pair[0], pair[1], 12)
        for pair in (reference, fixed, scale, placebo)
    )
    assert verdict(*estimates, span) == expected
