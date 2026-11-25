"""The example questions under docs/example-questions are what the committed
hypothetical-lab study renders."""

import difflib
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.contracts.public_frontier import PublicFrontierQuadrature
from catmodel.elicitation.contextual_risk import ContextualPrompt
from catmodel.evidence import source_inputs
from catmodel.evidence.source_inputs import build_source_bundle
from catmodel.json_io import strict_json_loads
from catmodel.resources import config_path, data_path
from catmodel.worktree import CodeProvenance
from scripts.render_example_questions import SCENARIO, example_questions, main
from scripts.run_hypothetical_lab_study import stage_authored_source

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "example-questions"
STUDY = config_path("studies", "hypothetical-frontier-lab")


def _run(*args):
    result = CliRunner().invoke(cli, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> Path:
    """The 15-node hypothetical-lab plan, built from the committed inputs by
    the study script's command."""
    root = tmp_path_factory.mktemp("example-questions")
    bundle = root / "source-bundle"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            source_inputs,
            "resolve_code_provenance",
            lambda _: CodeProvenance("a" * 40, True, "clean", None),
        )
        build_source_bundle(stage_authored_source(root / "authored-source"), bundle)
    _run(
        "--profile",
        "example",
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
        "--assumptions",
        STUDY / "public-assembly-assumptions.json",
        "--nodes",
        "15",
        "--output",
        root / "public-study",
    )
    return root / "public-study"


def test_the_committed_examples_are_what_the_study_renders(study, tmp_path):
    """docs/example-questions/README.md says how to regenerate them."""
    assert main([str(study), "--output", str(tmp_path)]) == 0
    rendered = {path.name: path.read_text() for path in tmp_path.iterdir()}
    assert rendered == example_questions(study)
    committed = {
        path.name: path.read_text()
        for path in EXAMPLES.iterdir()
        if path.name != "README.md"
    }
    assert set(committed) == set(rendered)
    for name, text in rendered.items():
        assert committed[name] == text, (
            f"docs/example-questions/{name} differs from the rendered question"
        )


def test_the_examples_wrap_paragraphs_without_changing_their_words(study):
    instrument = ContextualRiskInstrument.model_validate(
        strict_json_loads((study / "instruments" / "node-1-2-public.json").read_bytes())
    )
    prompt = ContextualPrompt(
        instrument,
        next(row for row in instrument.nodes if row.node_id == SCENARIO),
        instrument.respondents[0],
        instrument.anchor_scores[0],
    )
    example = example_questions(study)["direct.txt"]
    system, user = example.removeprefix("======== system prompt ========\n").split(
        "\n\n======== user prompt ========\n"
    )
    assert system.split() == prompt.system_prompt.split()
    assert user.split() == prompt.render().split()
    # Only source URLs and, last, the answer schema, one line of JSON without
    # spaces to break at, stay longer than the wrapping width.
    long_lines = [line for line in user.split("\n") if len(line) > 100]
    assert len(long_lines) >= 2
    assert all(line.startswith("https://") for line in long_lines[:-1])
    assert long_lines[-1].startswith("{")


def _direct(study: Path, node: str) -> str:
    instrument = ContextualRiskInstrument.model_validate(
        strict_json_loads((study / "instruments" / f"{node}-public.json").read_bytes())
    )
    return ContextualPrompt(
        instrument,
        next(row for row in instrument.nodes if row.node_id == SCENARIO),
        instrument.respondents[0],
        instrument.anchor_scores[0],
    ).render()


def test_a_question_differs_between_trajectories_only_in_its_path_and_scales(study):
    """docs/limitations.md, item 6: between the slowest and the fastest
    trajectory a direct question differs in about 0.3% of its characters."""
    quadrature = PublicFrontierQuadrature.model_validate(
        strict_json_loads((study / "public-frontier.json").read_bytes())
    )
    slowest, fastest = (
        _direct(study, node.draw_id)
        for node in (quadrature.nodes[1], quadrature.nodes[-1])
    )
    changed = [
        (slow, fast)
        for slow, fast in zip(slowest.split("\n"), fastest.split("\n"), strict=True)
        if slow != fast
    ]
    kinds = [
        "scales"
        if slow.startswith("Use: ")
        else "row"
        if re.match(r"\d{4}-\d{2}-\d{2}  ", slow)
        else "path"
        if slow.startswith("Assume that public AI capability follows this path")
        else slow
        for slow, fast in changed
    ]
    # The path's two rows, the paragraph above them, which names a change of
    # leader and the doublings the frontier's rise would be, and the
    # deployment scales stated for the trajectory.
    assert sorted(kinds) == ["path", "row", "row", "scales"]
    characters = sum(
        max(i2 - i1, j2 - j1)
        for slow, fast in changed
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, slow, fast, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    )
    assert 0.0028 < characters / len(slowest) < 0.0036
