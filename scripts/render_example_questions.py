"""Render the example questions under docs/example-questions from a planned study.

    uv run --frozen python scripts/render_example_questions.py ../hfl-dry-run/public-study

The study is the 15-node hypothetical-frontier-lab plan: the ``public-study``
directory that ``scripts/run_hypothetical_lab_study.py`` writes, for which
``--dry-run`` is enough. The files show one scenario's questions at the
forecast's median trajectory, and as diffs what changes for the lab's
not-yet-public models and on the fastest trajectory. Paragraphs are wrapped
for reading; a respondent receives each as one line.
``tests/test_example_questions.py`` compares the committed files with what
this script renders, and ``docs/example-questions/README.md`` describes them.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import textwrap
from pathlib import Path

from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.contracts.public_frontier import PublicFrontierQuadrature
from catmodel.elicitation.contextual_frequency import ContextualChainPrompt
from catmodel.elicitation.contextual_hierarchy import ContextualParentPrompt
from catmodel.elicitation.contextual_risk import ContextualPrompt
from catmodel.json_io import strict_json_loads
from catmodel.resources import source_checkout_root

SCENARIO = "output_reliance_software"
MEDIAN_NODE = "node-1-2"
"""The node at the base forecast's median frontier progress, in every rule."""
WIDTH = 100


def _instrument(study: Path, node: str, status: str) -> ContextualRiskInstrument:
    return ContextualRiskInstrument.from_saved(
        strict_json_loads(
            (study / "instruments" / f"{node}-{status}.json").read_bytes()
        )
    )


def _wrapped(text: str) -> str:
    """``text`` with every line longer than ``WIDTH`` broken at spaces."""
    return "\n".join(
        "\n".join(
            textwrap.wrap(line, WIDTH, break_long_words=False, break_on_hyphens=False)
        )
        if len(line) > WIDTH
        else line
        for line in text.split("\n")
    )


def _call(prompt) -> str:
    """Everything one call sends a respondent: its system and user prompts."""
    return (
        "======== system prompt ========\n"
        f"{_wrapped(prompt.system_prompt)}\n\n"
        "======== user prompt ========\n"
        f"{_wrapped(prompt.render())}\n"
    )


def _diff(before: str, after: str, before_name: str, after_name: str) -> str:
    return (
        "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=before_name,
                tofile=after_name,
                lineterm="",
                n=0,
            )
        )
        + "\n"
    )


def example_questions(study: Path) -> dict[str, str]:
    """The example files by name, rendered from a planned public study."""
    quadrature = PublicFrontierQuadrature.from_saved(
        strict_json_loads((study / "public-frontier.json").read_bytes())
    )
    fastest_node = quadrature.nodes[-1].draw_id

    def direct(node: str, status: str) -> str:
        instrument = _instrument(study, node, status)
        return _call(
            ContextualPrompt(
                instrument,
                next(row for row in instrument.nodes if row.node_id == SCENARIO),
                instrument.respondents[0],
                instrument.anchor_scores[0],
            )
        )

    instrument = _instrument(study, MEDIAN_NODE, "public")
    scenario = next(row for row in instrument.nodes if row.node_id == SCENARIO)
    parent = next(row for row in instrument.parents if SCENARIO in row.children)
    respondent, score = instrument.respondents[0], instrument.anchor_scores[0]
    median = direct(MEDIAN_NODE, "public")
    return {
        "direct.txt": median,
        "chain.txt": _call(
            ContextualChainPrompt(instrument, scenario, respondent, score)
        ),
        "parent.txt": _call(
            ContextualParentPrompt(instrument, parent, respondent, score)
        ),
        "not-yet-public.diff": _diff(
            median,
            direct(MEDIAN_NODE, "not_yet_public"),
            "direct.txt",
            "the same question about the lab's not-yet-public models",
        ),
        "fastest-trajectory.diff": _diff(
            median,
            direct(fastest_node, "public"),
            "direct.txt",
            f"the same question on the fastest trajectory ({fastest_node})",
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study", type=Path, help="a planned study's public-study directory"
    )
    checkout = source_checkout_root()
    parser.add_argument(
        "--output",
        type=Path,
        default=None if checkout is None else checkout / "docs" / "example-questions",
        help="directory to write the files to (default: docs/example-questions in the source checkout)",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output is required outside a source checkout")
    try:
        files = example_questions(args.study)
    except (OSError, ValueError, StopIteration) as exc:
        print(f"Error: {args.study} is not the planned study: {exc}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.output / name).write_text(text, encoding="utf-8")
        print(args.output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
