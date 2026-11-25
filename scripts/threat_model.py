#!/usr/bin/env python3
"""Research briefs, draft checks and blind comparisons for threat models.

A threat model is researched before a scenario's quantities are asked
(docs/threat-models.md). This script is the maintainer's side of that stage;
it calls no model and reads only repository files and the files it is given.

    threat_model.py brief SCENARIO_ID --research-through 2026-10-31
        Print the research brief for one scenario and its digest. Give the
        brief to an agent that can retrieve and read sources.

    threat_model.py check DRAFT.yaml
        Check a returned draft against the contract and the scenario library,
        print it as a question would quote it with its length, and say
        whether it answers the current brief.

    threat_model.py compare REFERENCE.yaml CANDIDATE.yaml MAPPING.yaml
        Count how much of a published expert threat model, written out as
        REFERENCE, a blind CANDIDATE covers under a MAPPING of their parts.

A draft enters data/threat_models.yaml without its drafting notes.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from catmodel.resources import scenarios_path
from catmodel.scenarios.base_rates import SOURCE_CARDS_RELATIVE_PATH, SourceCardsFile
from catmodel.scenarios.loader import load_scenarios
from catmodel.scenarios.threat_models import (
    MAX_RENDERED_WORDS,
    ReferenceMapping,
    ThreatModel,
    answers_current_brief,
    brief_sha256,
    coverage_against_reference,
    research_brief,
    threat_model_errors,
)
from catmodel.yaml_io import safe_load

REPO_ROOT = Path(__file__).resolve().parent.parent


def _cards(root: Path) -> SourceCardsFile:
    return SourceCardsFile.model_validate(
        safe_load((root / SOURCE_CARDS_RELATIVE_PATH).read_text(encoding="utf-8"))
    )


def _threat_model(path: Path) -> ThreatModel:
    return ThreatModel.model_validate(safe_load(path.read_text(encoding="utf-8")))


def _brief(args: argparse.Namespace) -> int:
    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    scenario = scenarios.get(args.scenario_id)
    if scenario is None:
        print(f"Error: {args.scenario_id!r} is not a library scenario", file=sys.stderr)
        return 2
    brief = research_brief(
        scenario,
        _cards(args.project_root).scenario_cards(scenario.id),
        research_through=args.research_through,
    )
    print(brief)
    print(f"\nbrief_sha256: {brief_sha256(brief)}")
    return 0


def _check(args: argparse.Namespace) -> int:
    try:
        draft = _threat_model(args.draft)
    except (OSError, ValueError) as exc:
        print(f"Error: {args.draft}: {exc}", file=sys.stderr)
        return 1
    scenarios = load_scenarios(scenarios_path())
    errors = threat_model_errors((draft,), scenarios)
    for error in errors:
        print(f"Error: {error}", file=sys.stderr)
    if errors:
        return 1
    rendered = draft.render()
    print(rendered)
    cards = _cards(args.project_root)
    print(
        f"\nThe draft meets the contract and takes {len(rendered.split()):,} of "
        f"{MAX_RENDERED_WORDS:,} words in a question."
    )
    if draft.drafting is None:
        print(
            "It records no brief digest (drafting.brief_sha256), so whether it "
            "answers the current brief is not checked."
        )
    elif not answers_current_brief(
        draft,
        scenarios,
        {
            scenario_id: cards.scenario_cards(scenario_id)
            for scenario_id in cards.scenarios
        },
    ):
        print(
            "It answers an earlier brief: the scenario's source cards or the "
            "brief's wording have changed since it was written."
        )
    print(
        "The contract checks form only. Before the draft enters "
        "data/threat_models.yaml, open every cited source, check each finding "
        "against it and check that no step says how it is done. The register "
        "entry leaves out the drafting notes."
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        result = coverage_against_reference(
            _threat_model(args.reference),
            _threat_model(args.candidate),
            ReferenceMapping.model_validate(
                safe_load(args.mapping.read_text(encoding="utf-8"))
            ),
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Checkout holding the source cards (default: this repository).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    brief = commands.add_parser("brief", help="print a scenario's research brief")
    brief.add_argument("scenario_id")
    brief.add_argument(
        "--research-through",
        type=date.fromisoformat,
        required=True,
        help="Last date whose information the research may use (YYYY-MM-DD).",
    )
    brief.set_defaults(run=_brief)
    check = commands.add_parser("check", help="check a draft threat model")
    check.add_argument("draft", type=Path)
    check.set_defaults(run=_check)
    compare = commands.add_parser(
        "compare", help="count a blind candidate's coverage of a reference"
    )
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("mapping", type=Path)
    compare.set_defaults(run=_compare)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
