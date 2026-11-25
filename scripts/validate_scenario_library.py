#!/usr/bin/env python
"""Validate the scenario library and regenerate everything derived from it.

This is the one maintainer command for the scenario library. It loads every
scenario file and reports every error with its path, then checks the hierarchy
contract, the lineage and coverage-gap registers, evidence references and that
the source-cards file lists every scenario, and finally that each derived
artifact is current: the scenario catalog, the taxonomy explorer and the counts
that README.md and the docs state. With ``--fix`` it regenerates the derived
artifacts instead of reporting them stale; review the resulting diff. Run it
from a repository checkout or an unpacked source distribution: the wheel does
not carry the source-cards file. docs/adding-a-scenario.md walks through a change.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catmodel.scenarios.catalog import (  # noqa: E402
    DEFAULT_CATALOG_RELATIVE_PATH,
    validate_scenario_library,
)
from catmodel.scenarios.hierarchy_contract import load_hierarchy_contract  # noqa: E402
from catmodel.scenarios.loader import load_scenarios  # noqa: E402
from scripts.export_taxonomy import (  # noqa: E402
    DEFAULT_EXPLORER_RELATIVE_PATH,
    render_html,
    taxonomy_data,
)


def explorer_errors(scenarios_dir: Path, explorer: Path, *, fix: bool) -> list[str]:
    expected = render_html(
        taxonomy_data(load_hierarchy_contract(), load_scenarios(scenarios_dir))
    )
    actual = explorer.read_text(encoding="utf-8") if explorer.exists() else None
    if actual == expected:
        return []
    if fix:
        explorer.parent.mkdir(parents=True, exist_ok=True)
        explorer.write_text(expected, encoding="utf-8")
        return []
    return [f"ERROR taxonomy explorer {explorer} is stale; rerun with --fix"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=REPO_ROOT / "scenarios",
        help="Scenario YAML directory (default: <repo>/scenarios).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Project root holding the registers, the source cards and the "
            "documentation (default: repo)."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Regenerate the catalog, the explorer and the documented counts.",
    )
    args = parser.parse_args()

    scenarios_dir = args.scenarios_dir.resolve()
    project_root = args.project_root.resolve()
    errors = validate_scenario_library(
        scenarios_dir,
        project_root=project_root,
        catalog_path=scenarios_dir / DEFAULT_CATALOG_RELATIVE_PATH.name,
        fix=args.fix,
    )
    if not errors:
        errors = explorer_errors(
            scenarios_dir,
            project_root / DEFAULT_EXPLORER_RELATIVE_PATH,
            fix=args.fix,
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(
        "Scenario library validation passed"
        + ("; derived artifacts are current." if args.fix else ".")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
