"""Write the JSON Schema documents of results.json and its diagnostics file.

The documents are generated from the results contracts
(``catmodel.contracts.results``) and committed under ``contracts/results/``.
``tests/test_results_schema.py`` fails until they are written again after a
contract changes:

    uv run python scripts/generate_results_schemas.py
"""

from pathlib import Path

from catmodel.contracts.results_schema import write_results_schemas

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "contracts" / "results"


def main() -> None:
    for path in write_results_schemas(DIRECTORY):
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
