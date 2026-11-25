"""Commands for the answer table: responses kept apart from the collections that asked for them."""

import json
from pathlib import Path

import click

from catmodel.cli._helpers import DIRECTORY, fail

ANSWERS = click.Path(file_okay=False, path_type=Path)
ANSWERS_HELP = (
    "Answer table directory: provider responses keyed by their exact prompt, "
    "model settings and sample. A request the table already answers is not "
    "sent, and every response a provider returns is added. A new path starts "
    "a table."
)


def register(cli: click.Group) -> None:
    cli.add_command(answers)


@click.group()
def answers() -> None:
    """Keep collected responses in an answer table and reuse them."""


@answers.command("add")
@click.option(
    "--answers",
    "table",
    required=True,
    type=ANSWERS,
    help="Answer table directory; a new path starts a table.",
)
@click.argument("collections", nargs=-1, required=True, type=DIRECTORY)
def add(table: Path, collections: tuple[Path, ...]) -> None:
    """Add every response the COLLECTIONS store to the table.

    A collection is any directory that capability-risk collect or
    scripts/respondent_calibration.py collect wrote, complete or not, by this
    code revision or an earlier one with the same stored-call layout. Nothing is
    replayed, so a capture the current code can no longer import still gives
    its responses. The table is then checked, every prompt against its name
    and every row against its path. Prints the responses added, those already
    present and the number the table holds.
    """
    from catmodel.collection.answer_table import AddedRows, AnswerTable

    try:
        opened = AnswerTable(table)
        result = AddedRows()
        for collection in collections:
            result += opened.add_collection(collection)
        responses = opened.verify()
    except (ValueError, OSError) as error:
        fail(error)
    click.echo(
        json.dumps(
            {
                "added": result.added,
                "already_present": result.present,
                "collections": len(collections),
                "responses": responses,
            },
            sort_keys=True,
        )
    )
