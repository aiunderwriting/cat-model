"""Capture upstream source bundles."""

from pathlib import Path

import click

from catmodel.cli._helpers import FILE, fail, output_option


def register(cli: click.Group) -> None:
    cli.add_command(package)


@click.group()
def package() -> None:
    """Capture upstream source evidence."""


@package.command("build-source-bundle")
@click.option(
    "--recipe",
    "recipe_path",
    required=True,
    type=FILE,
    help="Source-input recipe (YAML) naming the authored files and the external "
    "sources they capture; its paths resolve beside it.",
)
@output_option("directory", "the source bundle")
def build_source_bundle_command(recipe_path: Path, output: Path) -> None:
    """Freeze authored upstream inputs as a checksummed source bundle.

    Run it from a clean committed Git checkout, whose commit the bundle's
    provenance records.
    """
    from catmodel.evidence.source_inputs import build_source_bundle

    try:
        manifest = build_source_bundle(recipe_path, output)
    except Exception as exc:
        fail(exc)
    click.echo(f"Source bundle built: {manifest.bundle_id} / {manifest.bundle_version}")
    click.echo(f"  Path: {output.resolve()}")
    click.echo(
        f"  Schema: {manifest.bundle_schema_version}; files: {len(manifest.files)}"
    )
