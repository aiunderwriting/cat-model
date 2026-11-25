"""Global options, shared option types and checks, and command failures."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NoReturn

import click

from catmodel.worker_pool import available_cpus

FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
DIRECTORY = click.Path(exists=True, file_okay=False, path_type=Path)


@dataclass(frozen=True)
class GlobalOptions:
    """The options given before the command name, which every command that
    uses them reads."""

    profile: str = "dev"
    workers: int = field(default_factory=available_cpus)


def global_options() -> GlobalOptions:
    """This invocation's global options, or their defaults for a command
    invoked without the ``catmodel`` group."""
    found = click.get_current_context().find_object(GlobalOptions)
    return GlobalOptions() if found is None else found


def _new_path(_ctx: click.Context, _param: click.Parameter, value: Path) -> Path:
    if value.exists() or value.is_symlink():
        raise click.BadParameter(
            f"{value} already exists; outputs are never overwritten, so name a new path."
        )
    return value


def output_option[F: Callable[..., object]](
    kind: Literal["file", "directory"], contents: str
) -> Callable[[F], F]:
    """``--output``: the new file or directory a command writes, which must
    not exist."""
    return click.option(
        "--output",
        required=True,
        type=click.Path(path_type=Path),
        callback=_new_path,
        help=f"New {kind} for {contents}; it must not exist.",
    )


def _built_setup(_ctx: click.Context, _param: click.Parameter, value: Path) -> Path:
    from catmodel.setup.store import MANIFEST

    if (value / MANIFEST).is_file():
        return value
    spec, plan = value / "setup.json", value / "plan.json"
    if spec.is_file() and plan.is_file():
        raise click.BadParameter(
            f"{value} holds authored setup inputs, not a built setup. Build one "
            f"first: catmodel setup build --spec {spec} --plan {plan} "
            "--setup-id ID --output NEW_DIRECTORY"
        )
    raise click.BadParameter(
        f"{value} is not a built setup: it has no {MANIFEST}. Pass a directory "
        "that setup build or setup import wrote."
    )


setup_option = click.option(
    "--setup",
    "setup_path",
    required=True,
    type=DIRECTORY,
    callback=_built_setup,
    help="Setup directory that setup build or setup import wrote.",
)


def _run_capture(_ctx: click.Context, _param: click.Parameter, value: Path) -> Path:
    from catmodel.results.execution import CAPTURE_MANIFEST

    if (value / CAPTURE_MANIFEST).is_file():
        return value
    if (value / "capture" / CAPTURE_MANIFEST).is_file():
        raise click.BadParameter(
            f"{value} is a run directory; pass the capture inside it: "
            f"--capture {value / 'capture'}"
        )
    raise click.BadParameter(
        f"{value} is not a run capture: it has no {CAPTURE_MANIFEST}. Pass the "
        "capture directory inside a run's output."
    )


capture_option = click.option(
    "--capture",
    "capture_directory",
    required=True,
    type=DIRECTORY,
    callback=_run_capture,
    help="Run capture: the capture directory inside the output of run.",
)


def fail(error: Exception) -> NoReturn:
    raise click.ClickException(str(error)) from error
