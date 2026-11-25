from catmodel.linear_algebra_threads import use_one_linear_algebra_thread

# Every CLI process and its workers, before the command modules load NumPy.
use_one_linear_algebra_thread()

import click  # noqa: E402

from catmodel.cli import (  # noqa: E402
    _answers,
    _capability,
    _package,
    _workflow,
)
from catmodel.cli._helpers import GlobalOptions  # noqa: E402
from catmodel.logging import configure_logging  # noqa: E402
from catmodel.worker_pool import available_cpus  # noqa: E402

_GLOBAL_OPTIONS = ("--profile", "--workers")


class _Catmodel(click.Group):
    """The root group, which says where a misplaced global option goes."""

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except click.NoSuchOption as error:
            if error.option_name not in _GLOBAL_OPTIONS or error.ctx is None:
                raise
            command = error.ctx.command_path.removeprefix(ctx.command_path).strip()
            raise click.UsageError(
                f"{error.option_name} is a global option; give it before the "
                f"command: {ctx.command_path} {error.option_name} VALUE {command} ...",
                ctx=error.ctx,
            ) from error


@click.group(cls=_Catmodel)
@click.version_option(package_name="catmodel", prog_name="catmodel")
@click.option(
    "--profile",
    default="dev",
    show_default=True,
    help="Model profile naming the respondent models and collection settings, "
    "for the commands that build or collect instruments. It is read from the "
    "profiles file that $CATMODEL_PROFILES names, else from AIUC's example "
    "data/example_model_profiles.yaml.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=available_cpus,
    show_default="the CPUs available",
    help="Worker processes for the commands that work in parallel: "
    "capability-risk collect --study, setup import, run, and results replay "
    "and export. The numbers and identities in the results, and a "
    "collection's frozen files, do not depend on it; run and results replay "
    "record it in results.json under execution_context.",
)
@click.pass_context
def cli(ctx: click.Context, profile: str, workers: int) -> None:
    """Experimental catastrophe model for AI-related losses.

    Global options go before the command name, as in
    `catmodel --workers 4 run --setup SETUP --output RUN`.
    """
    configure_logging()
    ctx.obj = GlobalOptions(profile=profile, workers=workers)


_capability.register(cli)
_package.register(cli)
_answers.register(cli)
_workflow.register(cli)
