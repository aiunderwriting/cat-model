"""The research CLI exposes setup, run and results as separate stages."""

import click

from catmodel.cli import _GLOBAL_OPTIONS, cli

GLOBAL_OPTIONS = {"--profile", "--workers"}
COMMANDS = {
    "answers add": {"--answers", "COLLECTIONS"},
    "capability-forecast public-study": {
        "--source-bundle",
        "--forecast",
        "--context",
        "--scenario",
        "--variant",
        "--assumptions",
        "--nodes",
        "--output",
    },
    "capability-risk build-instrument": {
        "--source-bundle",
        "--context",
        "--scenario",
        "--source-status",
        "--output",
    },
    "capability-risk collect": {
        "--study",
        "--instrument",
        "--run-directory",
        "--source-bundle",
        "--dry-run",
        "--answers",
    },
    "capability-risk validate": {"--run-directory"},
    "package build-source-bundle": {"--recipe", "--output"},
    "results export": {
        "--setup",
        "--capture",
        "--results",
        "--include-answers",
        "--output",
    },
    "results replay": {"--setup", "--capture", "--weights", "--level", "--output"},
    "results summary": {"--results", "--output"},
    "run": {"--setup", "--output"},
    "setup build": {"--spec", "--plan", "--setup-id", "--evidence", "--output"},
    "setup import": {
        "--study",
        "--run-directory",
        "--setup-id",
        "--output",
        "--comparison-family",
        "--variant-study",
    },
    "setup inspect": {"--setup"},
}
"""Every command with its options and, in capitals, its arguments. Adding,
renaming or removing a command or an option is a change to this table."""


def _parameters(command: click.Command) -> set[str]:
    """A command's options by their flags and its arguments by their names."""
    return {
        name
        for parameter in command.params
        for name in (
            parameter.opts
            if isinstance(parameter, click.Option)
            else [parameter.human_readable_name]
        )
    }


def _leaf_commands(group: click.Group, path: tuple[str, ...] = ()):
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            yield from _leaf_commands(command, (*path, name))
        else:
            yield " ".join((*path, name)), command


def test_the_commands_and_their_options_are_exactly_the_listed_ones():
    assert {
        path: _parameters(command) for path, command in _leaf_commands(cli)
    } == COMMANDS
    # The root group takes the global options and --version; a group between
    # it and a command takes none.
    assert _parameters(cli) == GLOBAL_OPTIONS | {"--version"}
    assert set(_GLOBAL_OPTIONS) == GLOBAL_OPTIONS
    for group in cli.commands.values():
        if isinstance(group, click.Group):
            assert not group.params, group.name


def test_source_authoring_and_setup_run_results_remain_explicit():
    setup = cli.commands["setup"]
    assert isinstance(setup, click.Group)
    assert set(setup.commands) == {"build", "import", "inspect"}
    assert isinstance(cli.commands["run"], click.Command)
    assert not isinstance(cli.commands["run"], click.Group)
    results = cli.commands["results"]
    assert isinstance(results, click.Group)
    assert set(results.commands) == {"replay", "export", "summary"}
    package = cli.commands["package"]
    assert isinstance(package, click.Group)
    assert set(package.commands) == {"build-source-bundle"}


def test_cli_help_and_execution_import_do_not_load_provider_or_scalar_orchestration():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from catmodel.cli import cli; "
            "from catmodel.results.execution import run_setup; "
            "assert not any(m.startswith(('catmodel.llm.factory', "
            "'catmodel.llm.litellm_provider', 'litellm', "
            "'catmodel.collection.contextual_collection', 'catmodel.collection.frozen_call_collection')) "
            "for m in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_read_only_results_services_import_with_collection_modules_prohibited():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoCollection(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        blocked = (
            'catmodel.collection.contextual_collection',
            'catmodel.collection.frozen_call_collection',
            'catmodel.llm',
            'litellm',
        )
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('read-only service imported collection: ' + fullname)

sys.meta_path.insert(0, NoCollection())
import catmodel.setup.contextual_import
import catmodel.setup.public_study
import catmodel.results.results_export
import catmodel.results.results_summary
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_upstream_collection_services_defer_provider_transport():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from catmodel.collection.contextual_collection import run_contextual_collection; "
            "from catmodel.collection.frozen_call_collection import collect_frozen_call; "
            "assert not any(m.startswith(('catmodel.llm.factory', "
            "'catmodel.llm.litellm_provider', 'litellm')) for m in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
