"""The CLI runs linear algebra on one thread per process.

A multithreaded BLAS stalls beside busy worker processes, so the CLI, and
each script that loads NumPy without it, sets one thread before NumPy loads,
unless the environment says otherwise.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_UNSET = ("unset", "unset", "unset")
_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _shown(imports: str, *expressions: str) -> list[str]:
    """What a new Python process, started in the checkout without the thread
    variables, prints for ``expressions`` and then the variables after
    ``imports``."""
    environment = {
        key: value for key, value in os.environ.items() if key not in _VARIABLES
    }
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os, sys; {imports}; print({', '.join(expressions)}, "
            f"*(os.environ.get(name, 'unset') for name in {_VARIABLES!r}))",
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


@pytest.mark.parametrize(
    ("explicit", "numpy_first", "expected"),
    [
        ({}, False, ("1", "1", "1")),
        ({"OPENBLAS_NUM_THREADS": "3"}, False, ("3", "1", "1")),
        # Each variable the environment sets keeps its value; the others,
        # OpenBLAS's own included, are one.
        ({"OMP_NUM_THREADS": "4"}, False, ("1", "4", "1")),
        ({"MKL_NUM_THREADS": "2"}, False, ("1", "1", "2")),
        ({}, True, _UNSET),
    ],
)
def test_the_cli_runs_linear_algebra_on_one_thread_unless_told_otherwise(
    explicit, numpy_first, expected
):
    """One thread unless the environment sets a variable; a process that
    loaded NumPy first keeps its environment, so its workers, which inherit
    it, compute as it does."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
    }
    environment.update(explicit)
    imports = (
        "import numpy, os, catmodel.cli" if numpy_first else "import os, catmodel.cli"
    )
    shown = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{imports}; print(*(os.environ.get(name, 'unset') for name in "
            "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert tuple(shown.split()) == expected


def test_the_thread_setting_is_importable_without_the_cli_or_numpy():
    """A program can set the threads before anything loads NumPy, and
    without registering the CLI's commands."""
    assert _shown(
        "from catmodel.linear_algebra_threads import use_one_linear_algebra_thread",
        "sorted({'numpy', 'click', 'catmodel.cli'} & set(sys.modules))",
    ) == ["[]", *_UNSET]


@pytest.mark.parametrize(
    "script",
    [
        "backtest_public_frontier",
        "conditioning_pilot",
        "run_hypothetical_lab_study",
        "respondent_calibration",
    ],
)
def test_a_script_that_loads_numpy_sets_one_thread_first(script):
    """The variables are set only while NumPy is unloaded, so their values
    show the script set them before its imports loaded NumPy; it does so
    without the CLI."""
    assert _shown(
        f"import scripts.{script}",
        "'numpy' in sys.modules",
        "'catmodel.cli' in sys.modules",
    ) == ["True", "False", "1", "1", "1"]
