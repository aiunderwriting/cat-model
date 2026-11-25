"""One linear-algebra thread per process.

A multithreaded BLAS stalls when worker processes already occupy every core,
so parallel work uses worker processes instead (``catmodel --workers``) and
each process computes on one thread. NumPy reads the thread count when it
loads its BLAS, so a program calls ``use_one_linear_algebra_thread`` before
its first import that loads NumPy; this module imports the standard library
only, so importing it loads nothing else.
"""

import os
import sys

_BLAS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def use_one_linear_algebra_thread() -> None:
    """Set one BLAS thread for this process and the workers it starts.

    A variable the environment sets keeps its value, but OpenBLAS reads
    OPENBLAS_NUM_THREADS before OMP_NUM_THREADS, so it gets one thread unless
    OPENBLAS_NUM_THREADS says otherwise. A process that loaded NumPy first
    keeps its BLAS threads and environment, which its workers inherit.
    """
    if "numpy" not in sys.modules:
        for variable in _BLAS_THREADS:
            os.environ.setdefault(variable, "1")
