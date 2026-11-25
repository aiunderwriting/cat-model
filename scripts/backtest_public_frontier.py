"""Score pace shapes by rolling-origin pinball loss on a forecast's releases.

A maintainer check of the public-frontier forecast's pace dispersion
(methodology.md, "Observable capability conditioning"). At each month-start
origin with a full trend window of earlier releases it refits the pooled rate
from the releases available then and scores the public frontier's quantiles
at 3, 6 and 12 months against the frontier the releases later reached. The
candidates are pace shapes 1.5, 2, 3, 5, 8 and 13, each truncated at the
forecast's pace cap, trend only and no progress, each from 2,000 sampled
paths. It writes the backtest to a new file and prints each candidate's mean
loss.

The forecast states its releases, as the ``forecast.json`` inside a planned
study directory does (``catmodel capability-forecast public-study``).

    uv run --frozen python scripts/backtest_public_frontier.py ../hfl-dry-run/public-study/forecast.json ../pace-backtest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from catmodel.linear_algebra_threads import use_one_linear_algebra_thread

# Before NumPy loads: the backtest refits the forecast at every origin, so
# this process takes the one linear-algebra thread every catmodel command
# runs with.
use_one_linear_algebra_thread()

from catmodel.setup.public_study import (  # noqa: E402
    backtest_public_frontier_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "forecast",
        type=Path,
        help="public-frontier forecast JSON with its releases",
    )
    parser.add_argument(
        "output", type=Path, help="new file for the backtest JSON; it must not exist"
    )
    args = parser.parse_args(argv)
    try:
        result = backtest_public_frontier_file(args.forecast, args.output)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scored_cases": result.scored_cases,
                "lowest_loss_candidate": result.lowest_loss_candidate,
                "mean_pinball_loss": {
                    row.candidate_id: row.mean_pinball_loss for row in result.candidates
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
