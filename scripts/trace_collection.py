"""Trace a full collection of the committed hypothetical-lab study offline.

A development tool for scheduling work. It builds the study's instruments,
answers every question with synthetic fixtures from a fake provider whose
response times follow a latency law, and runs collection, as one study
collection (``--mode shared``) or as separate per-context runs each with a
fixed share of slots (``--mode per-context``), on a warping event loop:
provider waits pass instantly while local work runs in real time and is
charged to the clock. It reports wall clock, each model's slot utilisation
over time and idle slot-hours, the data-dependency critical path with
unlimited slots, requests per minute and, with ``--profile-cpu``, local CPU
per request by step (``CPU_STEPS``; a step is ``null`` when none of its
functions ran, and ``cpu_entries_without_samples`` lists every function that
did not). No provider is called, so nothing it reports is evidence about any
model.

Each model's slots are the profile's ceiling from its model profiles file
(``--profiles``; by default ``CATMODEL_PROFILES``, else AIUC's example
``data/example_model_profiles.yaml``); ``--slots PROVIDER=N`` overrides them
for one trace, for example to replay the caps a measured run used. Output
tokens and throughput are set per provider, for example
``--tokens-per-second anthropic=120``.

    uv run python scripts/trace_collection.py --samples 1 --contexts 2
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import hashlib
import json
import math
import pstats
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from catmodel.collection.contextual_collection import collect_study  # noqa: E402
from catmodel.config import ExecutionSettings, load_model_profile  # noqa: E402
from test_support.collection_trace import (  # noqa: E402
    CollectionTrace,
    LatencyProvider,
    ModelLatency,
    StudyResponder,
    TracedRequest,
    run_warped,
    traced_collection,
)

PROVIDER_LATENCY = {
    "openai": ModelLatency(output_tokens=24_000, tokens_per_second=80),
    "anthropic": ModelLatency(output_tokens=18_000, tokens_per_second=120),
}
"""Latencies by provider, matched to a test-profile run's long generations:
about 24k output tokens per generation from its OpenAI model and 18k from its
Anthropic model, roughly 95% reasoning. The throughputs are assumptions of
this tool, not measurements."""

LEAF_STAGES = {
    "direct": 0,
    "chain": 0,
    "direct_retry": 1,
    "chain_guardrail_retry": 1,
    "crosscheck": 2,
    "crosscheck_retry": 3,
}

CPU_STEPS = {
    "prompt rendering": (("elicitation/contextual_risk.py", "render"),),
    "frozen-call bookkeeping": (
        ("collection/frozen_call_collection.py", "collect_frozen_call"),
    ),
    "parser provenance": (("catmodel/frozen_calls.py", "parser_protocol"),),
    "capture replay": (("catmodel/frozen_calls.py", "replay_call"),),
    "chain numerics": (
        ("elicitation/contextual_frequency.py", "chain_count_law_for_answer"),
    ),
    "implied EAL": (
        ("setup/implied_eal.py", "recorded_crosscheck"),
        ("setup/implied_eal.py", "crosscheck_reconciliation"),
    ),
    "hierarchy values": (
        ("setup/contextual_hierarchy.py", "hierarchy_outcome"),
        ("setup/contextual_hierarchy.py", "recorded_parents"),
    ),
    "capture validation": (
        ("elicitation/contextual_frequency.py", "validate_contextual_artifact"),
        ("setup/implied_eal.py", "validate_eal_crosschecks"),
        ("setup/contextual_hierarchy.py", "validate_hierarchy_cases"),
    ),
}
"""Each step's entry functions, by file and name. Where one entry calls
another directly, the callee's time under that caller is subtracted, so
nothing counts twice; an entry reaching another only through other functions
is refused. Prompt rendering has one entry, ``RenderedOnce.render``: every
question of the direct, chain, parent and cross-check modules renders its
text through it, a retry question's original included, and the profiler
counts a function that re-enters itself once. Frozen-call bookkeeping
includes the fake provider's own work each time its request resumes, which
is small but not collection's."""


def seed_for(*parts: object) -> int:
    """A seed fixed by ``parts`` in every process, unlike ``hash``."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


# Measures


def _busy_intervals(trace: CollectionTrace, model_id: str):
    """Service intervals measured from the first request's ready time."""
    origin = min((row.ready for row in trace.requests), default=0.0)
    return sorted(
        (row.start - origin, row.end - origin)
        for row in trace.requests
        if row.model_id == model_id and row.start is not None and row.end is not None
    )


def utilisation(
    trace: CollectionTrace, model_id: str, ceiling: int, wall: float, bins: int = 20
) -> dict[str, Any]:
    """Slot-seconds held (including any 429 back-off) against the
    ceiling's, overall and in time bins."""
    intervals = _busy_intervals(trace, model_id)
    busy = math.fsum(end - start for start, end in intervals)
    edges = np.linspace(0, wall, bins + 1)
    timeline = []
    for left, right in pairwise(edges):
        overlap = math.fsum(
            max(0.0, min(end, right) - max(start, left)) for start, end in intervals
        )
        timeline.append(round(overlap / ((right - left) * ceiling), 3))
    capacity = ceiling * wall
    return {
        "ceiling": ceiling,
        "requests": len(intervals),
        "mean_utilisation": busy / capacity if capacity else None,
        "busy_slot_hours": busy / 3600,
        "idle_slot_hours": (capacity - busy) / 3600,
        "timeline": timeline,
    }


def critical_path(trace: CollectionTrace) -> dict[str, Any]:
    """The longest data-dependency chain with unlimited slots: every request
    starts when its prerequisites finish and takes its traced service time,
    excluding any 429 back-off.

    Prerequisites are those of each (context, anchor, respondent) group
    alone: each leaf stage needs the case's earlier direct or chain stages,
    and a parent question needs nothing. It is the bound a dependency-driven
    schedule can reach, not that of a schedule with barriers across groups or
    contexts.
    """
    finish: dict[int, tuple[float, int]] = {}
    groups: dict[tuple, list[TracedRequest]] = defaultdict(list)
    for row in trace.requests:
        groups[row.context_id, row.anchor_score, row.respondent].append(row)
    longest = (0.0, 0)
    for rows in groups.values():
        cases: dict[str, list[TracedRequest]] = defaultdict(list)
        parents = []
        for row in rows:
            if row.kind == "parent":
                parents.append(row)
            else:
                cases[row.subject].append(row)

        def service(row: TracedRequest) -> float:
            if row.admitted is None or row.end is None:
                return 0.0
            return row.end - row.admitted

        leaf_done = (0.0, 0)
        for case in cases.values():
            done_by_stage: dict[int, tuple[float, int]] = {}
            for row in sorted(case, key=lambda r: LEAF_STAGES[r.kind]):
                stage = LEAF_STAGES[row.kind]
                before = max(
                    (value for key, value in done_by_stage.items() if key < stage),
                    default=(0.0, 0),
                )
                value = (before[0] + service(row), before[1] + 1)
                finish[row.token] = value
                done_by_stage[stage] = max(done_by_stage.get(stage, (0.0, 0)), value)
            leaf_done = max(leaf_done, *done_by_stage.values())
        parent_done = max(((service(row), 1) for row in parents), default=(0.0, 0))
        longest = max(longest, leaf_done, parent_done)
    return {"seconds": longest[0], "turns": longest[1]}


def summary(
    trace: CollectionTrace, wall: float, ceilings: Mapping[str, int]
) -> dict[str, Any]:
    kinds: dict[str, int] = defaultdict(int)
    for row in trace.requests:
        kinds[row.kind] += 1
    return {
        "wall_clock_hours": wall / 3600,
        "requests": len(trace.requests),
        "requests_by_kind": dict(sorted(kinds.items())),
        "requests_per_minute": len(trace.requests) / (wall / 60) if wall else None,
        "rate_limited_retries": {
            model: len(times) for model, times in trace.rate_limits.items()
        },
        "critical_path": {
            **(path := critical_path(trace)),
            "hours": path["seconds"] / 3600,
            "definition": (
                "longest data-dependency chain of one (context, anchor, "
                "respondent) group with unlimited slots, excluding 429 back-off"
            ),
        },
        "by_model": {
            model: utilisation(trace, model, ceiling, wall)
            for model, ceiling in ceilings.items()
        },
    }


# The committed hypothetical-lab study


def build_study(
    root, *, samples: int, profile: str = "test", profiles: Path | None = None
):
    """Build the committed hypothetical-lab public study offline: a source
    bundle capturing the Epoch table and source cards, and the study planned
    from it, whose instruments ask ``samples`` samples of every model of the
    profile on every trajectory."""
    import json

    from catmodel.contracts.contextual_risk import ContextualRiskInstrument
    from catmodel.contracts.elicitation import Respondent
    from catmodel.evidence.source_bundle import SOURCE_CARDS_PATH
    from catmodel.forecast.public_frontier_epoch import EPOCH_ECI_SCORES_PATH
    from catmodel.resources import config_path, data_path
    from catmodel.setup.public_study import plan_public_frontier_study
    from catmodel.yaml_io import safe_load
    from test_support.capability_sources import (
        authored_source_files,
        write_source_bundle,
    )

    study = config_path("studies", "hypothetical-frontier-lab")
    table = data_path("capability", "epoch_eci_scores.csv").read_bytes()
    provenance = safe_load(
        data_path("capability", "epoch_eci_scores.provenance.yaml").read_text()
    )
    bundle = write_source_bundle(
        root / "source",
        files={
            **authored_source_files(),
            SOURCE_CARDS_PATH: data_path("elicitation_source_cards.yaml").read_bytes(),
            EPOCH_ECI_SCORES_PATH: table,
        },
        extra_sources=[
            {
                **provenance,
                "content_hash": hashlib.sha256(table).hexdigest(),
                "captured_paths": [EPOCH_ECI_SCORES_PATH],
            }
        ],
    )
    # The public-study command asks the profile's own samples per question; a
    # trace sets its own.
    respondents = tuple(
        Respondent(
            respondent_id=f"{spec.model_id}:{sample}",
            model_id=spec.model_id,
            sample_index=sample,
        )
        for spec in load_model_profile(profile=profile, path=profiles).models
        for sample in range(samples)
    )
    plan_public_frontier_study(
        source_bundle=bundle,
        forecast_path=data_path("capability", "public-frontier-forecast-template.json"),
        context_path=study / "context.json",
        respondents=respondents,
        scenario_ids=("all",),
        assumptions_path=study / "public-assembly-assumptions.json",
        output=root / "study",
    )
    plan = json.loads((root / "study" / "public-study.json").read_text())
    contexts = [
        (
            ContextualRiskInstrument.from_saved(
                json.loads((root / "study" / row["instrument"]).read_text())
            ),
            root / "study" / row["capture"],
        )
        for row in plan["contexts"]
    ]
    return bundle, contexts


async def collect_per_context(
    contexts,
    *,
    providers_for: Callable[[Any], list],
    concurrent_contexts: int,
    config,
) -> None:
    """Collection as separate per-context runs: up to ``concurrent_contexts``
    collections at once, each with its own providers and so its own fixed
    share of every model's slots, beside which ``collect_study`` is
    compared."""
    from catmodel.collection.contextual_collection import collect_contextual_risk

    gate = asyncio.Semaphore(concurrent_contexts)

    async def one(instrument, run_directory):
        async with gate:
            await collect_contextual_risk(
                instrument=instrument,
                run_directory=run_directory,
                providers=providers_for(instrument),
                config=config,
            )

    await asyncio.gather(*(one(*row) for row in contexts))


# CPU profile


@contextmanager
def threads_inline() -> Generator[None]:
    """Run ``asyncio.to_thread`` work on the loop thread, so a profile of
    the loop attributes it like any other call."""
    original = asyncio.to_thread

    async def inline(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(asyncio, "to_thread", inline)  # noqa: B010
    try:
        yield
    finally:
        setattr(asyncio, "to_thread", original)  # noqa: B010


def _matches(key: tuple[str, int, str], entry: tuple[str, str]) -> bool:
    return key[0].endswith(entry[0]) and key[2] == entry[1]


def _reaches(callees: Mapping[Any, Any], start: Any, goal: Any) -> bool:
    seen, pending = {start}, [start]
    while pending:
        for callee in callees.get(pending.pop(), ()):
            if callee == goal:
                return True
            if callee not in seen:
                seen.add(callee)
                pending.append(callee)
    return False


def _step_seconds(
    table: Mapping[Any, Any], callees: Mapping[Any, Any], step: str, entries
) -> float | None:
    """The union of the entries' cumulative times, or None when the profile
    holds none of them."""
    keys = [key for key in table if any(_matches(key, entry) for entry in entries)]
    if not keys:
        return None
    total = math.fsum(table[key][3] for key in keys)
    for key in keys:
        for caller, edge in table[key][4].items():
            if caller in keys and caller != key:
                total -= edge[3]
    for first in keys:
        for second in keys:
            if (
                first != second
                and second not in callees.get(first, ())
                and _reaches(callees, first, second)
            ):
                raise RuntimeError(f"{step}: {first[2]} reaches {second[2]}")
    return total


def cpu_breakdown(
    profile: cProfile.Profile, requests: int, process_seconds: float
) -> tuple[dict[str, float | None], list[str]]:
    """Local CPU milliseconds per request: the whole process, catmodel's own
    frames (summed ``tottime``, so nothing counts twice) and each step's
    entry functions (cumulative, see ``CPU_STEPS``). Steps nest within one
    another, so the step rows overlap and do not add up to the totals.

    A step none of whose entries ran under the profile is None, not zero: it
    was not measured. The second value names every entry that did not run,
    as ``step: file function``; an entry whose file or function has been
    renamed shows up there on any trace."""
    stats = pstats.Stats(profile)
    stats.calc_callees()
    table = vars(stats)["stats"]
    callees = {key: tuple(value) for key, value in vars(stats)["all_callees"].items()}
    own = math.fsum(row[2] for key, row in table.items() if "/catmodel/" in key[0])
    rows: dict[str, float | None] = {
        "whole process": 1000 * process_seconds / requests,
        "catmodel frames": 1000 * own / requests,
    }
    for step, entries in CPU_STEPS.items():
        seconds = _step_seconds(table, callees, step, entries)
        rows[step] = None if seconds is None else 1000 * seconds / requests
    without_samples = [
        f"{step}: {entry[0]} {entry[1]}"
        for step, entries in CPU_STEPS.items()
        for entry in entries
        if not any(_matches(key, entry) for key in table)
    ]
    return rows, without_samples


# The trace


def _per_provider(values: list[str], cast=float) -> dict[str, float]:
    settings = {}
    for value in values:
        provider, _, number = value.partition("=")
        if not number:
            raise SystemExit(f"expected PROVIDER=NUMBER, got {value!r}")
        settings[provider] = cast(number)
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--profile", default="test")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=None,
        help="model profiles file to read --profile from (default: the file "
        "CATMODEL_PROFILES names, else AIUC's example)",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--contexts", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("shared", "per-context"),
        default="shared",
        help="shared: every context at once through one lane per model (the "
        "study collection); per-context: --concurrent-contexts collections at "
        "a time, each with its own fixed share of every ceiling",
    )
    parser.add_argument("--concurrent-contexts", type=int, default=6)
    parser.add_argument(
        "--slots",
        action="append",
        default=[],
        metavar="PROVIDER=N",
        help="operational override of a provider's models' slot ceiling for "
        "this trace (default: the profile's limits)",
    )
    parser.add_argument("--output-tokens", action="append", default=[])
    parser.add_argument("--tokens-per-second", action="append", default=[])
    parser.add_argument("--tokens-per-minute", type=float, default=float("inf"))
    parser.add_argument(
        "--profile-cpu",
        action="store_true",
        help="profile local CPU and report it per request by step",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="write the raw CPU profile here (requires --profile-cpu)",
    )
    args = parser.parse_args(argv)
    if args.profile_output is not None and not args.profile_cpu:
        parser.error("--profile-output requires --profile-cpu")

    model_profile = load_model_profile(profile=args.profile, path=args.profiles)
    roster = {spec.model_id: spec.provider for spec in model_profile.models}
    limits = model_profile.execution_settings(
        model_profile.models, max_output_tokens=100_000
    ).concurrency_by_model
    slots = _per_provider(args.slots, int)
    ceilings = {
        model: int(slots.get(provider, limits[model]))
        for model, provider in roster.items()
    }
    output_tokens = _per_provider(args.output_tokens)
    throughput = _per_provider(args.tokens_per_second)
    latency = {}
    for model, provider in roster.items():
        law = PROVIDER_LATENCY.get(provider)
        tokens = output_tokens.get(provider) or (law.output_tokens if law else None)
        speed = throughput.get(provider) or (law.tokens_per_second if law else None)
        if tokens is None or speed is None:
            raise SystemExit(
                f"set --output-tokens and --tokens-per-second for {provider}"
            )
        latency[model] = ModelLatency(
            output_tokens=tokens,
            tokens_per_second=speed,
            tokens_per_minute=args.tokens_per_minute,
        )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, contexts = build_study(
            root, samples=args.samples, profile=args.profile, profiles=args.profiles
        )
        contexts = contexts[: args.contexts] if args.contexts else contexts
        instruments = [instrument for instrument, _ in contexts]
        responder = StudyResponder(instruments)
        trace = CollectionTrace()
        share = 1 if args.mode == "shared" else max(1, args.concurrent_contexts)
        slot_share = {
            model: max(1, ceiling // share) for model, ceiling in ceilings.items()
        }
        config = ExecutionSettings(
            samples_per_question=args.samples,
            max_output_tokens_by_model=dict.fromkeys(slot_share, 100_000),
            request_timeout_seconds=7200.0,
            concurrency_by_model=slot_share,
        )

        def providers_for(models, key):
            return [
                LatencyProvider(
                    model_id=model,
                    provider_name=roster[model],
                    latency=latency[model],
                    responder=responder,
                    trace=trace,
                    max_concurrent=slot_share[model],
                    seed=seed_for(key, model),
                )
                for model in sorted(models)
            ]

        if args.mode == "shared":
            collection = collect_study(
                contexts, providers=providers_for(roster, "study"), config=config
            )
        else:
            collection = collect_per_context(
                contexts,
                providers_for=lambda instrument: providers_for(
                    {row.model_id for row in instrument.respondents},
                    instrument.context_id,
                ),
                concurrent_contexts=args.concurrent_contexts,
                config=config,
            )
        profile = cProfile.Profile(time.process_time) if args.profile_cpu else None
        with traced_collection(trace):
            if profile is None:
                results, wall = run_warped(collection)
            else:
                with threads_inline():
                    cpu = time.process_time()
                    profile.enable()
                    results, wall = run_warped(collection)
                    profile.disable()
                    cpu = time.process_time() - cpu
        failed = [row for row in results or () if isinstance(row, Exception)]
        if failed:
            raise SystemExit(f"{len(failed)} collections failed: {failed[0]}")
        report = {
            "mode": args.mode,
            "profile": args.profile,
            "contexts": len(contexts),
            "samples_per_question": args.samples,
            "concurrent_contexts": (
                len(contexts) if args.mode == "shared" else args.concurrent_contexts
            ),
            **summary(trace, wall, ceilings),
        }
        if profile is not None:
            rows, without_samples = cpu_breakdown(profile, len(trace.requests), cpu)
            report["cpu_ms_per_request"] = rows
            report["cpu_entries_without_samples"] = without_samples
            if args.profile_output is not None:
                profile.dump_stats(args.profile_output)
    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
