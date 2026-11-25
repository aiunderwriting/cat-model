"""Ask selected models of a profile for one tiny answer at their output ceilings.

A provider that rejects a model's recorded ``max_output_tokens`` fails a
collection's first call to that model; this optional probe finds that out
before a run (runbook section 3, ``NUM-CALL-LIMITS``). It makes one live call
per model with credentials, asks for one word, and exits nonzero if any model
rejects its ceiling or has no API key.

By default it probes every profile model. Repeat ``--model MODEL_ID`` to probe
only selected roster entries; an ID outside the profile is rejected with its
available IDs. ``--profiles FILE`` reads the profile from a model profiles file
other than the default (``CATMODEL_PROFILES``, else AIUC's packaged example).

    uv run python scripts/probe_output_ceilings.py --profile test
    uv run python scripts/probe_output_ceilings.py --profile example --model MODEL_ID
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from catmodel.config import load_model_profile, settings
from catmodel.llm import transport
from catmodel.llm.factory import build_providers
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import LLMProvider, with_open_runs_cancelled

PROMPT = "Reply with the single word OK."


def select_models(
    specs: Sequence[ModelSpec], model_ids: Sequence[str], profile: str
) -> tuple[ModelSpec, ...]:
    """Select a profile subset in roster order, or report its valid ids."""
    available = tuple(spec.model_id for spec in specs)
    unknown = sorted(set(model_ids) - set(available))
    if unknown:
        raise ValueError(
            f"unknown model id(s) for profile {profile!r}: {', '.join(unknown)}; "
            f"valid profile ids: {', '.join(available)}"
        )
    if not model_ids:
        return tuple(specs)
    selected = set(model_ids)
    return tuple(spec for spec in specs if spec.model_id in selected)


async def probe(
    specs: Sequence[ModelSpec],
    providers: Sequence[LLMProvider],
    request_timeout_seconds: float,
) -> tuple[list[str], list[str]]:
    """Each model's outcome at its ceiling: (accepted, failed) lines."""
    by_model = {provider.model_id: provider for provider in providers}
    accepted, failed = [], []
    for spec in specs:
        provider = by_model.get(spec.model_id)
        if provider is None:
            failed.append(f"{spec.identity}: no API key")
            continue
        try:
            response = await transport.complete(
                provider,
                system_prompt=PROMPT,
                user_prompt=PROMPT,
                max_tokens=spec.max_output_tokens,
                request_timeout_seconds=request_timeout_seconds,
            )
        except Exception as exc:
            failed.append(f"{spec.identity}: {type(exc).__name__}: {exc}")
        else:
            accepted.append(
                f"{spec.identity}: accepted max_output_tokens="
                f"{spec.max_output_tokens} ({response.output_tokens} tokens used)"
            )
    return accepted, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--profiles",
        type=Path,
        help="model profiles file to read --profile from (default: the file "
        "CATMODEL_PROFILES names, else AIUC's example "
        "data/example_model_profiles.yaml)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="probe one model from the profile; repeat to select several",
    )
    args = parser.parse_args(argv)
    try:
        profile = load_model_profile(profile=args.profile, path=args.profiles)
        specs = select_models(profile.models, args.model, args.profile)
    except ValueError as exc:
        parser.error(str(exc))
    providers = build_providers(
        specs,
        api_keys=settings.provider_api_keys(),
        concurrency_limits={spec.model_id: 1 for spec in specs},
    )
    accepted, failed = asyncio.run(
        with_open_runs_cancelled(
            probe(specs, providers, profile.settings.request_timeout_seconds)
        )
    )
    print("\n".join(accepted))
    if failed:
        print("\n".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
