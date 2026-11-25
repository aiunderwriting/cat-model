"""Construction of one bounded provider per configured model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .litellm_provider import LiteLLMProvider
from .models import ModelSpec
from .transport import LLMProvider


def build_providers(
    model_specs: Sequence[ModelSpec],
    *,
    api_keys: Mapping[str, str | None],
    concurrency_limits: Mapping[str, int],
) -> list[LLMProvider]:
    """Build a provider, in roster order, for each model whose provider has an
    API key."""
    specs = list(model_specs)
    if not all(isinstance(spec, ModelSpec) for spec in specs):
        raise TypeError("model_specs must contain only ModelSpec instances")

    model_ids = [spec.model_id for spec in specs]
    duplicates = sorted(
        {model_id for model_id in model_ids if model_ids.count(model_id) > 1}
    )
    if duplicates:
        raise ValueError(
            "model_id values must be unique; duplicates: " + ", ".join(duplicates)
        )

    active: list[tuple[ModelSpec, str, int]] = []
    for spec in specs:
        api_key = api_keys.get(spec.provider)
        if api_key is None or api_key == "":
            continue
        if not api_key.strip():
            raise ValueError(
                f"{spec.provider} API key is set but blank; unset it or provide "
                "a real key"
            )
        if spec.model_id not in concurrency_limits:
            raise ValueError(f"missing concurrency limit for model {spec.model_id!r}")
        limit = concurrency_limits[spec.model_id]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(
                f"concurrency limit for model {spec.model_id!r} must be a positive integer"
            )
        active.append((spec, api_key, limit))

    return [
        LiteLLMProvider(
            api_key=api_key,
            provider_name=spec.provider,
            model=spec.model_id,
            max_concurrent=limit,
            reasoning_effort=spec.reasoning_effort,
        )
        for spec, api_key, limit in active
    ]
