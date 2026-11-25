"""Tests for configured provider construction."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from catmodel.llm import factory
from catmodel.llm.models import ModelSpec, ProviderName, ReasoningEffort


class _FakeProvider:
    instances: list[_FakeProvider] = []

    def __init__(
        self,
        *,
        api_key: str,
        provider_name: str,
        model: str,
        max_concurrent: int,
        reasoning_effort: str,
    ) -> None:
        self.api_key = api_key
        self.provider_name = provider_name
        self.model_id = model
        self.max_concurrent = max_concurrent
        self.reasoning_effort = reasoning_effort
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.instances.append(self)


def _spec(provider: str, model_id: str, effort: str = "low") -> ModelSpec:
    return ModelSpec(
        provider=cast(ProviderName, provider),
        model_id=model_id,
        reasoning_effort=cast(ReasoningEffort, effort),
        max_output_tokens=1000,
    )


@pytest.fixture(autouse=True)
def _reset_fake_instances(monkeypatch: pytest.MonkeyPatch):
    _FakeProvider.instances = []
    monkeypatch.setattr(factory, "LiteLLMProvider", _FakeProvider)


def test_build_providers_filters_unset_keys_and_preserves_order() -> None:
    providers = [
        cast(_FakeProvider, provider)
        for provider in factory.build_providers(
            [
                _spec("openai", "gpt-a"),
                _spec("google", "gemini-b"),
                _spec("anthropic", "claude-c"),
            ],
            api_keys={"openai": "openai-key", "google": None, "anthropic": ""},
            concurrency_limits={"gpt-a": 7, "gemini-b": 8, "claude-c": 9},
        )
    ]

    assert [provider.model_id for provider in providers] == ["gpt-a"]
    assert providers[0].api_key == "openai-key"
    assert providers[0].max_concurrent == 7


def test_build_providers_rejects_blank_key_before_building_any_provider() -> None:
    with pytest.raises(ValueError, match="openai API key is set but blank"):
        factory.build_providers(
            [_spec("openai", "gpt-a"), _spec("google", "gemini-b")],
            api_keys={"openai": "  ", "google": "google-key"},
            concurrency_limits={"gpt-a": 7, "gemini-b": 8},
        )
    assert _FakeProvider.instances == []


def test_build_providers_gives_each_model_its_own_limit_and_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from catmodel.llm.litellm_provider import LiteLLMProvider

    monkeypatch.setattr(factory, "LiteLLMProvider", LiteLLMProvider)
    providers = factory.build_providers(
        [
            _spec("openrouter", "model-a", "max"),
            _spec("openrouter", "model-b", "xhigh"),
        ],
        api_keys={"openrouter": "router-key"},
        concurrency_limits={"model-a": 2, "model-b": 5},
    )

    assert len(providers) == 2
    assert providers[0] is not providers[1]
    first = cast(Any, providers[0])
    second = cast(Any, providers[1])
    assert first._semaphore is not second._semaphore
    assert first._semaphore._value == 2
    assert second._semaphore._value == 5
    assert [provider.reasoning_effort for provider in providers] == ["max", "xhigh"]


def test_build_providers_requires_a_limit_for_each_active_model() -> None:
    with pytest.raises(ValueError, match=r"missing concurrency limit.*gpt-a"):
        factory.build_providers(
            [_spec("openai", "gpt-a")],
            api_keys={"openai": "openai-key"},
            concurrency_limits={},
        )
    assert _FakeProvider.instances == []


@pytest.mark.parametrize("limit", [0, -1, True])
def test_build_providers_requires_positive_integer_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        factory.build_providers(
            [_spec("openai", "gpt-a")],
            api_keys={"openai": "openai-key"},
            concurrency_limits={"gpt-a": cast(int, limit)},
        )


def test_build_providers_rejects_duplicate_model_ids_before_building() -> None:
    with pytest.raises(ValueError, match=r"unique.*duplicates: gpt-a"):
        factory.build_providers(
            [_spec("openai", "gpt-a"), _spec("google", "gpt-a")],
            api_keys={"openai": "openai-key", "google": "google-key"},
            concurrency_limits={"gpt-a": 2},
        )
    assert _FakeProvider.instances == []


def test_build_providers_rejects_non_model_specs() -> None:
    with pytest.raises(TypeError, match="ModelSpec"):
        factory.build_providers(
            cast(list[ModelSpec], [object()]),
            api_keys={},
            concurrency_limits={},
        )
