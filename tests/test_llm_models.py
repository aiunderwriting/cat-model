"""Tests for lightweight LLM model identities."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from catmodel.llm.models import (
    PROVIDER_API_KEY_ENV,
    ModelSpec,
)


def test_model_spec_strips_identity_fields() -> None:
    spec = ModelSpec.model_validate(
        {
            "provider": " openai ",
            "model_id": " gpt-test ",
            "reasoning_effort": "medium",
            "max_output_tokens": 1000,
        }
    )

    assert spec.provider == "openai"
    assert spec.model_id == "gpt-test"
    assert spec.identity == "openai/gpt-test"
    assert spec.model_dump(mode="json") == {
        "provider": "openai",
        "model_id": "gpt-test",
        "reasoning_effort": "medium",
        "max_output_tokens": 1000,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("provider", "   "), ("model_id", "\t\n")],
)
def test_model_spec_rejects_blank_identity(field: str, value: str) -> None:
    payload = {
        "provider": "openai",
        "model_id": "gpt-test",
        "reasoning_effort": "low",
        "max_output_tokens": 1000,
    }
    payload[field] = value

    with pytest.raises(ValidationError, match="must not be blank"):
        ModelSpec.model_validate(payload)


def test_model_spec_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError, match="provider"):
        ModelSpec.model_validate(
            {
                "provider": "unknown",
                "model_id": "gpt-test",
                "reasoning_effort": "low",
                "max_output_tokens": 1000,
            }
        )


def test_model_spec_rejects_unknown_effort() -> None:
    with pytest.raises(ValidationError, match="reasoning_effort"):
        ModelSpec.model_validate(
            {
                "provider": "openai",
                "model_id": "gpt-test",
                "reasoning_effort": "extreme",
                "max_output_tokens": 1000,
            }
        )


def test_model_spec_requires_reasoning_effort() -> None:
    with pytest.raises(ValidationError, match="reasoning_effort"):
        ModelSpec.model_validate(
            {"provider": "openai", "model_id": "gpt-test", "max_output_tokens": 1000}
        )


@pytest.mark.parametrize("ceiling", [{}, {"max_output_tokens": None}])
def test_model_spec_requires_an_output_ceiling_and_names_the_model(ceiling) -> None:
    with pytest.raises(
        ValidationError, match="model 'gpt-test' needs a max_output_tokens ceiling"
    ):
        ModelSpec.model_validate(
            {
                "provider": "openai",
                "model_id": "gpt-test",
                "reasoning_effort": "low",
                **ceiling,
            }
        )


def test_model_spec_is_strict_and_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelSpec.model_validate(
            {
                "provider": "openai",
                "model_id": "gpt-test",
                "reasoning_effort": "low",
                "max_output_tokens": 1000,
                "extra": True,
            }
        )
    # Lax validation would decode bytes to a string; strict validation refuses.
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        ModelSpec.model_validate(
            {
                "provider": "openai",
                "model_id": b"gpt-test",
                "reasoning_effort": "low",
                "max_output_tokens": 1000,
            }
        )


def test_model_spec_is_frozen() -> None:
    spec = ModelSpec(
        provider="openai",
        model_id="gpt-test",
        reasoning_effort="low",
        max_output_tokens=1000,
    )

    with pytest.raises(ValidationError, match="frozen"):
        spec.model_id = "other"  # ty: ignore[invalid-assignment]


def test_provider_api_key_environment_mapping() -> None:
    assert PROVIDER_API_KEY_ENV == {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "meta": "META_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }


def test_llm_modules_have_a_cold_import_without_litellm() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import catmodel.llm\n"
                "import catmodel.llm.models\n"
                "import catmodel.llm.transport\n"
                "assert 'litellm' not in sys.modules\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
