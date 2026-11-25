"""Lightweight model identities and provider configuration values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderName = Literal["openai", "anthropic", "google", "xai", "meta", "openrouter"]
ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]
REASONING_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Frozen with execution settings: every provider receives the caller's system
# prompt in the system role, then the user prompt in the user role. Cache
# metadata does not change this layout; changing roles or order needs a new value.
PROVIDER_MESSAGE_LAYOUT: Literal["system_then_user"] = "system_then_user"

PROVIDER_API_KEY_ENV: dict[ProviderName, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "meta": "META_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class ModelSpec(BaseModel):
    """One configured model and its provider-neutral reasoning effort."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: ProviderName
    model_id: str
    reasoning_effort: ReasoningEffort
    max_output_tokens: Annotated[int, Field(ge=1)]
    """The largest output-token count one request to this model may ask for,
    as the model profile records it. A collection asks for the smaller of it
    and the elicitation output budget."""

    @model_validator(mode="before")
    @classmethod
    def _records_an_output_ceiling(cls, value: object) -> object:
        if isinstance(value, Mapping) and value.get("max_output_tokens") is None:
            raise ValueError(
                f"model {value.get('model_id')!r} needs a max_output_tokens ceiling"
            )
        return value

    @field_validator("provider", "model_id", mode="before")
    @classmethod
    def _nonempty_identity(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("must not be blank")
            return normalized
        return value

    @property
    def identity(self) -> str:
        """Canonical ``provider/model_id`` string."""
        return f"{self.provider}/{self.model_id}"
