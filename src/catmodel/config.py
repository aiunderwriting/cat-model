"""Model profiles, collection execution settings and provider credentials."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from catmodel.llm.models import (
    PROVIDER_API_KEY_ENV,
    PROVIDER_MESSAGE_LAYOUT,
    ModelSpec,
    ProviderName,
)
from catmodel.resources import data_path
from catmodel.yaml_io import safe_load

EXAMPLE_PROFILES = "example_model_profiles.yaml"
"""AIUC's example model profiles, packaged as ``data/example_model_profiles.yaml``."""
PROFILES_ENV = "CATMODEL_PROFILES"
"""The environment variable that names a model profiles file to read instead
of the packaged example."""

_DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
_DEFAULT_MAX_CONCURRENCY = 20
_MAX_PROFILE_SAMPLES = 100
_MAX_CONCURRENCY = 1000
_MAX_OUTPUT_TOKENS = 100_000

_BoundedProfileInt = Annotated[int, Field(ge=1, le=_MAX_CONCURRENCY)]
_BoundedSampleCount = Annotated[int, Field(ge=1, le=_MAX_PROFILE_SAMPLES)]
_BoundedOutputTokens = Annotated[int, Field(ge=1, le=_MAX_OUTPUT_TOKENS)]


class ProfileSettings(BaseModel):
    """Strict settings authored in a model profile."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    samples_per_question: _BoundedSampleCount = 5
    max_concurrency: _BoundedProfileInt = _DEFAULT_MAX_CONCURRENCY
    max_concurrency_by_provider: Mapping[ProviderName, _BoundedProfileInt] = Field(
        default_factory=dict
    )
    request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS

    @field_validator("max_concurrency_by_provider", mode="before")
    @classmethod
    def _normalize_provider_limit_keys(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized: dict[object, object] = {}
        for raw_key, raw_limit in value.items():
            key = raw_key.strip() if isinstance(raw_key, str) else raw_key
            if key in normalized:
                raise ValueError(
                    "max_concurrency_by_provider contains duplicate keys after "
                    "whitespace stripping"
                )
            normalized[key] = raw_limit
        return normalized

    @field_validator("request_timeout_seconds", mode="before")
    @classmethod
    def _coerce_numeric_timeout(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                "request_timeout_seconds must be a number, not a boolean or string"
            )
        return float(value)

    @field_validator("request_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "request_timeout_seconds must be finite and greater than 0"
            )
        return value


class ExecutionSettings(BaseModel):
    """The validated settings of a collection run. Its plan freezes all but
    ``concurrency_by_model``, which only schedules requests and may change
    between runs."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    provider_message_layout: Literal["system_then_user"] = PROVIDER_MESSAGE_LAYOUT
    samples_per_question: _BoundedSampleCount
    max_output_tokens_by_model: Mapping[str, _BoundedOutputTokens]
    """Each model's output-token budget per request: the elicitation budget,
    lowered to the model's profile ceiling where that is smaller."""
    request_timeout_seconds: float
    concurrency_by_model: Mapping[str, _BoundedProfileInt]

    @model_validator(mode="after")
    def _same_models(self) -> Self:
        if set(self.max_output_tokens_by_model) != set(self.concurrency_by_model):
            raise ValueError(
                "output-token budgets and concurrency limits must name the same models"
            )
        return self

    @field_validator("request_timeout_seconds", mode="before")
    @classmethod
    def _coerce_numeric_timeout(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                "request_timeout_seconds must be a number, not a boolean or string"
            )
        return float(value)

    @field_validator("request_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "request_timeout_seconds must be finite and greater than 0"
            )
        return value


class ProfilesFile(BaseModel):
    """A model profiles file other than the packaged example: its path as
    given and the SHA-256 of the bytes read. Collection plans record it."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelProfile(BaseModel):
    """An ordered model roster and the strict settings governing its runs.

    ``profiles_file`` is the file the profile was read from, unless that was
    the packaged example."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    models: tuple[ModelSpec, ...] = Field(min_length=1)
    settings: ProfileSettings = Field(default_factory=ProfileSettings)
    profiles_file: ProfilesFile | None = None

    @field_validator("models", mode="before")
    @classmethod
    def _normalize_models(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("settings", mode="before")
    @classmethod
    def _default_null_settings(cls, value: object) -> object:
        return {} if value is None else value

    @model_validator(mode="after")
    def _validate_roster(self) -> Self:
        model_ids = [spec.model_id for spec in self.models]
        duplicates = sorted(
            {model_id for model_id in model_ids if model_ids.count(model_id) > 1}
        )
        if duplicates:
            raise ValueError(
                "model_id values must be unique; duplicates: " + ", ".join(duplicates)
            )
        providers = {spec.provider for spec in self.models}
        unrepresented = sorted(
            provider
            for provider in self.settings.max_concurrency_by_provider
            if provider not in providers
        )
        if unrepresented:
            raise ValueError(
                "max_concurrency_by_provider names providers absent from the "
                "profile: " + ", ".join(unrepresented)
            )
        return self

    def execution_settings(
        self,
        selected_specs: Sequence[ModelSpec],
        *,
        max_output_tokens: int,
    ) -> ExecutionSettings:
        """Resolve effective per-model limits for a selected profile subset.

        ``max_output_tokens`` is the elicitation output budget; each model
        asks for the smaller of it and its profile ceiling."""
        profile_models = set(self.models)
        selected = tuple(selected_specs)
        for spec in selected:
            if not isinstance(spec, ModelSpec) or spec not in profile_models:
                identity = getattr(spec, "identity", repr(spec))
                raise ValueError(f"selected model {identity!r} is not in the profile")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= _MAX_OUTPUT_TOKENS
        ):
            raise ValueError(
                f"the output budget must be an integer from 1 to {_MAX_OUTPUT_TOKENS:,}"
            )
        limits = {
            spec.model_id: self.settings.max_concurrency_by_provider.get(
                spec.provider, self.settings.max_concurrency
            )
            for spec in selected
        }
        return ExecutionSettings(
            samples_per_question=self.settings.samples_per_question,
            max_output_tokens_by_model={
                spec.model_id: min(max_output_tokens, spec.max_output_tokens)
                for spec in selected
            },
            request_timeout_seconds=self.settings.request_timeout_seconds,
            concurrency_by_model=limits,
        )


def _load_profiles(path: Path | str | None) -> tuple[Path, dict[str, ModelProfile]]:
    """Read and strictly validate every profile of a model profiles file:
    ``path``, else the file ``CATMODEL_PROFILES`` names, else the packaged
    example."""
    given = path if path is not None else os.environ.get(PROFILES_ENV) or None
    location = data_path(EXAMPLE_PROFILES) if given is None else Path(given)
    if not location.is_file():
        raise ValueError(f"model profiles file not found: {location}")
    data = location.read_bytes()
    try:
        payload = safe_load(data)
    except yaml.YAMLError as exc:
        raise ValueError(f"{location} is not valid YAML: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"profiles"}:
        raise ValueError(f"{location} must be a mapping whose only key is 'profiles'")
    profiles = payload["profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError(f"{location}: 'profiles' must map names to profiles")
    source = (
        None
        if given is None
        else ProfilesFile(path=str(location), sha256=hashlib.sha256(data).hexdigest())
    )
    loaded: dict[str, ModelProfile] = {}
    for name, block in profiles.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{location}: a profile name must be a nonblank string, not {name!r}"
            )
        if not isinstance(block, Mapping) or "profiles_file" in block:
            raise ValueError(
                f"{location}: profile {name!r} must be a mapping of 'models' "
                "and, optionally, 'settings'"
            )
        try:
            loaded[name] = ModelProfile.model_validate(
                {**block, "profiles_file": source}
            )
        except ValidationError as exc:
            raise ValueError(f"{location}: profile {name!r}: {exc}") from exc
    return location, loaded


def load_model_profile(profile: str, path: Path | str | None = None) -> ModelProfile:
    """Load one profile's ordered model roster and its settings.

    The profile comes from the model profiles file ``path``, else from the file
    the ``CATMODEL_PROFILES`` environment variable names, else from AIUC's
    packaged example. Every profile in the file is validated."""
    location, profiles = _load_profiles(path)
    if profile not in profiles:
        raise ValueError(
            f"Unknown model profile {profile!r} in {location}. "
            f"Available profiles: {', '.join(profiles)}"
        )
    return profiles[profile]


class Settings(BaseSettings):
    """Provider credentials and the output-token budget of every elicitation call.

    The ``.env`` file is resolved relative to the current working directory at
    instantiation; environment variables take precedence over it.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    XAI_API_KEY: str = ""
    META_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    # Each model asks for the smaller of this and its max_output_tokens in
    # its model profile (NUM-CALL-LIMITS). Collection plans freeze the
    # per-model budgets, so a change needs a new run directory.
    ELICITATION_MAX_TOKENS: int = 100_000

    def provider_api_keys(self) -> dict[str, str | None]:
        """Return provider name to API key, or None when the key is unset or empty."""
        return {
            provider: getattr(self, environment_name) or None
            for provider, environment_name in PROVIDER_API_KEY_ENV.items()
        }


settings = Settings()
