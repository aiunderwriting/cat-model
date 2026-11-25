import hashlib
import math
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from catmodel.config import (
    EXAMPLE_PROFILES,
    PROFILES_ENV,
    ExecutionSettings,
    ModelProfile,
    ProfileSettings,
    Settings,
    load_model_profile,
    settings,
)
from catmodel.llm.models import ModelSpec
from catmodel.resources import data_path


def _write_yaml(tmp_path, data):
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _spec(provider: str, model_id: str, reasoning_effort: str = "low") -> dict:
    return {
        "provider": provider,
        "model_id": model_id,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": 64_000,
    }


NESTED_PROFILES = {
    "example": {
        "models": [
            _spec("openai", "gpt-6.2-sol"),
            _spec("anthropic", "claude-opus-5-5"),
        ],
        "settings": {
            "request_timeout_seconds": 1800.0,
            "max_concurrency": 200,
            "max_concurrency_by_provider": {"anthropic": 80},
        },
    },
    "dev": {
        "models": [_spec("openai", "gpt-6-luna")],
        "settings": {
            "request_timeout_seconds": 300.0,
            "max_concurrency": 40,
        },
    },
    "test": {
        "models": [
            _spec("openai", "gpt-6-luna", "max"),
            _spec("openrouter", "deepseek/deepseek-v4.1-flash-1017", "xhigh"),
        ],
        "settings": {"max_concurrency_by_provider": {" openrouter ": 20}},
    },
}


def test_profile_loads_native_specs_in_order_and_yaml_lists_become_tuples(tmp_path):
    profile = load_model_profile(
        path=_write_yaml(tmp_path, {"profiles": NESTED_PROFILES}), profile="test"
    )

    assert isinstance(profile, ModelProfile)
    assert isinstance(profile.models, tuple)
    assert all(isinstance(spec, ModelSpec) for spec in profile.models)
    assert [
        (spec.provider, spec.model_id, spec.reasoning_effort) for spec in profile.models
    ] == [
        ("openai", "gpt-6-luna", "max"),
        ("openrouter", "deepseek/deepseek-v4.1-flash-1017", "xhigh"),
    ]
    assert profile.settings.max_concurrency_by_provider == {"openrouter": 20}


def test_unknown_profile_names_the_file_and_its_profiles(tmp_path):
    path = _write_yaml(tmp_path, {"profiles": NESTED_PROFILES})
    with pytest.raises(ValueError, match="Unknown model profile 'staging'") as error:
        load_model_profile(path=path, profile="staging")
    assert f"in {path}. Available profiles: example, dev, test" in str(error.value)


def test_default_profiles_are_the_packaged_example():
    payload = yaml.safe_load(data_path(EXAMPLE_PROFILES).read_text(encoding="utf-8"))
    assert list(payload["profiles"]) == ["example", "test", "dev"]

    example = load_model_profile("example")
    # AIUC's example roster: three frontier LLMs from three developers.
    assert [spec.provider for spec in example.models] == [
        "openai",
        "anthropic",
        "xai",
    ]
    assert example.settings.samples_per_question == 5
    # The packaged example is the default, so no plan records its file.
    assert example.profiles_file is None
    with pytest.raises(ValueError, match="Unknown model profile 'prod'"):
        load_model_profile("prod")


def test_a_supplied_profiles_file_records_its_path_and_digest(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path, {"profiles": NESTED_PROFILES})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    supplied = load_model_profile("dev", path)
    assert [spec.model_id for spec in supplied.models] == ["gpt-6-luna"]
    assert supplied.profiles_file is not None
    assert supplied.profiles_file.model_dump() == {"path": str(path), "sha256": digest}

    # The environment variable names the same file; an explicit path wins
    # over it, and an empty one leaves the packaged example.
    monkeypatch.setenv(PROFILES_ENV, str(path))
    assert load_model_profile("dev") == supplied
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({"profiles": {"dev": NESTED_PROFILES["dev"]}}))
    from_other = load_model_profile("dev", other).profiles_file
    assert from_other is not None and from_other.path == str(other)
    monkeypatch.setenv(PROFILES_ENV, "")
    assert load_model_profile("dev").profiles_file is None


def test_missing_profiles_file_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="model profiles file not found"):
        load_model_profile("example", tmp_path / "nonexistent.yaml")
    monkeypatch.setenv(PROFILES_ENV, str(tmp_path / "unset.yaml"))
    with pytest.raises(ValueError, match=r"not found: .*unset\.yaml"):
        load_model_profile("example")


_MODEL = (
    "      - provider: openai\n"
    "        model_id: model-a\n"
    "        reasoning_effort: low\n"
    "        max_output_tokens: 1000\n"
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "only key is 'profiles'"),
        ("profiles: {}\n", "'profiles' must map names to profiles"),
        ("profiles:\n  - mine\n", "'profiles' must map names to profiles"),
        ("profile:\n  mine:\n    models: []\n", "only key is 'profiles'"),
        (
            "defaults: {}\nprofiles:\n  mine:\n    models: []\n",
            "only key is 'profiles'",
        ),
        ("profiles:\n  1:\n    models: []\n", "nonblank string, not 1"),
        ("profiles:\n  mine: [model-a]\n", "must be a mapping of 'models'"),
        ("profiles:\n  mine:\n    settings: {}\n", r"models\n  Field required"),
        ("profiles:\n  mine:\n    models: []\n", "at least 1 item"),
        (
            f"profiles:\n  mine:\n    models:\n{_MODEL}    setings: {{}}\n",
            r"setings\n  Extra inputs are not permitted",
        ),
        (
            f"profiles:\n  mine:\n    models:\n{_MODEL}    profiles_file: {{}}\n",
            "must be a mapping of 'models'",
        ),
        (
            f"profiles:\n  mine:\n    models:\n{_MODEL}  mine:\n    models:\n{_MODEL}",
            "found duplicate key 'mine'",
        ),
        (
            f"profiles:\n  mine:\n    models:\n{_MODEL}        reasoning_effort: high\n",
            "found duplicate key 'reasoning_effort'",
        ),
        (
            f"profiles:\n  mine:\n    models:\n{_MODEL.replace('openai', 'opnai')}",
            "provider",
        ),
        (
            f"profiles:\n  mine:\n    models:\n{_MODEL}    settings:\n"
            "      samples_per_question: '5'\n",
            "samples_per_question",
        ),
        ("profiles: [unclosed\n", "is not valid YAML"),
    ],
    ids=[
        "empty",
        "no-profiles",
        "profile-list",
        "misspelled-section",
        "extra-section",
        "number-name",
        "profile-list-value",
        "no-models",
        "empty-roster",
        "misspelled-settings",
        "forged-provenance",
        "duplicate-profile",
        "duplicate-field",
        "unknown-provider",
        "string-setting",
        "invalid-yaml",
    ],
)
def test_a_malformed_profiles_file_is_refused_with_its_location(
    tmp_path, text, message
):
    path = tmp_path / "profiles.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        load_model_profile("mine", path)
    assert str(path) in str(error.value)


def test_every_profile_of_a_file_is_validated(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        f"profiles:\n  mine:\n    models:\n{_MODEL}  broken:\n    models: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="profile 'broken'"):
        load_model_profile("mine", path)


def test_profiles_may_share_settings_through_yaml_merge_keys(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "profiles:\n"
        f"  slow: &slow\n    models:\n{_MODEL}"
        "    settings: &settings\n      max_concurrency: 10\n"
        "      request_timeout_seconds: 900\n"
        "  fast:\n    <<: *slow\n    settings:\n      <<: *settings\n"
        "      max_concurrency: 4\n",
        encoding="utf-8",
    )
    fast = load_model_profile("fast", path)
    assert fast.models == load_model_profile("slow", path).models
    assert fast.settings.max_concurrency == 4
    assert fast.settings.request_timeout_seconds == 900.0


def test_profile_settings_defaults_and_null_settings(tmp_path):
    profile = load_model_profile(
        path=_write_yaml(
            tmp_path,
            {
                "profiles": {
                    "example": {
                        "models": [_spec("openai", "model-a")],
                        "settings": None,
                    }
                }
            },
        ),
        profile="example",
    )

    assert profile.settings.samples_per_question == 5
    assert profile.settings.max_concurrency == 20
    assert profile.settings.max_concurrency_by_provider == {}
    assert profile.settings.request_timeout_seconds == 600.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples_per_question", True),
        ("samples_per_question", "5"),
        ("max_concurrency", True),
        ("max_concurrency", "20"),
        ("max_concurrency_by_provider", {"openai": "20"}),
        ("max_concurrency_by_provider", {"openai": True}),
        ("request_timeout_seconds", True),
        ("request_timeout_seconds", "600"),
    ],
)
def test_profile_settings_are_strict(field, value):
    with pytest.raises(ValidationError):
        ProfileSettings(**{field: value})


@pytest.mark.parametrize(
    "payload",
    [
        {"samples_per_question": 0},
        {"samples_per_question": 101},
        {"max_concurrency": 0},
        {"max_concurrency": 1001},
        {"max_concurrency_by_provider": {"openai": 0}},
        {"max_concurrency_by_provider": {"openai": 1001}},
        {"max_concurrency_by_provider": None},
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": -1},
        {"request_timeout_seconds": math.inf},
        {"request_timeout_seconds": -math.inf},
        {"request_timeout_seconds": math.nan},
        {"unknown_setting": 1},
    ],
)
def test_profile_settings_reject_bounds_and_unknown_fields(payload):
    with pytest.raises(ValidationError):
        ProfileSettings(**payload)


def test_integer_request_timeout_is_accepted():
    assert ProfileSettings(request_timeout_seconds=600).request_timeout_seconds == 600.0


def test_duplicate_model_ids_and_unrepresented_provider_caps_are_rejected():
    with pytest.raises(ValidationError, match="model_id values must be unique"):
        ModelProfile(
            models=(
                ModelSpec(
                    provider="openai",
                    model_id="same",
                    reasoning_effort="low",
                    max_output_tokens=1000,
                ),
                ModelSpec(
                    provider="google",
                    model_id="same",
                    reasoning_effort="low",
                    max_output_tokens=1000,
                ),
            )
        )
    with pytest.raises(ValidationError, match="providers absent from the profile"):
        ModelProfile(
            models=(
                ModelSpec(
                    provider="openai",
                    model_id="model-a",
                    reasoning_effort="low",
                    max_output_tokens=1000,
                ),
            ),
            settings=ProfileSettings(max_concurrency_by_provider={"openrouter": 20}),
        )


def test_execution_settings_resolve_effective_per_model_limits_and_subset_caps():
    profile = ModelProfile(
        models=(
            ModelSpec(
                provider="openai",
                model_id="openai-a",
                reasoning_effort="low",
                max_output_tokens=128_000,
            ),
            ModelSpec(
                provider="openrouter",
                model_id="router-a",
                reasoning_effort="high",
                max_output_tokens=1000,
            ),
            ModelSpec(
                provider="openrouter",
                model_id="router-b",
                reasoning_effort="max",
                max_output_tokens=2000,
            ),
        ),
        settings=ProfileSettings(
            samples_per_question=7,
            max_concurrency=40,
            max_concurrency_by_provider={"openrouter": 9},
            request_timeout_seconds=12,
        ),
    )

    execution = profile.execution_settings(
        (profile.models[0], profile.models[1]), max_output_tokens=1234
    )
    assert execution.model_dump(mode="json") == {
        "provider_message_layout": "system_then_user",
        "samples_per_question": 7,
        "max_output_tokens_by_model": {"openai-a": 1234, "router-a": 1000},
        "request_timeout_seconds": 12.0,
        "concurrency_by_model": {"openai-a": 40, "router-a": 9},
    }
    both_router = profile.execution_settings(profile.models[1:], max_output_tokens=1234)
    assert both_router.concurrency_by_model == {"router-a": 9, "router-b": 9}


def test_provider_cap_for_absent_selected_subset_is_allowed(tmp_path):
    profile = load_model_profile(
        path=_write_yaml(tmp_path, {"profiles": NESTED_PROFILES}), profile="test"
    )

    execution = profile.execution_settings((profile.models[0],), max_output_tokens=1234)

    assert execution.concurrency_by_model == {"gpt-6-luna": 20}


def test_execution_settings_reject_model_outside_profile_and_token_bounds():
    profile = ModelProfile(
        models=(
            ModelSpec(
                provider="openai",
                model_id="model-a",
                reasoning_effort="low",
                max_output_tokens=128_000,
            ),
        )
    )
    outside = ModelSpec(
        provider="openai",
        model_id="model-b",
        reasoning_effort="low",
        max_output_tokens=128_000,
    )

    with pytest.raises(ValueError, match="not in the profile"):
        profile.execution_settings((outside,), max_output_tokens=100)
    for value in (0, 100_001, True, "100"):
        with pytest.raises(ValueError, match="output budget must be an integer"):
            profile.execution_settings(
                profile.models, max_output_tokens=cast(int, value)
            )
    for ceiling in (0, True, "100"):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate(
                {
                    "provider": "openai",
                    "model_id": "model-a",
                    "reasoning_effort": "low",
                    "max_output_tokens": ceiling,
                }
            )


@pytest.mark.parametrize("ceiling", ["", "        max_output_tokens:\n"])
def test_a_model_without_an_output_ceiling_in_any_profile_refuses_the_file(
    tmp_path, ceiling
):
    # The profile a command selects is complete; the fault is in another one.
    path = tmp_path / "profiles.yaml"
    path.write_text(
        f"profiles:\n  mine:\n    models:\n{_MODEL}"
        f"  unused:\n    models:\n{_MODEL}"
        "      - provider: anthropic\n        model_id: bare\n"
        f"        reasoning_effort: low\n{ceiling}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="profile 'unused'") as error:
        load_model_profile("mine", path)
    assert str(path) in str(error.value)
    assert "model 'bare' needs a max_output_tokens ceiling" in str(error.value)


def test_execution_settings_name_the_same_models_in_both_limits():
    with pytest.raises(ValidationError, match="name the same models"):
        ExecutionSettings(
            samples_per_question=1,
            max_output_tokens_by_model={"other": 100},
            request_timeout_seconds=1,
            concurrency_by_model={"model-a": 1},
        )


def test_execution_settings_is_frozen_and_forbids_extra_fields():
    execution = ExecutionSettings(
        samples_per_question=1,
        max_output_tokens_by_model={"model-a": 100},
        request_timeout_seconds=1,
        concurrency_by_model={"model-a": 1},
    )

    with pytest.raises(ValidationError, match="frozen"):
        execution.request_timeout_seconds = 101  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError, match="provider_message_layout"):
        ExecutionSettings.model_validate(
            {**execution.model_dump(), "provider_message_layout": "system_with_user"}
        )
    with pytest.raises(ValidationError):
        ExecutionSettings.model_validate(
            {
                "samples_per_question": 1,
                "max_output_tokens_by_model": {"model-a": 100},
                "request_timeout_seconds": 1,
                "concurrency_by_model": {"model-a": 1},
                "extra": True,
            }
        )


def test_packaged_profiles_load():
    payload = yaml.safe_load(data_path(EXAMPLE_PROFILES).read_text(encoding="utf-8"))
    for profile_name in payload["profiles"]:
        profile = load_model_profile(profile=profile_name)
        assert profile.models
        execution = profile.execution_settings(
            profile.models, max_output_tokens=settings.ELICITATION_MAX_TOKENS
        )
        assert set(execution.concurrency_by_model) == {
            spec.model_id for spec in profile.models
        }
        # Every roster model records a ceiling at or above the budget, so each
        # asks for the whole budget.
        assert set(execution.max_output_tokens_by_model.values()) == {
            settings.ELICITATION_MAX_TOKENS
        }


def test_settings_use_environment_over_cwd_dotenv_and_map_api_keys(
    tmp_path, monkeypatch
):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\n"
        "ANTHROPIC_API_KEY=anthropic-dotenv\n"
        "ELICITATION_MAX_TOKENS=123\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ELICITATION_MAX_TOKENS", raising=False)

    configured = Settings()

    assert configured.OPENAI_API_KEY == "from-environment"
    assert configured.ANTHROPIC_API_KEY == "anthropic-dotenv"
    assert configured.ELICITATION_MAX_TOKENS == 123
    assert configured.provider_api_keys() == {
        "openai": "from-environment",
        "anthropic": "anthropic-dotenv",
        "google": None,
        "xai": None,
        "meta": None,
        "openrouter": None,
    }
    assert set(Settings.model_fields) == {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "META_API_KEY",
        "OPENROUTER_API_KEY",
        "ELICITATION_MAX_TOKENS",
    }
