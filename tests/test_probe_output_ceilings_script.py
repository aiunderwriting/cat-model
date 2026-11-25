"""The ceiling probe asks each model once at its ceiling and reports failures."""

import asyncio

import pytest

from catmodel.llm.models import ModelSpec, ReasoningEffort
from catmodel.llm.transport import ProviderResponse
from scripts.probe_output_ceilings import main, probe, select_models


class _Provider:
    provider_name = "fixture"
    reasoning_effort: ReasoningEffort = "low"

    def __init__(self, model_id: str, limit: int) -> None:
        self.model_id = model_id
        self.limit = limit
        self.requested: list[int] = []

    async def complete(self, *, max_tokens: int, **_kwargs) -> ProviderResponse:
        self.requested.append(max_tokens)
        if max_tokens > self.limit:
            raise ValueError(f"max_tokens: {max_tokens} > {self.limit}")
        return ProviderResponse(text="OK", model_id=self.model_id, output_tokens=2)


def _spec(model_id: str, ceiling: int) -> ModelSpec:
    return ModelSpec(
        provider="openai",
        model_id=model_id,
        reasoning_effort="low",
        max_output_tokens=ceiling,
    )


def test_probe_reports_each_ceiling_a_provider_rejects_or_cannot_check():
    fits, too_high = _Provider("fits", 128_000), _Provider("too-high", 64_000)
    specs = (
        _spec("fits", 128_000),
        _spec("too-high", 100_000),
        _spec("no-key", 100_000),
    )
    accepted, failed = asyncio.run(probe(specs, [fits, too_high], 60))
    assert fits.requested == [128_000] and too_high.requested == [100_000]
    assert accepted == [
        "openai/fits: accepted max_output_tokens=128000 (2 tokens used)"
    ]
    assert failed == [
        "openai/too-high: ValueError: max_tokens: 100000 > 64000",
        "openai/no-key: no API key",
    ]


def test_select_models_filters_one_profile_in_roster_order():
    specs = (_spec("first", 100), _spec("second", 200), _spec("third", 300))

    assert select_models(specs, ("third", "first"), "fixture") == (
        specs[0],
        specs[2],
    )
    assert select_models(specs, (), "fixture") == specs
    with pytest.raises(ValueError, match="profile 'fixture'") as error:
        select_models(specs, ("outside",), "fixture")
    assert "valid profile ids: first, second, third" in str(error.value)


def test_unknown_model_is_an_argparse_error_before_provider_setup(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--profile", "test", "--model", "not-in-profile"])

    assert error.value.code == 2
    assert "valid profile ids:" in capsys.readouterr().err


def test_profiles_file_supplies_the_probed_roster(tmp_path, capsys):
    profiles = tmp_path / "my-profiles.yaml"
    profiles.write_text(
        "profiles:\n"
        "  mine:\n"
        "    models:\n"
        "      - provider: openai\n"
        "        model_id: my-model\n"
        "        reasoning_effort: low\n"
        "        max_output_tokens: 1000\n",
        encoding="utf-8",
    )
    arguments = ["--profile", "mine", "--profiles", str(profiles)]
    with pytest.raises(SystemExit) as error:
        main([*arguments, "--model", "not-in-profile"])
    assert error.value.code == 2
    assert "valid profile ids: my-model" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        main(["--profile", "staging", "--profiles", str(profiles)])
    assert error.value.code == 2
    assert "Unknown model profile 'staging'" in capsys.readouterr().err
