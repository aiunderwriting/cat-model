"""Parser treatment identity follows the code parsing reaches; replay catches the rest."""

import asyncio
import importlib.util
import json
import sys
import types
from functools import partial
from typing import cast

import pytest

from catmodel import frozen_calls
from catmodel.collection.frozen_call_collection import (
    ProviderFailed,
    collect_frozen_call,
)
from catmodel.config import ExecutionSettings
from catmodel.contracts.contextual_risk import ContextualChainAnswer
from catmodel.elicitation.contextual_frequency import parse_chain_answer
from catmodel.frozen_calls import (
    AcquisitionTreatment,
    parser_protocol,
    replay_call,
    validate_acquisition_capture,
    validate_acquisition_closure,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.llm.transport import LLMProvider, ProviderError, ProviderResponse
from test_support.version_markers import unstated_markers

# The response model's module, a helper module and the parser's module. The
# parser reaches ``EmptyAnswer``, ``_within_limit`` and ``LIMIT`` in its own
# module and the nested ``Detail`` model; ``Question``, which only a type hint
# names, ``report`` and ``Summary`` it does not.
SOURCES = {
    "answer": """from pydantic import BaseModel, field_validator
class Detail(BaseModel):
    note: str = ""
    @field_validator("note")
    @classmethod
    def short(cls, note):
        if len(note) > 20:
            raise ValueError("note too long")
        return note
class Answer(BaseModel):
    value: int
    detail: Detail = Detail()
    @field_validator("value")
    @classmethod
    def bounded(cls, value):
        if value > 100:
            raise ValueError("value above 100")
        return value
class Summary(BaseModel):
    text: str
""",
    "checks": """def validate(value):
    if value <= 0:
        raise ValueError("positive value required")
""",
    "parser": """from {package} import checks
from {package}.answer import Answer
LIMIT = 1000
class EmptyAnswer(ValueError):
    pass
class Question:
    def render(self):
        return "What is the value?"
def _within_limit(value, question: Question | None = None):
    if value > LIMIT:
        raise ValueError("value above the limit")
def parse(text, *, checked):
    if not text:
        raise EmptyAnswer("empty answer")
    answer = Answer.model_validate_json(text)
    _within_limit(answer.value)
    if checked:
        checks.validate(answer.value)
    return answer
def report(answer):
    return "The answer is " + str(answer.value) + "."
""",
}

REQUEST = {
    "call_id": "parser-test",
    "respondent": {
        "respondent_id": "fixture",
        "model_id": "fixture",
        "sample_index": 0,
    },
    "prompt_sha256": "a" * 64,
    "system_prompt": "fixture",
    "user_prompt": "fixture",
}
CONFIG = ExecutionSettings(
    samples_per_question=1,
    max_output_tokens_by_model={"fixture": 100},
    request_timeout_seconds=60,
    concurrency_by_model={"fixture": 1},
)


class Provider:
    provider_name = "fixture"
    model_id = "fixture"
    reasoning_effort = "low"

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else '{"value":1}'
        if isinstance(outcome, Exception):
            raise outcome
        return ProviderResponse(
            text=outcome, model_id=self.model_id, provider_name=self.provider_name
        )


class Fixture:
    """A parser package on disk whose modules each test edits, and one
    frozen call collected with it."""

    def __init__(self, tmp_path, monkeypatch, provider):
        self.package = "parser_fixture_" + tmp_path.name.replace("-", "_")
        self.root = tmp_path / self.package
        self.root.mkdir()
        self.run = tmp_path / "run"
        self.provider = provider
        package = types.ModuleType(self.package)
        package.__path__ = [str(self.root)]
        monkeypatch.setitem(sys.modules, self.package, package)
        for name, source in SOURCES.items():
            path = self.root / f"{name}.py"
            path.write_text(source.format(package=self.package))
            spec = importlib.util.spec_from_file_location(
                f"{self.package}.{name}", path
            )
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            monkeypatch.setitem(sys.modules, spec.name, module)
            monkeypatch.setattr(package, name, module, raising=False)
        self.load()

    def load(self):
        """Run every module from its file again, as a new process would
        import the code on disk."""
        for name in SOURCES:
            module = sys.modules[f"{self.package}.{name}"]
            path = self.root / f"{name}.py"
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        self.parse = partial(sys.modules[f"{self.package}.parser"].parse, checked=True)
        self.answer = sys.modules[f"{self.package}.answer"].Answer

    def edit(self, name, old, new):
        path = self.root / f"{name}.py"
        source = path.read_text()
        assert old in source
        path.write_text(source.replace(old, new))
        self.load()

    def protocol(self):
        return parser_protocol(self.parse, self.answer)

    def collect(self, request=REQUEST):
        return asyncio.run(
            collect_frozen_call(
                run_directory=self.run,
                request=request,
                provider=cast(LLMProvider, self.provider),
                config=CONFIG,
                parse_answer=self.parse,
                response_model=self.answer,
            )
        )

    def files(self):
        return {
            path.relative_to(self.run).as_posix(): path.read_bytes()
            for path in self.run.rglob("*.json")
        }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return Fixture(tmp_path, monkeypatch, Provider())


def test_a_code_change_in_the_parser_module_blocks_reuse_of_a_still_valid_answer(
    fixture,
):
    fixture.collect()
    original = fixture.protocol()
    fixture.edit("parser", "if checked:", "if checked is True:")
    changed = fixture.protocol()
    assert changed.callable_name == original.callable_name
    assert changed.response_schema_sha256 == original.response_schema_sha256
    assert changed.source_definitions != original.source_definitions
    with pytest.raises(ValueError, match="frozen artifact differs"):
        fixture.collect()
    assert fixture.provider.calls == 1
    treatment = validate_acquisition_capture(REQUEST, fixture.files())
    assert isinstance(treatment, AcquisitionTreatment)
    assert treatment.parser_protocol_sha256 == canonical_json_sha256(
        original.model_dump(mode="json")
    )


def test_a_validator_change_in_the_response_model_module_blocks_reuse(fixture):
    fixture.collect()
    original = fixture.protocol()
    fixture.edit("answer", "value > 100", "value > 50")
    changed = fixture.protocol()
    # The schema cannot show a validator; the source digest does.
    assert changed.response_schema_sha256 == original.response_schema_sha256
    assert changed.source_definitions != original.source_definitions
    with pytest.raises(ValueError, match="frozen artifact differs"):
        fixture.collect()
    assert fixture.provider.calls == 1


def test_docstrings_comments_formatting_and_partial_adapter_do_not_change_treatment(
    fixture,
):
    fixture.collect()
    original = fixture.protocol()
    for name in ("parser", "answer"):
        path = fixture.root / f"{name}.py"
        source = path.read_text()
        path.write_text(
            f'"""The {name} module, documented."""\n# Cosmetic comment.\n'
            + source.replace("    ", "  ")
        )
    fixture.edit(
        "parser",
        "def parse(text, *, checked):\n",
        'def parse(text, *, checked):\n  """Parse one answer."""\n  # Validate.\n',
    )
    # Documenting an empty class replaces its ``pass``.
    fixture.edit(
        "parser",
        "class EmptyAnswer(ValueError):\n  pass\n",
        'class EmptyAnswer(ValueError):\n  """An empty response."""\n',
    )
    fixture.edit(
        "answer",
        "  value: int\n",
        '  value: int\n  """An attribute docstring the schema does not show."""\n',
    )
    assert fixture.protocol() == original
    assert fixture.collect().value == 1
    assert fixture.provider.calls == 1
    # A docstring the response schema shows is part of the schema instead.
    fixture.edit(
        "answer",
        "class Answer(BaseModel):\n",
        'class Answer(BaseModel):\n  """Shown."""\n',
    )
    shown = fixture.protocol()
    assert shown.source_definitions == original.source_definitions
    assert shown.response_schema_sha256 != original.response_schema_sha256


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("parser", "LIMIT = 1000", "LIMIT = 500"),
        ("parser", "if value > LIMIT:", "if value >= LIMIT:"),
        ("answer", "len(note) > 20", "len(note) > 10"),
    ],
)
def test_a_change_to_a_definition_parsing_reaches_changes_the_treatment(
    fixture, module, old, new
):
    original = fixture.protocol()
    fixture.edit(module, old, new)
    changed = fixture.protocol()
    assert changed.response_schema_sha256 == original.response_schema_sha256
    assert changed.source_definitions != original.source_definitions


def test_the_rest_of_the_defining_modules_does_not_change_the_treatment(fixture):
    fixture.collect()
    original = fixture.protocol()
    fixture.edit("parser", '"The answer is "', '"Its answer is "')
    fixture.edit("parser", '"What is the value?"', '"Which value is it?"')
    fixture.edit("answer", "    text: str\n", "    text: str\n    words: int = 0\n")
    # Moving a definition does not change what parsing runs.
    path = fixture.root / "parser.py"
    source = path.read_text()
    limit = "LIMIT = 1000\n"
    path.write_text(source.replace(limit, "") + limit)
    fixture.load()
    assert fixture.protocol() == original
    assert fixture.collect().value == 1
    assert fixture.provider.calls == 1


def test_a_helper_change_is_caught_by_replay_rather_than_the_treatment(fixture):
    fixture.collect()
    original = fixture.protocol()
    # A helper change that every stored response still satisfies: the stored
    # answer stands and nothing is sent again.
    fixture.edit("checks", "value <= 0", "value < 0")
    assert fixture.protocol() == original
    assert fixture.collect().value == 1
    # One that changes how a stored response parses: resume replays the
    # stored attempt before sending anything and refuses, as validation does.
    fixture.edit("checks", "value < 0", "value <= 1")
    assert fixture.protocol() == original
    with pytest.raises(ValueError, match="parse success differs"):
        fixture.collect()
    with pytest.raises(ValueError, match="parse success differs"):
        replay_call(request=REQUEST, files=fixture.files(), parse_answer=fixture.parse)
    assert fixture.provider.calls == 1


@pytest.fixture
def new_process():
    """Forgets what this process cached about its runtime and parsers, as a
    resume in a new process starts without; forgets again afterwards."""

    def forget():
        frozen_calls._runtime_versions.cache_clear()
        frozen_calls._parser_protocol.cache_clear()

    yield forget
    forget()


def test_a_runtime_upgrade_does_not_block_resume_and_each_attempt_records_it(
    tmp_path, monkeypatch, new_process
):
    provider = Provider("not json", ProviderError("502", transient=True))
    fixture = Fixture(tmp_path, monkeypatch, provider)
    with pytest.raises(ProviderFailed, match="502"):
        fixture.collect()
    real = frozen_calls.version
    installed = real("pydantic")
    monkeypatch.setattr(
        frozen_calls,
        "version",
        lambda name: "2.99.0" if name == "pydantic" else real(name),
    )
    new_process()
    assert fixture.collect().value == 1
    assert provider.calls == 3
    rows = [
        json.loads(data)
        for name, data in sorted(fixture.files().items())
        if "/attempt-" in name
    ]
    assert [row["runtime_versions"]["pydantic"] for row in rows] == [
        installed,
        installed,
        "2.99.0",
    ]
    assert rows[2]["runtime_versions"].keys() == {
        "python",
        "pydantic",
        "numpy",
        "scipy",
    }


def test_a_package_parser_is_identified_by_the_modules_defining_its_code():
    """The parser's module and those of the response schema's models, not the
    dozens of package modules they import."""
    protocol = parser_protocol(parse_chain_answer, ContextualChainAnswer)
    assert protocol.source_definitions.keys() == {
        "catmodel.elicitation.contextual_frequency",
        "catmodel.contracts.contextual_risk",
        "catmodel.contracts.elicitation",
    }


def test_response_schema_and_exact_request_conditioning_are_bound(fixture):
    fixture.collect()
    original = fixture.protocol()
    fixture.edit("answer", "value: int", "value: float")
    changed = fixture.protocol()
    assert changed.response_schema_sha256 != original.response_schema_sha256
    with pytest.raises(ValueError, match="bound question and conditioning"):
        validate_acquisition_capture(
            {**REQUEST, "user_prompt": "different conditioning"}, fixture.files()
        )


def test_an_acquisition_record_that_binds_only_a_parser_digest_is_refused(fixture):
    """A record without the request digest and the captured parser protocol
    proves neither what was asked nor how the answer was parsed."""
    fixture.collect()
    capture = fixture.files()
    name = next(name for name in capture if name.startswith("acquisition/"))
    old = json.loads(capture[name])
    protocol_sha = old.pop("parser_protocol_sha256")
    (fixture.run / "parser-protocols" / (protocol_sha + ".json")).unlink()
    old.pop("request_sha256")
    old["parser_sha256"] = "a" * 64
    (fixture.run / name).write_text(json.dumps(old))
    unbound = "does not bind its request and parser protocol"
    with pytest.raises(ValueError, match=unbound):
        validate_acquisition_capture(REQUEST, fixture.files())
    with pytest.raises(ValueError, match=unbound):
        validate_acquisition_closure([REQUEST], fixture.files())
    # Nor is the stored call resumed under a current treatment.
    original_attempts = {
        name: data for name, data in fixture.files().items() if "/attempt-" in name
    }
    with pytest.raises(ValueError, match="frozen artifact differs"):
        fixture.collect()
    assert fixture.provider.calls == 1
    assert original_attempts == {
        name: data for name, data in fixture.files().items() if "/attempt-" in name
    }


@pytest.mark.parametrize(
    "mutation", ["missing", "changed", "unreferenced", "missing_acquisition"]
)
def test_shared_parser_protocol_closure_rejects_corruption(fixture, mutation):
    fixture.collect()
    capture = fixture.files()
    protocol_name = next(
        name for name in capture if name.startswith("parser-protocols/")
    )
    acquisition_name = next(name for name in capture if name.startswith("acquisition/"))
    if mutation == "missing":
        capture.pop(protocol_name)
    elif mutation == "changed":
        record = json.loads(capture[protocol_name])
        record["callable_name"] = "wrong.parser"
        capture[protocol_name] = json.dumps(record).encode()
    elif mutation == "missing_acquisition":
        capture.pop(acquisition_name)
    else:
        capture["parser-protocols/" + "a" * 64 + ".json"] = capture[protocol_name]
    with pytest.raises(ValueError, match=r"parser protocol|acquisition files"):
        validate_acquisition_closure([REQUEST], capture)


def test_a_captured_parser_protocol_must_state_its_schema(fixture):
    fixture.collect()
    capture = fixture.files()
    protocol_name = next(
        name for name in capture if name.startswith("parser-protocols/")
    )
    acquisition_name = next(name for name in capture if name.startswith("acquisition/"))
    for record in unstated_markers(json.loads(capture[protocol_name])).values():
        # The record under its own digest, and the acquisition naming it.
        digest = canonical_json_sha256(record)
        acquisition = json.loads(capture[acquisition_name])
        acquisition["parser_protocol_sha256"] = digest
        changed = {
            name: data for name, data in capture.items() if name != protocol_name
        } | {
            f"parser-protocols/{digest}.json": json.dumps(record).encode(),
            acquisition_name: json.dumps(acquisition).encode(),
        }
        with pytest.raises(ValueError, match="parser protocol schema version"):
            validate_acquisition_closure([REQUEST], changed)
    assert len(validate_acquisition_closure([REQUEST], capture)) == 1


def test_identical_parser_definition_is_stored_once_for_multiple_requests(fixture):
    fixture.collect()
    second = {**REQUEST, "call_id": "second-request"}
    fixture.collect(second)
    capture = fixture.files()
    assert len([name for name in capture if name.startswith("parser-protocols/")]) == 1
    treatments = validate_acquisition_closure([REQUEST, second], capture)
    assert len(treatments) == 2
    first, second = treatments
    assert isinstance(first, AcquisitionTreatment) and isinstance(
        second, AcquisitionTreatment
    )
    assert first.parser_protocol_sha256 == second.parser_protocol_sha256
