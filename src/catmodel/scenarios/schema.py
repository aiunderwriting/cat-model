"""Base model and field types shared by the scenario library's YAML schemas.

Every scenario-library record forbids unknown keys, so a misspelled or unknown
field fails loading instead of being ignored. Authored text is either ``Prose``,
whose whitespace runs collapse to single spaces because it is rendered inline,
or ``Text``, which keeps its paragraphs; both refuse blank values.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _prose(value: str) -> str:
    collapsed = " ".join(value.split())
    if not collapsed:
        raise ValueError("text must not be blank")
    return collapsed


def _text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("text must not be blank")
    return stripped


type Prose = Annotated[str, AfterValidator(_prose)]
"""Nonblank authored text with its whitespace collapsed."""

type Text = Annotated[str, AfterValidator(_text)]
"""Nonblank authored text with its paragraphs kept."""

type Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
"""A lowercase snake-case identifier."""
