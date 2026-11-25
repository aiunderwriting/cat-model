"""One fields set for every instance of a contract given the same fields.

Pydantic gives each model instance its own set of the field names it was
given. A run setup holds hundreds of thousands of small frozen models, and in
a loaded setup those sets were about a third of the models' memory, although
only a few dozen distinct ones exist. No code here reads them, and a frozen
model never changes the fields it was given, so its instances share one.

The answer contracts (``contracts.elicitation``, ``contracts.contextual_risk``)
do not inherit this. A response model's base class is part of the parser
identity every stored call records (``frozen_calls.parser_protocol``), so
changing it would give calls collected afterwards another identity for the
same parsing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

_SETS: dict[tuple[type, frozenset[str]], set[str]] = {}
"""The shared set of each class and combination of given fields."""


class SharedFieldsSet(BaseModel):
    """A frozen model whose instances share their fields set with every other
    instance of the class that was given the same fields.

    Validation, ``model_construct`` and revalidation all share it. A
    ``model_copy`` takes its own copy of the set before any update, and
    assignment is refused, so nothing changes a shared set. A subclass must
    stay frozen: assignment to a model that is not would add the field to
    the set of every instance sharing it.
    """

    model_config = ConfigDict(frozen=True)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if not cls.model_config.get("frozen"):
            raise TypeError(
                f"{cls.__name__} shares its fields set, so it must be frozen"
            )

    def model_post_init(self, context: Any, /) -> None:
        given = self.__pydantic_fields_set__
        shared = _SETS.setdefault((type(self), frozenset(given)), given)
        if shared is not given:
            object.__setattr__(self, "__pydantic_fields_set__", shared)
