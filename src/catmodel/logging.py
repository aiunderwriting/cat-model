"""Structured logging: JSON on non-TTY stderr and colored console output on TTY.

A warning repeated with identical fields is written once; its repeats are
counted and summarized, one line per repeated warning, when the process exits
(``RepeatedWarnings``).
"""

import atexit
import sys
from typing import Any, TextIO

import structlog

INFO = 20


class _NamedWriteLogger(structlog.WriteLogger):
    name: str | None

    def __init__(self, file: TextIO, name: str | None) -> None:
        super().__init__(file)
        self.name = name


class _NamedWriteLoggerFactory:
    def __call__(self, *args: Any) -> _NamedWriteLogger:
        name = str(args[0]) if args else None
        return _NamedWriteLogger(sys.stderr, name)


def _add_logger_name(
    logger: Any,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    if name := getattr(logger, "name", None):
        event_dict["logger"] = name
    return event_dict


class RepeatedWarnings:
    """A processor that writes a warning's first occurrence, drops its exact
    repeats (same logger, event and fields) and counts them.

    ``summarise`` logs one ``repeated warnings suppressed`` line per warning
    that repeated, with its fields, its event as ``repeated_event`` and its
    repeat count as ``repeats``. At most ``limit`` distinct warnings are
    tracked; beyond that, new ones pass unchanged."""

    def __init__(self, limit: int = 10_000) -> None:
        self.limit = limit
        self.repeats: dict[tuple[str, ...], int] = {}
        self._first: dict[tuple[str, ...], dict[str, Any]] = {}

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: structlog.types.EventDict,
    ) -> structlog.types.EventDict:
        if method_name != "warning" or event_dict.get("summarises_repeats"):
            return event_dict
        key = (
            str(getattr(logger, "name", None)),
            *sorted(f"{name}={value!r}" for name, value in event_dict.items()),
        )
        if key in self.repeats:
            self.repeats[key] += 1
            raise structlog.DropEvent
        if len(self.repeats) < self.limit:
            self.repeats[key] = 0
            self._first[key] = dict(event_dict)
        return event_dict

    def summarise(self) -> None:
        """Log each repeated warning once more with its repeat count, and
        start counting afresh."""
        repeated = [(key, count) for key, count in self.repeats.items() if count]
        first = self._first
        self.repeats, self._first = {}, {}
        for key, count in repeated:
            fields = dict(first[key])
            event = fields.pop("event", None)
            fields.update(repeated_event=event, repeats=count, summarises_repeats=True)
            named = () if key[0] == "None" else (key[0],)
            structlog.get_logger(*named).warning(
                "repeated warnings suppressed", **fields
            )


_repeated = RepeatedWarnings()
atexit.register(_repeated.summarise)


def configure_logging(level: int = INFO) -> None:
    """Call once at application startup."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _repeated,
        _add_logger_name,
        structlog.processors.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if sys.stderr.isatty():
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        logger_factory=_NamedWriteLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=False,
    )
