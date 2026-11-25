"""A warning repeated with identical fields is written once and summarized."""

import json

import structlog

from catmodel import logging as catmodel_logging


def _lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_identical_warnings_are_written_once_then_counted(capsys, monkeypatch):
    repeats = catmodel_logging.RepeatedWarnings()
    monkeypatch.setattr(catmodel_logging, "_repeated", repeats)
    catmodel_logging.configure_logging()
    logger = structlog.get_logger("catmodel.example")
    for _ in range(5):
        logger.warning("tail at the guardrail", gpd_shape=1.0, severity_cap=1e14)
    logger.warning("tail at the guardrail", gpd_shape=1.2, severity_cap=1e14)
    for index in range(3):
        logger.info("progress", step=index)
        logger.info("progress", step=index)
    lines = _lines(capsys.readouterr().err)
    warnings = [row for row in lines if row["level"] == "warning"]
    # The first of each distinct warning; information is never suppressed.
    assert [row["gpd_shape"] for row in warnings] == [1.0, 1.2]
    assert sum(row["event"] == "progress" for row in lines) == 6

    repeats.summarise()
    summary = _lines(capsys.readouterr().err)
    assert len(summary) == 1
    assert summary[0] == {
        **summary[0],
        "event": "repeated warnings suppressed",
        "repeated_event": "tail at the guardrail",
        "repeats": 4,
        "gpd_shape": 1.0,
        "severity_cap": 1e14,
        "logger": "catmodel.example",
    }
    # Counting starts afresh: the next occurrence is written again.
    logger.warning("tail at the guardrail", gpd_shape=1.0, severity_cap=1e14)
    assert len(_lines(capsys.readouterr().err)) == 1


def test_warnings_beyond_the_tracking_limit_pass_unchanged(capsys, monkeypatch):
    monkeypatch.setattr(
        catmodel_logging, "_repeated", catmodel_logging.RepeatedWarnings(limit=1)
    )
    catmodel_logging.configure_logging()
    logger = structlog.get_logger("catmodel.example")
    for _ in range(2):
        logger.warning("first")
        logger.warning("second")
    events = [row["event"] for row in _lines(capsys.readouterr().err)]
    assert events == ["first", "second", "second"]
