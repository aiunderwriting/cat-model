"""Shared datetime parsing and arithmetic utilities."""

from datetime import UTC, date, datetime


def parse_datetime_utc(value: str | datetime | date | None) -> datetime | None:
    """Parse a datetime-like value and normalize it to UTC.

    Accepts ``datetime``, ``date``, ISO-8601 strings including trailing ``Z``,
    or ``None``; unsupported types raise ``TypeError``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=UTC)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = datetime.fromisoformat(f"{text}T00:00:00+00:00")
    else:
        raise TypeError(f"Unsupported datetime value: {type(value).__name__}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def add_one_year(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("add_one_year() requires a timezone-aware datetime")
    try:
        return dt.replace(year=dt.year + 1)
    except ValueError:
        # Feb 29 -> Feb 28 in non-leap year.
        return dt.replace(year=dt.year + 1, month=2, day=28)
