"""Splitting the collected tests over parallel jobs.

``pytest --shard I/N`` (``tests/conftest.py``) runs only part I of N. A test's
part is fixed by its id, so the N parts of one collection are disjoint and
together hold every test, whichever process or machine collects them.
"""

import zlib


def parse_shard(text: str) -> tuple[int, int]:
    """``"2/4"`` as ``(2, 4)``: part 2 of 4, counted from 1."""
    part, separator, parts = text.partition("/")
    if not separator or not _is_count(part) or not _is_count(parts):
        raise ValueError(f"--shard takes I/N, such as 2/4, and got {text!r}")
    if not 1 <= int(part) <= int(parts):
        raise ValueError(f"--shard {text}: I runs from 1 to N")
    return int(part), int(parts)


def shard_of(test_id: str, parts: int) -> int:
    """The part, from 1 to ``parts``, that runs the test with this id."""
    return zlib.crc32(test_id.encode()) % parts + 1


def _is_count(text: str) -> bool:
    return text.isascii() and text.isdigit()
