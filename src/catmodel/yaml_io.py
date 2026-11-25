"""YAML helpers using LibYAML acceleration when available.

Every YAML file the model reads goes through ``safe_load``, which refuses a
key repeated within one mapping instead of keeping its last occurrence.
"""

from collections.abc import Hashable
from typing import Any

import yaml
from yaml.constructor import ConstructorError

_SafeLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_SafeDumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_MERGE_TAG = "tag:yaml.org,2002:merge"


class _UniqueKeyLoader(_SafeLoader):
    """Safe YAML that refuses a key repeated within one mapping.

    A merge key (``<<``) is not a key of its mapping: a mapping may merge
    several others and override what it merges.
    """

    def construct_mapping(self, node, deep=False):
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                continue
            if key in seen:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def safe_load(stream: Any) -> Any:
    """Parse one YAML document, raising ``yaml.YAMLError`` when it is not
    valid YAML or repeats a key within one mapping."""
    return yaml.load(stream, Loader=_UniqueKeyLoader)


def safe_dump(data: Any, **kwargs: Any) -> str:
    return yaml.dump(data, Dumper=_SafeDumper, **kwargs)
