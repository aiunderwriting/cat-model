"""Runtime resource path resolution for source checkouts and wheels."""

from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path

_RESOURCE_STACK = ExitStack()
_PACKAGE_RESOURCE_ROOT = "_resources"
_RESOURCE_DIRS = {
    "config",
    "contracts",
    "data",
    "registries",
    "scenarios",
}


def source_checkout_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (
        candidate / "src" / "catmodel"
    ).is_dir():
        return candidate
    return None


def resource_path(root: str, *parts: str) -> Path:
    """Return a path under a runtime resource tree.

    Source checkouts use repository directories; wheels use bundled
    ``catmodel/_resources`` resources.
    """
    if root not in _RESOURCE_DIRS:
        raise ValueError(f"Unknown resource root: {root!r}")

    source_root = source_checkout_root()
    if source_root is not None:
        return source_root.joinpath(root, *parts)

    resource = files("catmodel").joinpath(_PACKAGE_RESOURCE_ROOT).joinpath(root)
    for part in parts:
        resource = resource.joinpath(part)
    if not resource.is_file() and not resource.is_dir():
        missing = "/".join((root, *parts))
        raise FileNotFoundError(f"Packaged catmodel resource not found: {missing}")
    if isinstance(resource, Path):
        return resource
    return _RESOURCE_STACK.enter_context(as_file(resource))


def config_path(*parts: str) -> Path:
    return resource_path("config", *parts)


def data_path(*parts: str) -> Path:
    return resource_path("data", *parts)


def scenarios_path(*parts: str) -> Path:
    return resource_path("scenarios", *parts)
