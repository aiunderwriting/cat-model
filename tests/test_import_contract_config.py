import ast
import re
import tomllib
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _root_import_contracts() -> list[dict[str, Any]]:
    config = tomllib.loads(
        (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return config["tool"]["importlinter"]["contracts"]


def _contract(name: str) -> dict[str, Any]:
    return next(
        contract for contract in _root_import_contracts() if contract["name"] == name
    )


def test_every_module_belongs_to_exactly_one_pipeline_layer() -> None:
    """A module outside every layer is unconstrained, so each is declared once."""
    layers = _contract("Pipeline stages")["layers"]
    declared = [name.strip() for layer in layers for name in re.split("[|:]", layer)]
    assert len(declared) == len(set(declared))
    package = _REPOSITORY_ROOT / "src" / "catmodel"
    for path in package.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        module = ".".join(
            ("catmodel", *path.relative_to(package).with_suffix("").parts)
        )
        owners = [
            name for name in declared if module == name or module.startswith(f"{name}.")
        ]
        assert len(owners) == 1, (module, owners)


def _module_name(path: Path) -> str:
    parts = path.relative_to(_REPOSITORY_ROOT / "src").with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported_module(module: str, package: bool, node: ast.ImportFrom) -> str:
    """The absolute name of the module ``node`` imports from."""
    if not node.level:
        return node.module or ""
    base = module.split(".") if package else module.split(".")[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def test_no_module_imports_another_modules_private_name() -> None:
    """A leading underscore keeps a name inside its module, so a module that
    needs another's name imports a public one; a package may import its own
    private submodules."""
    package = _REPOSITORY_ROOT / "src" / "catmodel"
    modules = {_module_name(path) for path in package.rglob("*.py")}
    found = []
    for path in sorted(package.rglob("*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            origin = _imported_module(module, path.name == "__init__.py", node)
            if origin.split(".")[0] != "catmodel":
                continue
            for alias in node.names:
                name = alias.name
                if not name.startswith("_") or name.startswith("__"):
                    continue
                if f"{origin}.{name}" in modules and (
                    module == origin or module.startswith(f"{origin}.")
                ):
                    continue
                found.append(f"{module}:{node.lineno} imports {origin}.{name}")
    assert found == []


def _packages() -> set[str]:
    package = _REPOSITORY_ROOT / "src" / "catmodel"
    return {
        f"catmodel.{path.name}"
        for path in package.iterdir()
        if (path / "__init__.py").exists()
    }


def test_collection_cannot_import_the_source_bundle_builder() -> None:
    contract = _contract("Collection does not author source bundles")

    assert contract["source_modules"] == [
        "catmodel.cli._capability",
        "catmodel.collection",
    ]
    assert contract["forbidden_modules"] == ["catmodel.evidence.source_inputs"]
    assert contract["allow_indirect_imports"] == "true"


def test_only_the_cli_imports_click() -> None:
    contract = _contract("Only the CLI uses the command-line framework")

    assert contract["forbidden_modules"] == ["click"]
    assert set(contract["source_modules"]) == _packages() - {
        "catmodel.cli",
        "catmodel.llm",
    }


def test_only_collection_reaches_the_provider_transport() -> None:
    """Setup import, execution and export cannot collect: collection sits
    above setup and beside results in the layers, and no package but
    collection may import the transport, directly or through another module."""
    layers = [
        {name.strip() for name in layer.split("|")}
        for layer in _contract("Pipeline stages")["layers"]
    ]
    level = {name: index for index, names in enumerate(layers) for name in names}
    assert level["catmodel.collection"] == level["catmodel.results"]
    assert level["catmodel.collection"] < level["catmodel.setup"]
    contract = _contract("Only collection uses the LLM transport")
    assert set(contract["source_modules"]) == _packages() - {
        "catmodel.cli",
        "catmodel.collection",
        "catmodel.llm",
    }
    assert set(contract["forbidden_modules"]) >= {"catmodel.llm", "litellm"}
    assert contract.get("allow_indirect_imports") not in (True, "true")
