"""Fail if the package imports anything outside the standard library.

The project advertises zero runtime dependencies; this keeps that honest.

Instead of comparing names against a list, the check resolves every imported
module and asks whether it lives inside the standard library. That is correct
on 3.9 (no ``sys.stdlib_module_names``), on Windows, and for extension modules
such as ``math`` that live in ``lib-dynload``.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import sysconfig

PACKAGE = "tracegrade"


def _stdlib_root() -> pathlib.Path:
    return pathlib.Path(sysconfig.get_paths()["stdlib"]).resolve()


def is_stdlib(name: str) -> bool:
    if name in ("builtins", "__future__"):
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return False
    if spec is None:
        return False
    if spec.origin in ("built-in", "frozen"):
        return True
    if spec.submodule_search_locations:
        location = list(spec.submodule_search_locations)[0]
    else:
        location = spec.origin or ""
    try:
        path = pathlib.Path(location).resolve()
    except OSError:
        return False
    root = _stdlib_root()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    bad = []
    for path in sorted(pathlib.Path(PACKAGE).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root != PACKAGE and not is_stdlib(root):
                        bad.append("%s: import %s" % (path, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root != PACKAGE and not is_stdlib(root):
                    bad.append("%s: from %s" % (path, node.module))
    if bad:
        print("third-party imports found:")
        for entry in bad:
            print("  " + entry)
        return 1
    print("no third-party imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())

