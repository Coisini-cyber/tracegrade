"""Fail if tracegrade imports anything outside the standard library.

tracegrade advertises zero runtime dependencies; this keeps that honest.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import sysconfig


def stdlib_names() -> set:
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return set(names) | {"tracegrade", "builtins", "__future__"}
    # Python 3.9 fallback: list the standard library directory.
    root = pathlib.Path(sysconfig.get_paths()["stdlib"])
    found = {"tracegrade", "builtins", "__future__", "sys"}
    found.update(sys.builtin_module_names)
    for entry in root.iterdir():
        if entry.name in ("site-packages", "dist-packages"):
            continue
        if entry.suffix == ".py":
            found.add(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").exists():
            found.add(entry.name)
    return found


def main() -> int:
    allowed = stdlib_names()
    bad = []
    for path in sorted(pathlib.Path("tracegrade").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in allowed:
                        bad.append("%s: import %s" % (path, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root not in allowed:
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

