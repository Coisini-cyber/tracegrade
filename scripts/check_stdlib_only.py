"""Fail if the package imports anything outside the standard library.

The project advertises zero runtime dependencies; this keeps that honest.
"""

from __future__ import annotations

import ast
import pathlib
import pkgutil
import sys
import sysconfig

PACKAGE = "tracegrade"


def stdlib_names() -> set:
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return set(names) | {PACKAGE, "builtins", "__future__"}
    # Python 3.9 fallback: enumerate the standard library directory. pkgutil
    # understands .py files, packages, and extension modules (.so/.pyd), which
    # is what a plain directory listing would miss for modules like math.
    root = pathlib.Path(sysconfig.get_paths()["stdlib"])
    found = {PACKAGE, "builtins", "__future__", "sys"}
    found.update(sys.builtin_module_names)
    for _, name, _ in pkgutil.iter_modules([str(root)]):
        found.add(name)
    return found


def main() -> int:
    allowed = stdlib_names()
    bad = []
    for path in sorted(pathlib.Path(PACKAGE).rglob("*.py")):
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

