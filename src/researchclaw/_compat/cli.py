"""Compatibility alias; implementation: researchclaw.application.cli."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("researchclaw.application.cli")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    _sys.modules[__name__] = _implementation
