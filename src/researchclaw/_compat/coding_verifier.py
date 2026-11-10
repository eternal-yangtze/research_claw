"""Legacy python -m researchclaw.coding_verifier entry point."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("researchclaw.coding.verifier")

if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
