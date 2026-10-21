"""Compatibility alias; implementation: researchops.coding.sandbox."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.coding.sandbox")
