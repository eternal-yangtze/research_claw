"""Compatibility alias; implementation: researchops.core.models."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.core.models")
