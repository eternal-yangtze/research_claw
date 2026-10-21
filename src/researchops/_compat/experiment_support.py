"""Compatibility alias; implementation: researchops.experiments.support."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.experiments.support")
