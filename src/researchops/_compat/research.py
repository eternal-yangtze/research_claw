"""Compatibility alias; implementation: researchops.pipelines.research."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.pipelines.research")
