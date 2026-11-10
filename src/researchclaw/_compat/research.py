"""Compatibility alias; implementation: researchclaw.pipelines.research."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchclaw.pipelines.research")
