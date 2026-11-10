"""Compatibility alias; implementation: researchclaw.evaluation.silver."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchclaw.evaluation.silver")
