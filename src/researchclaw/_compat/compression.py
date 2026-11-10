"""Compatibility alias; implementation: researchclaw.knowledge.compression."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchclaw.knowledge.compression")
