"""Compatibility alias; implementation: researchops.knowledge.retrieval."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.knowledge.retrieval")
