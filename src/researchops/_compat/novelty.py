"""Compatibility alias; implementation: researchops.knowledge.novelty."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.knowledge.novelty")
