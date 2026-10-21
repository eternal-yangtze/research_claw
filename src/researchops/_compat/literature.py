"""Compatibility alias; implementation: researchops.papers.connectors."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.papers.connectors")
