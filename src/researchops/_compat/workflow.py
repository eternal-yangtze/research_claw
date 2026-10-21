"""Compatibility alias; implementation: researchops.application.workflow."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.application.workflow")
