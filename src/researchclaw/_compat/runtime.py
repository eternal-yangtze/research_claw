"""Compatibility alias; implementation: researchclaw.harness.runtime."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchclaw.harness.runtime")
