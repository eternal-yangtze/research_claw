"""Compatibility alias; implementation: researchclaw.experiments.datasets."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchclaw.experiments.datasets")
