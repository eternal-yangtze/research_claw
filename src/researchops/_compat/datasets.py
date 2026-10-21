"""Compatibility alias; implementation: researchops.experiments.datasets."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("researchops.experiments.datasets")
