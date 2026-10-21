"""ResearchOps: auditable research-agent runtime and literature acquisition."""

from pathlib import Path as _Path

# Historical generated projects import researchops.experiment_support, etc.
# Resolve those names lazily from a separate compatibility directory, without
# importing every subsystem or loading torch when starting a simple CLI command.
__path__.append(str(_Path(__file__).resolve().parent / "_compat"))

__version__ = "0.2.0"
