"""Exception types for the Cortica engine.

All are subclasses of :class:`CorticaError` so callers can catch the whole family;
:class:`ParamError` also subclasses :class:`ValueError` for ergonomic handling.
"""


class CorticaError(Exception):
    """Base class for all Cortica engine errors."""


class ParamError(CorticaError, ValueError):
    """A parameter value failed validation."""


class StepError(CorticaError):
    """A step could not run (bad precondition, or the underlying op failed)."""


class PipelineError(CorticaError):
    """A pipeline could not be built, loaded, or executed."""
