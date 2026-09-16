"""The Step base class.

A step is one analysis action. Subclasses declare metadata + a parameter schema
and implement :meth:`run`. The public entry point is :meth:`apply`, which gates on
modality, validates parameters, checks preconditions, runs, and records history —
so subclasses only write the science.
"""
from __future__ import annotations

from .dataset import Dataset
from .errors import ParamError, StepError


class Step:
    #: stable identifier used in serialized pipelines (e.g. "bandpass_filter")
    id: str = ""
    #: human-readable name shown in the UI
    name: str = ""
    #: grouping in the step library (e.g. "Preprocess")
    category: str = ""
    #: modalities this step supports: any of "eeg", "fnirs"
    modalities: list = []
    #: parameter schema (list of cortica.core.params.Param)
    params: list = []

    def applies_to(self, modality: str) -> bool:
        return modality in self.modalities

    def validate_params(self, values: dict) -> dict:
        """Fill defaults, reject unknown keys, and validate every value."""
        schema = {p.name: p for p in self.params}
        unknown = set(values) - set(schema)
        if unknown:
            raise ParamError(f"{self.name}: unknown parameters {sorted(unknown)}")
        validated = {}
        for name, param in schema.items():
            raw = values[name] if name in values else param.default
            validated[name] = param.validate(raw)
        return validated

    def check(self, dataset: Dataset) -> None:
        """Optional precondition. Raise :class:`StepError` if unmet. Default: no-op."""

    def run(self, dataset: Dataset, values: dict) -> Dataset:
        """Do the science and return a new Dataset. Subclasses must implement."""
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

    def apply(self, dataset: Dataset, values: dict | None = None) -> Dataset:
        """Validate, check, run, and record this step against ``dataset``."""
        if not self.applies_to(dataset.modality):
            raise StepError(f"{self.name} does not apply to {dataset.modality} data")
        validated = self.validate_params(values or {})
        self.check(dataset)
        result = self.run(dataset, validated)
        record = {"step": self.id, "params": validated}
        return Dataset(
            result.payload, result.modality, dict(result.meta),
            dataset.history + (record,),
        )
