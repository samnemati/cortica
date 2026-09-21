"""The pipeline: an ordered, serializable, re-runnable list of steps.

This is Cortica's source of truth. The GUI appends steps here; saving it yields a
human-readable YAML file; re-running that file on new data reproduces the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from .. import __version__


@dataclass
class PipelineStep:
    """One entry in a pipeline: a step id plus the params to run it with."""

    step_id: str
    params: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, modality: str, steps: list | None = None):
        self.modality = modality
        self.steps: list = list(steps or [])

    def add(self, step_id: str, params: dict | None = None) -> Pipeline:
        """Append a step. Returns self so calls chain."""
        self.steps.append(PipelineStep(step_id, dict(params or {})))
        return self

    def run(self, dataset, registry, progress=None):
        """Execute every step in order, returning the final Dataset.

        Each step validates its own params and modality; a failing step raises
        (StepError/ParamError) and stops the run — the caller decides what to do.

        ``progress``, if given, is called *before* each step as
        ``progress(index, total, step_name)`` so a UI can report live per-step
        status. It runs on whatever thread calls ``run``.
        """
        ds = dataset
        total = len(self.steps)
        for index, pstep in enumerate(self.steps):
            step = registry.get(pstep.step_id)()
            if progress is not None:
                progress(index, total, getattr(step, "name", pstep.step_id))
            ds = step.apply(ds, pstep.params)
        return ds

    def to_dict(self) -> dict:
        return {
            "cortica_version": __version__,
            "modality": self.modality,
            "steps": [{"id": s.step_id, "params": s.params} for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Pipeline:
        pipeline = cls(data["modality"])
        for entry in data.get("steps", []):
            pipeline.add(entry["id"], entry.get("params", {}))
        return pipeline

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> Pipeline:
        return cls.from_dict(yaml.safe_load(text))
