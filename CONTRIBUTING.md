# Contributing to Cortica

Thanks for your interest. Cortica is pre-alpha and moving quickly, so opening an
issue to discuss a change before writing it will usually save you time.

## Development setup

```bash
git clone https://github.com/samnemati/cortica.git
cd cortica
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev,gui,gui-test]"
```

## The checks CI runs

Run these before opening a pull request — they are exactly what CI gates on:

```bash
ruff check .                           # lint (line length 100; E, F, I, UP)
mypy cortica                           # types
pytest -q                              # full suite
QT_QPA_PLATFORM=offscreen pytest -q tests/test_gui_*.py   # GUI, headless
```

CI runs the suite on Linux, macOS, and Windows against Python 3.10–3.13, and the
GUI suite offscreen on all three platforms.

`cortica.gui.main_window` is currently excluded from type checking (see the
`[[tool.mypy.overrides]]` block in `pyproject.toml`). Annotating it is a welcome
contribution; please don't add new untyped modules.

## Architecture in one minute

The layering matters, and keeping it intact is the main review criterion:

- **`cortica/core`** — the headless engine: `Dataset`, `Step`, `Pipeline`, registry.
  No Qt, no plotting. This is the source of truth for reproducibility.
- **`cortica/steps`** — concrete analysis steps, MNE-backed, registered into the
  default registry. Each declares its `id`, `name`, `category`, `modalities`, and a
  parameter schema.
- **`cortica/viz.py`, `cortica/group.py`, `cortica/behavior.py`** — computation for
  the analysis views, independent of the GUI so it can be tested headlessly.
- **`cortica/report`** — pipeline + outputs → self-contained HTML.
- **`cortica/gui`** — PySide6 app. Imported lazily; never required by the engine.

## Adding a step

A step is the unit of reproducibility, so it must round-trip through YAML.

1. Subclass `Step` in the right `cortica/steps/` module. Set a **stable** `id` —
   it is written into saved pipelines, so changing one breaks existing files.
2. Declare `modalities` and a `params` schema; `Step.apply` handles modality
   gating, parameter validation, preconditions, and history for you, so `run`
   only does the science.
3. Register it in that module's registration block.
4. Add a test in `tests/test_steps_*.py`. Cover the parameter schema and at least
   one determinism check — the same pipeline re-run must give an identical result.

## Conventions

- Match the surrounding code: comments explain *why*, not *what*.
- Keep the engine importable without Qt, and keep optional dependencies
  (`pyvista`, `pyqtgraph`) behind lazy imports with a clear error message naming
  the extra to install.
- Sample data stays synthetic and offline — a first-time user should never wait
  on a download.
- Add a `CHANGELOG.md` entry under `[Unreleased]` for anything user-visible.

## Scope

Cortica is a *workbench*: the guided UI, the pipeline, the report. The science
underneath belongs in [MNE-Python](https://mne.tools) — if a change would
reimplement something MNE already does, prefer wrapping it.

Cortica is for research only and is not a medical device. Please don't submit
changes that frame it as suitable for clinical diagnosis or treatment.
