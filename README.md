<h1 align="center">🧠 Cortica</h1>
<p align="center"><em>A guided, reproducible workbench for EEG &amp; fNIRS analysis.</em></p>

<p align="center">
  <a href="#status"><img alt="status" src="https://img.shields.io/badge/status-pre--alpha-orange"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="platforms" src="https://img.shields.io/badge/platforms-Windows%20%7C%20macOS-lightgrey">
</p>

Cortica is a cross-platform desktop app for analyzing **EEG and fNIRS** recordings.
Its one idea: **every action you take in the GUI is recorded as a step in a
re-runnable pipeline.** Reproducibility and a shareable report then come for free —
the same pipeline that ran interactively can be re-run headless on new data and
rendered into an HTML report.

It is built on [MNE-Python](https://mne.tools) (which handles both EEG and fNIRS),
so Cortica focuses on the *workbench* — the guided UI, the pipeline, the report —
rather than reinventing the science underneath.

> **Research use only.** Cortica is a research tool. It is **not** a medical device
> and must not be used for clinical diagnosis or treatment.

## Why Cortica

Powerful toolkits already exist (MNE-Python, MNELAB, EEGLAB, Brainstorm). Cortica's
niche is the combination:

- **Unified EEG + fNIRS** — one coherent tool for both, including studies that
  record them together. Steps declare which modalities they support, and the UI
  only offers steps valid for the loaded data.
- **Guided, but not rigid** — a suggested workflow (import → signal check →
  preprocess → segment → analyze → report) that newcomers can follow and power
  users can step outside of.
- **Reproducible by construction** — the pipeline is the source of truth. Save it
  as YAML, re-run it headless, or hand it to a colleague. What you see is what you
  publish.

## Status

Pre-alpha, under active development. Working today, all covered by tests:

- the reproducible-pipeline **engine** (`Dataset` / `Step` / `Pipeline` + YAML),
- MNE-backed **preprocessing** steps (band-pass filter, resample),
- a headless **CLI** — `cortica run pipeline.yaml raw.fif --out … --report …`,
- an HTML **report** (pipeline + provenance),
- a first **PySide6 GUI** — modality-aware step library, signal viewer, pipeline
  panel, and a threaded Run.

Next: epoching/averaging, fNIRS-specific steps (Beer–Lambert, SCI, TDDR), richer
report figures, and packaged installers. See
[`docs/specs/2026-09-15-cortica-design.md`](docs/specs/2026-09-15-cortica-design.md)
for the full design.

## Install

Cortica isn't on PyPI yet. For now, install from source:

```bash
git clone https://github.com/samnemati/cortica.git
cd cortica
pip install -e ".[gui]"     # engine + desktop GUI
```

The headless engine works without Qt (`pip install -e .`); the `[gui]` extra adds
PySide6 and pyqtgraph for the desktop app.

## Reproducible pipelines

A pipeline is a plain, human-readable YAML file describing an ordered list of
steps. The GUI writes it as you work; you can also run it headless:

```bash
cortica run analysis.pipeline.yaml raw.fif --report out.html
```

```yaml
# analysis.pipeline.yaml (illustrative)
cortica_version: 0.1.0
modality: eeg
steps:
  - id: bandpass_filter
    params: { l_freq: 1.0, h_freq: 40.0 }
  - id: epoch_by_events
    params: { tmin: -0.2, tmax: 0.8 }
```

## Development

```bash
pip install -e ".[dev]"
pytest        # run the engine test suite
ruff check .  # lint

# To also run the GUI tests you need a Qt binding; they run headless (offscreen):
pip install -e ".[dev,gui,gui-test]"
QT_QPA_PLATFORM=offscreen pytest
```

The engine in [`cortica/core`](cortica/core) has no Qt dependency and is fully
unit-tested without a GUI — including a round-trip test proving a saved pipeline
re-runs to an identical result. GUI logic lives in a headless `AppState`
controller so it can be tested offscreen with `pytest-qt`.

## License

[Apache-2.0](LICENSE).
