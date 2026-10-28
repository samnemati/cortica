# Changelog

Notable changes to Cortica. Format follows [Keep a Changelog](https://keepachangelog.com);
versions follow [semantic versioning](https://semver.org).

## [Unreleased]

## [0.1.0] — 2026-10-28

First public release.

### Added

**Engine.** Reproducible-pipeline core (`Dataset`, `Step`, `Pipeline`, registry) with
human-readable YAML serialization and a determinism guarantee — a saved pipeline
re-runs to an identical result. Data loading with automatic EEG/fNIRS modality
detection; every step declares the modalities it supports.

**EEG steps.** Set montage (built-in montages plus custom montage files), band-pass
filter, notch filter, resample, re-reference (any channel or average), bipolar
reference for adjacent depth contacts, interpolate bad channels, and ICA artifact
removal with a component-inspector dialog.

**fNIRS steps.** Optical density, Beer–Lambert law (→ HbO/HbR), scalp coupling index,
short-channel regression, HbO/HbR anti-correlation enhancement, keep-long-channels,
and three motion corrections (TDDR, wavelet, spline).

**Segmenting & averaging.** Fixed-length epoching, event-based epoching from stimulus
triggers or annotations, and averaging to evoked.

**Analysis views.** Time series, power spectrum, topographic head maps (absolute and
relative band power), time-frequency (Morlet and multitaper), signal quality (SCI for
fNIRS, variance outliers for EEG), beta bursts for LFP/DBS data, connectivity (PLV,
coherence, wPLI and more, as a matrix or connectogram), decoding (classifier choice,
temporal generalization, CSP), cluster-permutation statistics (channel-averaged and
spatiotemporal), condition comparison with difference wave, ERP peak picking,
template source localization (fsaverage dSPM) with region activity, brain-behavior
correlation and regression, regression-ERP (rERP), and an fNIRS GLM with contrasts
and a brain-space topomap.

**Group level.** Group analysis across subjects over band power, connectivity, and
fNIRS GLM beta, plus group-level brain-behavior analysis.

**Guided GUI.** PySide6 desktop app with a modality-aware step library, a workflow
panel tracking stages and suggesting next steps, auto-generated per-step parameter
editors, channel selection driving every view, reorder/remove, threaded runs with
per-step progress, and one-click EEG, fNIRS, and STN-LFP sample data (synthetic and
offline — no download).

**Output.** Self-contained HTML report with embedded figures and a provenance block,
an in-app report preview, figure saving (SVG/PNG), and CSV export of band-power and
connectivity values.

**Headless CLI.** `cortica run pipeline.yaml input --out … --report …` re-runs any
GUI-authored pipeline with no Qt installed.

### Notes

- Requires Python 3.10+. Tested on Linux, macOS, and Windows against Python
  3.10–3.13.
- The desktop GUI is an optional extra (`pip install "cortica[gui]"`) so the engine
  and CLI install without Qt.
- Pre-alpha and **research use only** — not a medical device.

[Unreleased]: https://github.com/samnemati/cortica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/samnemati/cortica/releases/tag/v0.1.0
