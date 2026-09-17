# Changelog

Notable changes to Cortica. Format follows [Keep a Changelog](https://keepachangelog.com);
versions follow [semantic versioning](https://semver.org).

## [Unreleased]

## [0.1.0] — 2026-09-17

First public pre-alpha.

### Added
- Reproducible-pipeline **engine** (`Dataset`, `Step`, `Pipeline`, registry) with
  human-readable YAML serialization and a determinism guarantee (a saved pipeline
  re-runs to an identical result).
- Data **loading** with automatic EEG/fNIRS modality detection.
- **Preprocessing** steps: band-pass filter, resample.
- **Segment/analyze** steps: fixed-length epoching, averaging (evoked).
- **fNIRS** steps: optical density, Beer–Lambert (→ HbO/HbR), scalp coupling index,
  TDDR motion correction.
- Self-contained **HTML report** (pipeline + provenance block).
- Headless **CLI**: `cortica run pipeline.yaml input --out … --report …`.
- **PySide6 desktop GUI**: modality-aware step library, signal viewer, auto-generated
  per-step parameter editors, reorder/remove, threaded run, and one-click EEG/fNIRS
  sample data.

[Unreleased]: https://github.com/samnemati/cortica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/samnemati/cortica/releases/tag/v0.1.0
