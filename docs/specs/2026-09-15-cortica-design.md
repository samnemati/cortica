# Cortica — Design Spec

**Status:** approved design, pre-implementation
**Date:** 2026-09-15
**Author:** samnemati
**Audience:** the author + future contributors evaluating or building Cortica

---

## 1. Overview & goals

Cortica is a **cross-platform (Windows + macOS) desktop application for analyzing
EEG and fNIRS recordings**, aimed at **research neuroscientists**.

Its defining idea — the reason it exists alongside MNE-Python, MNELAB, EEGLAB, and
Brainstorm — is a single mechanism:

> **Every action a user takes in the GUI is recorded as a step in a re-runnable
> pipeline.**

From that one mechanism, three features fall out for free:

1. **Undo/redo** — pop/replay steps.
2. **Reproducibility** — save the pipeline as YAML; re-run it headless on new data.
3. **Reporting** — each step self-describes into a shareable HTML report.

Cortica is a *workbench*, not a new analysis library. The science runs on
[MNE-Python](https://mne.tools), which supports both EEG and fNIRS natively. Cortica
supplies the guided UI, the pipeline model, and the report.

## 2. Scope

**v1 (this design):** EEG **and** fNIRS, together. Load → signal check → preprocess
→ segment → analyze → report, for both modalities, with a reproducible pipeline and
an HTML report.

**Deferred (future slices), explicitly out of scope for v1:**

- Volumetric modalities (MRI / PET / CT). These are a different data model (3D
  volumes vs. time series) and a separate future effort.
- Source localization, connectivity, and machine-learning decoding beyond a basic
  first pass.
- Any clinical / regulated use (see constraints).

## 3. Users & constraints

- **Users:** research neuroscientists. **Research use only** — Cortica is not a
  medical device; no clinical/diagnostic use, and therefore no medical-device
  regulatory burden (FDA/CE, QMS) is taken on.
- **Distribution constraint (hard):** **no paid distribution.** No Apple Developer
  ID, no code-signing certificates. Ship via free channels only (see §10).
- **Team:** solo / small. The architecture favors maintainability and testability
  over breadth.

## 4. Architecture

Three strictly separated layers. The dependency arrow points down; nothing lower
imports anything higher, and **`core` never imports Qt.**

```
┌───────────────────────────────────────────────┐
│  gui/     PySide6 desktop app (optional extra)  │  orchestrates only; never analyzes
├───────────────────────────────────────────────┤
│  report/  pipeline + outputs -> HTML            │  built on mne.Report
├───────────────────────────────────────────────┤
│  steps/   concrete MNE-backed Step subclasses   │  MNE imported lazily
├───────────────────────────────────────────────┤
│  core/    Dataset · Step · Pipeline · registry  │  headless, no Qt, fully testable
└───────────────────────────────────────────────┘
        core + a Step registry also run headless via the `cortica` CLI
```

**Rationale for native PySide6 + MNE in-process** (chosen over a web/Electron+Python
backend or a Panel/Dash rapid-GUI): keeps everything in one language and one process
(no serialization boundary for large recordings), gives fast interactive signal
views via `pyqtgraph`, matches the proven pattern (MNELAB, Brainstorm), and is the
simplest to maintain for a solo developer. PySide6 is LGPL, avoiding GPL entanglement
for a distributed product.

## 5. Core model (the reproducibility mechanism)

The heart of Cortica. All of it lives in `cortica.core` and is Qt-free.

- **`Dataset`** — the object passed between steps. Wraps the current payload (an MNE
  `Raw`/`Epochs` object in real use), a `modality` tag (`"eeg"` / `"fnirs"`),
  a free-form `meta` dict (sampling rate, channel count, …), and a `history` of
  applied steps.
- **`Param`** (and `Float`, `Int`, `Bool`, `Choice`, `Str`) — a declarative
  parameter schema. A step declares its parameters as data; this single declaration
  drives (a) validation, (b) serialization, and (c) auto-generated GUI form widgets.
  No hand-written Qt per step.
- **`Step`** — base class. Class-level metadata: `id` (stable, e.g.
  `bandpass_filter`), `name`, `category`, `modalities`, and `params`. Instance
  methods: `applies_to(modality)`, `validate_params(values)`, `run(dataset, values)
  -> Dataset` (subclasses implement), optional `report(...)`, optional
  precondition `check(dataset)`.
- **`Registry`** — maps step `id` → Step class; lists steps by category/modality;
  discovers drop-in plugin steps. The GUI's step library and the YAML loader both
  resolve steps through it.
- **`Pipeline`** — an ordered list of `(step_id, validated_params)`. Responsibilities:
  - `run(dataset, registry)` — execute each step in order, catching per-step errors
    and recording status.
  - `to_yaml()` / `from_yaml()` — a human-readable, versioned serialization
    including a provenance block (Cortica version, input hashes, seed).
  - **Determinism guarantee:** serialize → reload → re-run produces an identical
    result. This is a tested invariant, not a hope.

## 6. Modality-awareness (EEG + fNIRS)

Both modalities are MNE objects (MNE models fNIRS as channel types). Each `Step`
declares `modalities` (`["eeg"]`, `["fnirs"]`, or both). The registry filters the
step library by the loaded dataset's modality: shared steps (filter, crop, epoch)
apply to both; modality-specific steps register their applicability — e.g. **ICA**
and **re-reference** are EEG-only; **Beer–Lambert**, **TDDR motion correction**, and
**scalp coupling index** are fNIRS-only. The GUI greys out (rather than hides)
inapplicable steps so the difference is legible.

## 7. GUI structure & guided UX

A framed three-pane desktop window (see the design mockup):

- **Top:** a guided **stepper** (Import → Signal check → Preprocess → Segment →
  Analyze → Report) — scaffolding for newcomers, non-blocking for power users.
- **Left:** data browser + a **step library** filtered by modality.
- **Center:** the **signal viewer** (`pyqtgraph`): time series, power spectrum,
  sensors/topography.
- **Right:** the **pipeline** — the numbered, ordered, editable list of applied
  steps, each with its parameters. This is the reproducible artifact, given
  first-class real estate.

Heavy operations run off the GUI thread (see §11), so the pipeline can execute while
the window stays responsive.

## 8. Extensibility

Adding an analysis is writing one small `Step` subclass:

```python
class BandpassFilter(Step):
    id = "bandpass_filter"
    name = "Band-pass filter"
    category = "Preprocess"
    modalities = ["eeg", "fnirs"]
    params = [Float("l_freq", 1.0, unit="Hz"),
              Float("h_freq", 40.0, unit="Hz")]

    def run(self, ds, p):   # calls MNE, returns a new Dataset
        ...
    def report(self, ds, result):   # optional: figures + summary
        ...
```

The `params` schema auto-generates the GUI form, the serialization, and the report
entry — write once, get all three. Built-in steps plus user steps dropped into a
`plugins/` directory are discovered by the registry. This is the highest-leverage
decision for "make it mine": differentiating features are just new steps.

## 9. Reporting

Built on `mne.Report` (not reinvented). Each step's optional `report()` contributes a
section (parameters + figures + plain-language summary). The output is a
self-contained, shareable HTML file with an embedded **provenance block** (pipeline
YAML, input file hashes, package versions, seed). Because it is generated from the
same pipeline that ran interactively: what you see is what you publish.

## 10. Cross-platform packaging & distribution (free only)

Two channels for two audiences, all zero-cost:

- **`pip install` / conda** — for Python-savvy researchers. pip/conda packages are
  not quarantined, so **macOS Gatekeeper never triggers** — no warning, no signing.
  This is the primary channel.
- **Standalone installers** for non-coders — built with **Briefcase** (BeeWare;
  PyInstaller as fallback). Distributed **unsigned**:
  - macOS: user approves once via System Settings → Privacy & Security → "Open
    Anyway" (or `xattr -dr com.apple.quarantine`). A **Homebrew tap** is offered as a
    cleaner path (Homebrew strips quarantine, so an unsigned app opens directly).
  - Windows: unsigned `.msi`/`.exe`; a one-time SmartScreen "Run anyway".
- **Paid signing/notarization is deferred indefinitely** — only revisited if the
  tool gets real traction.
- **Dependency note:** the EEG/fNIRS stack (MNE + numpy/scipy) is pure-Python with
  clean wheels on Windows and macOS (Intel + Apple Silicon), so packaging risk is
  low — much lower than a volumetric stack (VTK/ITK/ANTs).
- **CI:** GitHub Actions matrix (Windows/macOS/Linux × Python 3.10–3.12) installs,
  lints, and tests on every push; builds installers on tagged releases.

## 11. Robustness

- **Responsiveness:** a step's `run()` executes on a **worker thread** (Qt
  `QThreadPool`), never the GUI thread; progress and cancel surface in the status
  bar. MNE's numpy-heavy sections release the GIL, so threading yields real
  responsiveness.
- **Errors are per-step and survivable:** parameters and preconditions are validated
  *before* execution (e.g. "Epoch needs events" → a clear message, not a crash); a
  failed step is marked failed with a readable reason, the app stays alive, and the
  pipeline stays intact for the user to fix and re-run. The pipeline (source of
  truth) is autosaved for crash recovery.
- **Testing:**
  - `core/` is headless → **fully unit-tested with pytest**, no GUI.
  - **Determinism/round-trip test:** build a pipeline, run it, serialize, reload,
    re-run headless → identical result. This directly tests the reproducibility
    promise.
  - Step tests build tiny in-memory MNE objects (no downloads).
  - Golden-file report comparison; light `pytest-qt` smoke tests for the GUI.
  - Ships tiny built-in EEG + fNIRS sample data for tests, demos, and first-run.

## 12. Implementation phases

1. **Engine core** (`cortica.core`): Dataset, Param schema, Step, Registry, Pipeline
   + serialization + determinism, all TDD. *(first slice)*
2. **CLI**: `cortica run pipeline.yaml input --report out.html` headless runner.
3. **First real steps** (`cortica.steps`, MNE-backed): load raw, band-pass filter,
   epoch, average — with in-memory tests.
4. **Report** (`cortica.report`): pipeline → `mne.Report` HTML with provenance.
5. **GUI shell** (`cortica.gui`): main window, data browser, pyqtgraph viewer,
   pipeline panel, modality-aware step library, threaded runner.
6. **fNIRS-specific steps**: Beer–Lambert, SCI, TDDR.
7. **Packaging**: Briefcase config, unsigned installers, Homebrew tap, release CI.

## 13. Risks & open questions

- **GUI test brittleness** — keep `pytest-qt` coverage to smoke tests; push logic
  into the headless core where the real coverage lives.
- **Unsigned-app friction** — mitigated by making pip/conda the primary channel and
  documenting the one-time bypass clearly.
- **fNIRS breadth** — `mne-nirs` covers the essentials; advanced GLM analysis may
  land in a later slice.
- **Naming/branding** — "Cortica" is the working name; confirm no conflicting
  package on PyPI before first release.
