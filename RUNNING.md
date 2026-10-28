# Running Cortica

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "cortica[gui]"       # cortica + MNE + PySide6 + pyqtgraph
```

From a clone instead, use `pip install -e ".[gui]"` from the repository root.

`source .venv/bin/activate` turns on the virtual environment — your prompt will
start with `(.venv)` so you know it's active. Leave it with `deactivate`.

## Launch

```bash
source .venv/bin/activate
cortica
```

On macOS and Linux, `cortica & disown` launches the app **in the background and
detached**, so your terminal stays usable and closing the app won't close the
terminal.

**Quit the app:** close the Cortica window (the terminal is unaffected).

## Using the app

1. Click **Load EEG sample**, **Load fNIRS sample**, or **Load STN-LFP sample** in
   the toolbar (top-left) to load example data instantly — or **Open…** to load your
   own recording (`.fif`, `.edf`, `.snirf`, …). The samples are synthetic and
   offline, so nothing is downloaded.
2. Double-click steps in the **Step library** (left) to add them. Steps that don't
   apply to the loaded modality are greyed out.
3. Select a step in the **Pipeline** (right) to edit its parameters; use **↑ ↓** and
   **Remove** to reorder or delete.
4. Switch the center **View** to explore the data — time series, power spectrum,
   topography, time-frequency, connectivity, decoding, statistics, and more.
5. Click **Run pipeline**, then **Export report…** to save a shareable HTML report.

## Headless (no GUI)

The engine and CLI install without Qt (`pip install cortica`):

```bash
cortica run analysis.pipeline.yaml raw.fif --out clean_raw.fif --report out.html
```
