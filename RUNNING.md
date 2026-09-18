# Running Cortica

## Every time (your `.venv` is already set up)

```bash
cd /Users/snemati/Documents/Projects/Cool_stuff/Medical_image_software/cortica
source .venv/bin/activate
cortica & disown
```

- `source .venv/bin/activate` turns on the virtual environment — your prompt will
  start with `(.venv)` so you know it's active.
- `cortica & disown` launches the app **in the background and detached**, so your
  terminal stays usable and **closing the app won't close the terminal**.

**Quit the app:** just close the Cortica window (the terminal is unaffected).
**Leave the venv** when you're done (optional): `deactivate`.

## Using the app

1. Click **Load EEG sample** or **Load fNIRS sample** in the toolbar (top-left) to
   load example data instantly — or **Open…** to load your own recording
   (`.fif`, `.edf`, `.snirf`, …).
2. Double-click steps in the **Step library** (left) to add them. Steps that don't
   apply to the loaded modality are greyed out.
3. Select a step in the **Pipeline** (right) to edit its parameters; use **↑ ↓** and
   **Remove** to reorder or delete.
4. Switch the center **View** between **Time series** and **Power spectrum**.
5. Click **Run pipeline**, then **Export report…** to save a shareable HTML report.

## First-time setup (only if `.venv` is missing or you cloned fresh)

```bash
cd /Users/snemati/Documents/Projects/Cool_stuff/Medical_image_software/cortica
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui]"      # cortica + MNE + PySide6 + pyqtgraph (a few minutes)
cortica & disown
```

## Headless (no GUI)

```bash
cortica run analysis.pipeline.yaml raw.fif --out clean_raw.fif --report out.html
```
