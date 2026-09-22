"""Tests for the Set montage step (electrode positions, needed for head maps)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.core.registry import default_registry  # noqa: E402
from cortica.io import dataset_from_raw  # noqa: E402
from cortica.steps.montage import SetMontage  # noqa: E402


def _raw_1020():
    names = ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]
    info = mne.create_info(names, 200.0, ch_types="eeg")
    return mne.io.RawArray(np.zeros((6, 200)), info, verbose=False)


def test_set_montage_adds_electrode_positions():
    ds = dataset_from_raw(_raw_1020())
    assert ds.payload.get_montage() is None
    out = SetMontage().apply(ds, {"montage": "standard_1005"})
    assert out.payload.get_montage() is not None


def test_set_montage_loads_a_custom_file(tmp_path):
    path = tmp_path / "custom.tsv"
    positions = {
        "Fp1": (-0.03, 0.08, 0.0), "Fp2": (0.03, 0.08, 0.0),
        "C3": (-0.05, 0.0, 0.07), "C4": (0.05, 0.0, 0.07),
        "O1": (-0.03, -0.08, 0.0), "O2": (0.03, -0.08, 0.0),
    }
    lines = ["name\tx\ty\tz"] + [f"{n}\t{x}\t{y}\t{z}" for n, (x, y, z) in positions.items()]
    path.write_text("\n".join(lines) + "\n")
    ds = dataset_from_raw(_raw_1020())
    out = SetMontage().apply(ds, {"montage": "standard_1005", "custom_file": str(path)})
    montage = out.payload.get_montage()
    assert montage is not None
    assert set(montage.ch_names) == set(positions)  # the custom file won, not the built-in


def test_set_montage_registered():
    import cortica.steps  # noqa: F401

    assert default_registry.get("set_montage") is SetMontage
