"""Tests for Dataset — the immutable value that flows between steps."""
from cortica.core.dataset import Dataset


def test_derive_appends_record_and_leaves_original_untouched():
    ds = Dataset(payload=[1], modality="eeg", meta={"sfreq": 250})
    out = ds.derive([1, 2], {"step": "x", "params": {}})

    assert out.payload == [1, 2]
    assert out.history == ({"step": "x", "params": {}},)
    assert out.meta == {"sfreq": 250}
    # original is untouched (immutability)
    assert ds.history == ()
    assert ds.payload == [1]


def test_derive_without_record_keeps_history_length():
    ds = Dataset(payload=0, modality="fnirs")
    out = ds.derive(5)

    assert out.payload == 5
    assert out.history == ()
    assert out.modality == "fnirs"
