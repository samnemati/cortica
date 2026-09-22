"""Tests for the fNIRS-specific steps: optical density, Beer-Lambert, SCI, TDDR.

Each runs on the synthetic CW-amplitude sample and enforces its input type, so
using them out of order fails with a clear message.
"""
import pytest

mne = pytest.importorskip("mne")

from cortica.core.errors import StepError  # noqa: E402
from cortica.core.registry import default_registry  # noqa: E402
from cortica.samples import fnirs_sample  # noqa: E402
from cortica.steps.fnirs import (  # noqa: E402
    BeerLambert,
    EnhanceNegativeCorrelation,
    KeepLongChannels,
    OpticalDensity,
    ScalpCouplingIndex,
    ShortChannelRegression,
    Tddr,
)


def test_optical_density_converts_cw_to_od():
    out = OpticalDensity().apply(fnirs_sample(), {})
    assert "fnirs_od" in set(out.payload.get_channel_types())


def test_optical_density_rejects_non_cw_input():
    od = OpticalDensity().apply(fnirs_sample(), {})
    with pytest.raises(StepError):
        OpticalDensity().apply(od, {})  # already optical density


def test_beer_lambert_produces_hbo_and_hbr():
    od = OpticalDensity().apply(fnirs_sample(), {})
    hb = BeerLambert().apply(od, {"ppf": 6.0})
    types = set(hb.payload.get_channel_types())
    assert "hbo" in types and "hbr" in types


def test_beer_lambert_requires_optical_density_first():
    with pytest.raises(StepError):
        BeerLambert().apply(fnirs_sample(), {"ppf": 6.0})  # raw CW, not OD


def test_scalp_coupling_index_flags_low_quality_channels():
    od = OpticalDensity().apply(fnirs_sample(), {})
    out = ScalpCouplingIndex().apply(od, {"threshold": 0.98})  # strict -> flags the weakest
    assert len(out.payload.info["bads"]) >= 1


def test_tddr_runs_on_optical_density():
    od = OpticalDensity().apply(fnirs_sample(), {})
    out = Tddr().apply(od, {})
    assert "fnirs_od" in set(out.payload.get_channel_types())


def test_short_channel_regression_runs_on_optical_density():
    od = OpticalDensity().apply(fnirs_sample(), {})
    out = ShortChannelRegression().apply(od, {})
    assert "fnirs_od" in set(out.payload.get_channel_types())


def test_short_channel_regression_requires_optical_density():
    with pytest.raises(StepError):
        ShortChannelRegression().apply(fnirs_sample(), {})  # raw CW, not OD


def test_enhance_negative_correlation_runs_on_haemoglobin():
    hb = BeerLambert().apply(OpticalDensity().apply(fnirs_sample(), {}), {"ppf": 6.0})
    out = EnhanceNegativeCorrelation().apply(hb, {})
    assert {"hbo", "hbr"} & set(out.payload.get_channel_types())


def test_enhance_negative_correlation_requires_haemoglobin():
    od = OpticalDensity().apply(fnirs_sample(), {})
    with pytest.raises(StepError):
        EnhanceNegativeCorrelation().apply(od, {})  # OD, not haemoglobin


def test_keep_long_channels_drops_the_short_pair():
    od = OpticalDensity().apply(fnirs_sample(), {})
    out = KeepLongChannels().apply(od, {})
    assert len(out.payload.ch_names) == 8  # 10 channels minus the 2 short ones


def test_fnirs_steps_register_themselves():
    import cortica.steps  # noqa: F401

    for step_id, cls in (
        ("optical_density", OpticalDensity),
        ("beer_lambert", BeerLambert),
        ("scalp_coupling_index", ScalpCouplingIndex),
        ("tddr", Tddr),
        ("short_channel_regression", ShortChannelRegression),
        ("enhance_negative_correlation", EnhanceNegativeCorrelation),
        ("keep_long_channels", KeepLongChannels),
    ):
        assert default_registry.get(step_id) is cls
