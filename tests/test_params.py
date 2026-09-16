"""Tests for the declarative parameter schema.

One schema declaration must drive validation, serialization, and (later) the
auto-generated GUI form. These tests pin the validation/coercion behavior and the
schema dict the GUI will consume.
"""
import pytest

from cortica.core.params import Bool, Choice, Float, Int, ParamError, Str


def test_float_accepts_value_in_range_and_coerces_int_to_float():
    p = Float("l_freq", 1.0, min=0.0, max=100.0, unit="Hz")
    result = p.validate(2)
    assert result == 2.0
    assert isinstance(result, float)


def test_float_rejects_value_above_max():
    p = Float("h_freq", 40.0, min=0.0, max=50.0)
    with pytest.raises(ParamError):
        p.validate(60.0)


def test_float_rejects_value_below_min():
    p = Float("h_freq", 40.0, min=1.0, max=50.0)
    with pytest.raises(ParamError):
        p.validate(0.0)


def test_float_rejects_non_numeric():
    p = Float("x", 1.0)
    with pytest.raises(ParamError):
        p.validate("not a number")


def test_int_rejects_non_integer_float():
    p = Int("n_components", 20)
    with pytest.raises(ParamError):
        p.validate(1.5)


def test_int_accepts_integral_float_and_returns_int():
    p = Int("n_components", 20)
    result = p.validate(3.0)
    assert result == 3
    assert isinstance(result, int)


def test_choice_rejects_value_not_in_options():
    p = Choice("method", "fir", options=["fir", "iir"])
    with pytest.raises(ParamError):
        p.validate("wavelet")


def test_choice_accepts_valid_option():
    p = Choice("method", "fir", options=["fir", "iir"])
    assert p.validate("iir") == "iir"


def test_bool_coerces_recognized_string():
    p = Bool("apply", True)
    assert p.validate("true") is True
    assert p.validate(False) is False


def test_str_rejects_non_string():
    p = Str("label", "")
    with pytest.raises(ParamError):
        p.validate(123)


def test_to_dict_exposes_schema_for_the_gui():
    p = Float("l_freq", 1.0, min=0.0, max=50.0, unit="Hz", label="Low cutoff")
    d = p.to_dict()
    assert d == {
        "name": "l_freq",
        "type": "float",
        "default": 1.0,
        "label": "Low cutoff",
        "min": 0.0,
        "max": 50.0,
        "unit": "Hz",
    }
