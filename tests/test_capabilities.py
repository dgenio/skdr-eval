"""Tests for skdr_eval.capabilities."""

import importlib

import skdr_eval
from skdr_eval import capabilities as cap_module


def test_get_capabilities_schema():
    caps = skdr_eval.get_capabilities()
    assert isinstance(caps, dict)
    for key in ("choice", "viz", "speed"):
        assert key in caps
        assert isinstance(caps[key], bool)
    assert "missing_extras" in caps
    assert isinstance(caps["missing_extras"], list)
    assert all(isinstance(x, str) for x in caps["missing_extras"])
    assert caps["missing_extras"] == sorted(caps["missing_extras"])


def test_choice_capability_matches_scipy_presence():
    caps = skdr_eval.get_capabilities()
    has_scipy = importlib.util.find_spec("scipy") is not None
    assert caps["choice"] is has_scipy


def test_missing_extras_listed_when_module_absent(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "matplotlib":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(cap_module.importlib.util, "find_spec", fake_find_spec)
    caps = cap_module.get_capabilities()
    assert caps["viz"] is False
    assert "viz" in caps["missing_extras"]


def test_no_missing_extras_when_all_present(monkeypatch):
    def all_present(_name):
        class _Spec:
            pass

        return _Spec()

    monkeypatch.setattr(cap_module.importlib.util, "find_spec", all_present)
    caps = cap_module.get_capabilities()
    assert caps["missing_extras"] == []
    assert caps["choice"] is True
    assert caps["viz"] is True
    assert caps["speed"] is True


def test_find_spec_value_error_treated_as_missing(monkeypatch):
    def raises(_name):
        raise ValueError("simulated namespace package")

    monkeypatch.setattr(cap_module.importlib.util, "find_spec", raises)
    caps = cap_module.get_capabilities()
    assert caps["choice"] is False
    assert caps["missing_extras"] == sorted(["choice", "viz", "speed"])
