"""Acceptance tests for the Qwen-generated workspace/calculator.py.

Run from the repository root:
    python -m pytest -q examples/coding_agent/test_calculator.py

This intentionally imports and executes locally generated code. Only run it
when you trust the workspace contents.
"""
import importlib.util
from pathlib import Path

import pytest


CALCULATOR = Path(__file__).resolve().parents[2] / "workspace" / "calculator.py"


@pytest.fixture(scope="module")
def calculator():
    assert CALCULATOR.is_file(), (
        f"Generated file not found: {CALCULATOR}. "
        "Run the live Coding Agent experiment first."
    )
    spec = importlib.util.spec_from_file_location("generated_calculator", CALCULATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add(calculator):
    assert calculator.add(2, 3) == 5


def test_subtract(calculator):
    assert calculator.subtract(7, 4) == 3


def test_multiply(calculator):
    assert calculator.multiply(3, 4) == 12


def test_divide(calculator):
    assert calculator.divide(10, 2) == 5


def test_divide_by_zero(calculator):
    with pytest.raises(ValueError):
        calculator.divide(10, 0)
