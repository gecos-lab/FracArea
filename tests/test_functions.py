from src.template_python_project.functions import (
    sum,
    subtract,
    multiply,
    divide,
    string_upper,
)
import os
import pytest


# @pytest.mark.skip(reason="Skipping this test for now because ...")
@pytest.mark.slow
def test_sum():
    assert sum(1, 2) == 3
    assert sum(2, 2) == 4


def test_subtract():
    assert subtract(2, 3) == -1
    assert subtract(3, 3) == 0


def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(3, 3) == 9


def test_divide():
    assert divide(10, 2) == 5
    assert divide(3, 3) == 1


def test_string_upper():
    assert string_upper() == "HELLO WORLD"
    assert string_upper("Hi") == "HI"
