import os
from src.utils.preprocessing.numerical_preprocessing import convert_string_to_float, convert_ranges_string
import numpy as np
import pytest


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("6$", 6.0),
        ("   6$", 6.0),
        (" 7.9$", 7.9),
        (" 7,000.9$", 7000.9),
    ],
)
def test_text_preprocessing_price_numeric(input_text, expected_output):
    
    # Test if text is converted to lowercase correctly
    assert convert_string_to_float(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        (" $7 - $9", '$7'),
    ],
)
def test_text_preprocessing_price_ranges(input_text, expected_output):
    
    # Test if text is converted to lowercase correctly
    assert convert_ranges_string(input_text) == expected_output



"""
@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("hola hola"),
        ("LLM"),
        ("\*><?"),
    ],
)
def test_text_preprocessing_price_numeric(input_text):
    # Test if text is converted to lowercase correctly
    assert np.isnan(input_text)
"""