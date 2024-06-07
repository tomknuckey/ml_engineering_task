"""
Here the test functions are defined
"""

import pandas as pd
import pytest
from end_to_end_utils import drop_distinct, collect_from_database


def test_drop_distinct_identical():
    """
    This runs the drop distinct function to check that a dataframe without duplicates stays the same
    """

    # Create a sample DataFrame with duplicate rows
    data = {"A": [1, 2, 4], "B": [5, 6, 8], "C": [9, 10, 12]}
    df = pd.DataFrame(data)

    # Apply the function
    result_df = drop_distinct(df)

    # Create the expected DataFrame after dropping duplicates
    expected_data = {"A": [1, 2, 4], "B": [5, 6, 8], "C": [9, 10, 12]}
    expected_df = pd.DataFrame(expected_data)

    # Check if the result DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_drop_distinct_extra_row():
    """
    This runs the drop distinct function to check that a dataframe with 2 duplicate rows drops it
    """

    # Create a sample DataFrame with duplicate rows
    data = {
        "A": [1, 2, 4, 4],
        "B": [5, 6, 8, 8],
        "C": [9, 10, 12, 12],
    }
    df = pd.DataFrame(data)

    # Apply the function
    result_df = drop_distinct(df)

    # Create the expected DataFrame after dropping duplicates
    expected_data = {"A": [1, 2, 4], "B": [5, 6, 8], "C": [9, 10, 12]}
    expected_df = pd.DataFrame(expected_data)

    # Check if the result DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_claim_response():

    """
    Check there's claim status of 0 and 1 and nothing else in the database
    """

    assert collect_from_database("SELECT * FROM CLAIMS.DS_DATASET")['claim_status'].max() == 1
    assert collect_from_database("SELECT * FROM CLAIMS.DS_DATASET")['claim_status'].min() == 0
