
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from train_model import load_and_prepare_data


def test_dataset_loads_successfully():
    """Dataset should load and return features and target."""
    X, y = load_and_prepare_data()

    assert isinstance(X, pd.DataFrame)
    assert len(X) > 0
    assert len(y) > 0


def test_features_and_target_have_matching_rows():
    """Features and target must have the same number of rows."""
    X, y = load_and_prepare_data()

    assert len(X) == len(y)


def test_features_do_not_contain_missing_values():
    """Prepared features should not contain missing values."""
    X, y = load_and_prepare_data()

    assert not X.isnull().any().any()
    assert not pd.isnull(y).any()


def test_target_contains_expected_churn_classes():
    """Target should contain the expected binary churn labels."""
    X, y = load_and_prepare_data()

    assert set(y.unique()).issubset({0, 1})
