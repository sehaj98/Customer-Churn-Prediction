
from pathlib import Path
import sys

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from train_model import load_and_prepare_data, train_models


MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILES = [
    "logistic_regression_model.joblib",
    "random_forest_model.joblib",
]


def test_training_data_has_valid_features_and_target():
    """Training data should have features and binary target labels."""
    X, y = load_and_prepare_data()

    assert not X.empty
    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}
    assert not X.isnull().any().any()


def test_both_trained_models_are_saved():
    """Training should save both model files."""
    train_models()

    for filename in MODEL_FILES:
        model_path = MODEL_DIR / filename
        assert model_path.exists()


@pytest.mark.parametrize("filename", MODEL_FILES)
def test_saved_model_loads_and_predicts(filename):
    """Each saved model should load and predict successfully."""
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        train_models()

    model = joblib.load(model_path)
    X, y = load_and_prepare_data()

    # Use one row to test the prediction interface.
    sample = X.iloc[[0]]
    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert probabilities[0][0] + probabilities[0][1] == pytest.approx(1.0)
    assert 0 <= probabilities[0][1] <= 1

    prediction = model.predict(sample)
    assert prediction[0] in [0, 1]


def test_random_forest_has_expected_type():
    """The saved Random Forest should be a RandomForestClassifier."""
    model_path = MODEL_DIR / "random_forest_model.joblib"

    if not model_path.exists():
        train_models()

    model = joblib.load(model_path)

    assert isinstance(model, RandomForestClassifier)


def test_logistic_regression_is_a_pipeline():
    """Logistic Regression should include preprocessing and a classifier."""
    model_path = MODEL_DIR / "logistic_regression_model.joblib"

    if not model_path.exists():
        train_models()

    model = joblib.load(model_path)

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps
