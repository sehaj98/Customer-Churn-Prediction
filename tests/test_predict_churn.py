
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

import predict_churn


def test_missing_model_file(capsys, monkeypatch, tmp_path):
    """Check that a missing model produces a helpful message."""
    missing_model = tmp_path / "missing_model.joblib"

    monkeypatch.setattr(
        predict_churn,
        "MODEL_PATH",
        missing_model,
    )

    predict_churn.predict_customer()

    output = capsys.readouterr().out

    assert "Model not found" in output


def test_ask_number_accepts_valid_input(monkeypatch):
    """Check that invalid input is rejected until a valid number is entered."""
    inputs = iter(["abc", "-5", "12"])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    result = predict_churn.ask_number(
        "Tenure in months",
        0,
        72,
        integer=True,
    )

    assert result == 12


def test_ask_choice_accepts_case_insensitive_input(monkeypatch):
    """Check that choice input is case-insensitive."""
    monkeypatch.setattr(
        "builtins.input",
        lambda _: "female",
    )

    result = predict_churn.ask_choice(
        "Gender",
        ["Female", "Male"],
    )

    assert result == "Female"


def test_prediction_output(monkeypatch, capsys):
    """Check the prediction flow using a fake model."""
    # Get the exact feature columns used by the training dataset.
    X, _ = predict_churn.load_and_prepare_data()

    class FakeModel:
        def predict_proba(self, customer):
            self.customer = customer.copy()
            return [[0.1415, 0.8585]]

    fake_model = FakeModel()

    # Replace model loading with the fake model.
    monkeypatch.setattr(
        predict_churn.joblib,
        "load",
        lambda path: fake_model,
    )

    # Return the real dataset's feature columns without reloading it.
    monkeypatch.setattr(
        predict_churn,
        "load_and_prepare_data",
        lambda: (X, None),
    )

    choices = {
        "Gender": "Male",
        "Senior citizen": "0",
        "Has partner": "Yes",
        "Has dependents": "No",
        "Phone service": "Yes",
        "Multiple lines": "No",
        "Internet service": "Fiber optic",
        "Online security": "No",
        "Online backup": "Yes",
        "Device protection": "No",
        "Tech support": "No",
        "Streaming TV": "Yes",
        "Streaming movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless billing": "Yes",
        "Payment method": "Electronic check",
    }

    numbers = {
        "Tenure in months": 12,
        "Monthly charges": 85.50,
        "Total charges": 1026.00,
    }

    # Simulate customer choices.
    monkeypatch.setattr(
        predict_churn,
        "ask_choice",
        lambda prompt, options: choices[prompt],
    )

    # Accept positional and keyword arguments from ask_number().
    monkeypatch.setattr(
        predict_churn,
        "ask_number",
        lambda prompt, *args, **kwargs: numbers[prompt],
    )

    # The application checks whether this path exists.
    monkeypatch.setattr(
        predict_churn,
        "MODEL_PATH",
        PROJECT_ROOT / "models" / "random_forest_model.joblib",
    )

    predict_churn.predict_customer()

    output = capsys.readouterr().out

    # Check the displayed prediction.
    assert "Predicted class: Churn" in output
    assert "85.85%" in output

    # Check that the correct features reach the model.
    assert list(fake_model.customer.columns) == list(X.columns)
    assert fake_model.customer.shape == (1, X.shape[1])

    # Check that numeric customer details were passed correctly.
    assert fake_model.customer.iloc[0]["tenure"] == 12
    assert fake_model.customer.iloc[0]["MonthlyCharges"] == 85.50
    assert fake_model.customer.iloc[0]["TotalCharges"] == 1026.00
