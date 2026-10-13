
from pathlib import Path

import joblib
import pandas as pd

from train_model import load_and_prepare_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.joblib"


def ask_choice(prompt, choices):
    """Accept options regardless of capitalization or extra spaces."""

    while True:
        print(f"\n{prompt}")

        for index, choice in enumerate(choices, start=1):
            print(f"  {index}. {choice}")

        value = input("Enter option name, first letter, or number: ")
        value = value.strip().casefold()

        # Allow selecting an option by its number.
        if value.isdigit():
            number = int(value)

            if 1 <= number <= len(choices):
                return choices[number - 1]

            print("Invalid number. Please select a listed option.")
            continue

        # Match the full option name, ignoring capitalization.
        exact_matches = [
            choice for choice in choices
            if choice.casefold() == value
        ]

        if len(exact_matches) == 1:
            return exact_matches[0]

        # Allow a first-letter shortcut only if it is unique.
        letter_matches = [
            choice for choice in choices
            if choice.casefold().startswith(value) and len(value) == 1
        ]

        if len(letter_matches) == 1:
            return letter_matches[0]

        if len(letter_matches) > 1:
            print("This letter matches multiple options. Type the full name or option number.")
        else:
            print("Invalid option. Please try again.")


def ask_number(prompt, minimum=0, maximum=None, integer=False):
    """Ask for a valid non-negative number."""
    while True:
        try:
            value = input(f"{prompt}: ").strip()

            if integer:
                number = int(value)
            else:
                number = float(value)

            if number < minimum:
                print(f"Value must be at least {minimum}.")
                continue

            if maximum is not None and number > maximum:
                print(f"Value must not exceed {maximum}.")
                continue

            return number

        except ValueError:
            print("Please enter a valid number.")


def predict_customer():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src/train_model.py")
        return

    model = joblib.load(MODEL_PATH)
    X, _ = load_and_prepare_data()

    print("\nCUSTOMER CHURN PREDICTION")
    print("-" * 40)
    print("Enter the customer's details below.\n")

    gender = ask_choice("Gender", ["Female", "Male"])

    senior = ask_choice(
        "Senior citizen", ["0", "1"]
    )
    senior = int(senior)

    partner = ask_choice("Has partner", ["Yes", "No"])
    dependents = ask_choice("Has dependents", ["Yes", "No"])

    tenure = ask_number(
        "Tenure in months", 0, 72, integer=True
    )

    phone = ask_choice("Phone service", ["Yes", "No"])

    multiple_lines = ask_choice(
        "Multiple lines",
        ["No phone service", "No", "Yes"]
    )

    internet = ask_choice(
        "Internet service",
        ["DSL", "Fiber optic", "No"]
    )

    internet_options = ["No internet service", "No", "Yes"]

    online_security = ask_choice(
        "Online security", internet_options
    )
    online_backup = ask_choice(
        "Online backup", internet_options
    )
    device_protection = ask_choice(
        "Device protection", internet_options
    )
    tech_support = ask_choice(
        "Tech support", internet_options
    )
    streaming_tv = ask_choice(
        "Streaming TV", internet_options
    )
    streaming_movies = ask_choice(
        "Streaming movies", internet_options
    )

    contract = ask_choice(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = ask_choice(
        "Paperless billing", ["Yes", "No"]
    )

    payment = ask_choice(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = ask_number("Monthly charges")
    total_charges = ask_number("Total charges")

    # Start with every feature set to zero.
    # This preserves the exact feature columns used during training.
    
    customer = pd.DataFrame(0.0, index=[0], columns=X.columns)

    # Numeric features
    customer.loc[0, "SeniorCitizen"] = senior
    customer.loc[0, "tenure"] = tenure
    customer.loc[0, "MonthlyCharges"] = monthly_charges
    customer.loc[0, "TotalCharges"] = total_charges

    # Categorical features
    categorical_values = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment
    }

    for column, value in categorical_values.items():
        feature_name = f"{column}_{value}"

        if feature_name in customer.columns:
            customer.loc[0, feature_name] = 1

    # Ensure the model receives features in the correct order.
    customer = customer[X.columns]

    probability = model.predict_proba(customer)[0][1]
    prediction = int(probability >= 0.5)

    print("\nPREDICTION RESULT")
    print("-" * 40)
    print("Predicted class:", "Churn" if prediction else "Stayed")
    print(f"Estimated churn probability: {probability:.2%}")

    if probability >= 0.5:
        print("Interpretation: The model predicts a higher likelihood of churn.")
    else:
        print("Interpretation: The model predicts a higher likelihood of staying.")

    print("\nNote: This is a model estimate, not a guarantee.")


if __name__ == "__main__":
    try:
        predict_customer()
    except (ValueError, KeyError) as error:
        print("\nCould not complete prediction:", error)
