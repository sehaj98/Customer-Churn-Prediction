
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def load_and_prepare_data():
    """Load the dataset and prepare features and target."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Remove customer identifier
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with missing values
    df = df.dropna().copy()

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Encode categorical features
    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=int
    )

    return X, y


def train_models():
    """Train and save Logistic Regression and Random Forest."""

    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # StandardScaler helps Logistic Regression handle
    # numerical features on different scales.
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    models = {
        "logistic_regression_model.joblib": logistic_model,
        "random_forest_model.joblib": random_forest_model
    }

    for filename, model in models.items():
        model.fit(X_train, y_train)

        output_path = MODEL_DIR / filename
        joblib.dump(model, output_path)

        print(f"Saved model: {output_path.name}")

    print("\nTraining completed successfully!")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Number of features:", X.shape[1])
    print("\nTarget classes: 0 = Stayed, 1 = Churned")


if __name__ == "__main__":
    train_models()
