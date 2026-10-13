
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from train_model import load_and_prepare_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def evaluate_models():
    """Evaluate saved models using the same reproducible test split."""

    X, y = load_and_prepare_data()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_files = {
        "Logistic Regression": "logistic_regression_model.joblib",
        "Random Forest": "random_forest_model.joblib"
    }

    results = []

    for model_name, filename in model_files.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Run python src/train_model.py first."
            )

        model = joblib.load(model_path)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        print("\n" + "=" * 55)
        print(model_name)
        print("=" * 55)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Stayed", "Churned"],
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ))

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "ROC_AUC": roc_auc
        })

    results_df = pd.DataFrame(results)
    output_path = REPORT_DIR / "script_model_evaluation.csv"
    results_df.to_csv(output_path, index=False)

    print("\nEvaluation summary:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    evaluate_models()
