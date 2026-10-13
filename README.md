
# Customer Churn Prediction System

A machine learning project that predicts whether a telecommunications customer is likely to leave a service. It compares Logistic Regression and Random Forest models and evaluates their performance using classification metrics.

## Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who may leave can help businesses investigate customer concerns and plan retention strategies.

This project uses customer information such as tenure, contract type, internet service, payment method, and monthly charges to predict churn.

**Project type:** Supervised Machine Learning — Binary Classification

## Objectives

- Prepare and clean customer data.
- Explore and encode categorical features.
- Train Logistic Regression and Random Forest models.
- Compare model performance using evaluation metrics.
- Predict churn probability for an individual customer.
- Test data preparation, trained models, and prediction behaviour.

## Dataset

The project uses the Telco Customer Churn dataset.

The target column is `Churn`:
- `Yes`: The customer churned.
- `No`: The customer stayed.

The preprocessing pipeline removes `customerID`, converts `TotalCharges` to numeric values, removes rows with missing values, and applies one-hot encoding to categorical features.

Dataset source: IBM Telco Customer Churn sample dataset. Add the original dataset URL here if you have verified the exact source.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- pytest
- Jupyter Notebook

## Project Structure

```text
Customer-Churn-Prediction/
├── data/
├── models/
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
│   └── script_model_evaluation.csv
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_churn.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_predict_churn.py
├── README.md
└── requirements.txt
```

## Setup and Installation

### 1. Clone the repository


### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## How to Run

Run the commands from the project root directory.

### Train the models

```bash
python src/train_model.py
```

This trains Logistic Regression and Random Forest and saves their model files in `models/`.

### Evaluate the models

```bash
python src/evaluate_model.py
```

This prints classification reports and confusion matrices and saves the comparison to `reports/script_model_evaluation.csv`.

### Predict churn for a customer

```bash
python src/predict_churn.py
```

Follow the prompts to enter the customer's details.

### Run automated tests

```bash
python -m pytest -v
```

The current test suite contains 14 tests covering data preparation, model loading and predictions, and prediction input/output behaviour.

## Model Evaluation Results

Evaluation was performed on a held-out test set of 1,407 customers.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.38% | 75.27% |
| Churn precision | 65% | 52% |
| Churn recall | 57% | 78% |
| Churn F1-score | 0.61 | 0.62 |
| ROC-AUC | 0.8357 | 0.8347 |

These results reflect the current evaluation run. Re-run the evaluation script to verify results after changing the dataset or training code.

## Key Findings

- Logistic Regression achieved higher accuracy and slightly higher ROC-AUC.
- Random Forest identified a greater proportion of actual churn customers.
- Random Forest also produced more false-positive churn predictions.
- The best model depends on the business cost of missed churn customers versus unnecessary retention offers.

## Limitations

- Predictions are estimates, not guarantees of future customer behaviour.
- Model performance may change on new or different customer data.
- The current workflow uses a fixed classification threshold and a single held-out test split.
- The current command-line interface is intended as a project demonstration, not a production deployment.
- Data preprocessing and model training would benefit from a unified, saved pipeline to ensure consistent transformations.

## Future Improvements

- Add exploratory data analysis visualizations.
- Tune classification thresholds based on business costs.
- Use cross-validation and systematic hyperparameter tuning.
- Improve reproducibility by saving preprocessing and modelling together.
- Add a Streamlit interface or prediction API.
- Track model performance on new data.

## Disclaimer

This project is for educational and portfolio purposes. Predictions should not be treated as definitive assessments of individual customers.

## Author

Add your name and links to your GitHub and LinkedIn profiles here.
