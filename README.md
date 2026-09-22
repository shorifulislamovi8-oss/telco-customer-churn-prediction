# Telco Customer Churn Prediction

## Project Overview

This is a real-world machine learning classification project focused on predicting whether a telecom customer is likely to churn. The objective is to build a reliable predictive model that helps identify customers at risk of leaving the service so that proactive retention strategies can be applied.

The project includes data exploration, preprocessing, feature engineering, model training, evaluation, threshold analysis, and customer-level prediction. It also covers business-oriented insights to interpret model results in a practical telecom context.

Two models were evaluated: Logistic Regression and Random Forest. The workflow demonstrates how to train, compare, and evaluate these models for customer churn prediction.


## Dataset

The dataset used in this project is the Telco Customer Churn dataset. It contains 7,043 customer records and 21 columns in the original dataset.

The target variable is Churn, where Yes indicates that the customer churned and No indicates that the customer did not churn.

The dataset includes customer demographic information, services used, contract details, payment method, tenure, monthly charges, and total charges, making it suitable for real-world customer retention analysis.

## Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Feature Preparation
4. Train-Test Split
5. Preprocessing

   * StandardScaler for numerical features
   * OneHotEncoder for categorical features
6. Model Training

   * Logistic Regression
   * Random Forest
7. Model Evaluation

   * Classification Report
   * Confusion Matrix
   * ROC-AUC
   * Precision-Recall Curve
8. Threshold Analysis
9. Business Insights
10. Customer-Level Churn Prediction
11. Model Saving and Loading with Joblib

## Models

### Logistic Regression

Logistic Regression was used as the primary classification model for customer churn prediction.

To address class imbalance, `class_weight="balanced"` was applied. The model was implemented inside a preprocessing Pipeline so that feature scaling and encoding were handled consistently during training and inference.

### Random Forest

Random Forest was used as a second classification model for comparison with Logistic Regression.

The model was configured with 200 trees using `n_estimators=200` and `class_weight="balanced"` to better handle the imbalanced churn distribution.

The model was trained on the preprocessed and encoded features to evaluate its classification performance.

## Model Evaluation

| Metric             | Logistic Regression | Random Forest |
| ------------------ | ------------------: | ------------: |
| Accuracy           |                0.74 |          0.77 |
| Churn Recall (Yes) |                0.78 |          0.63 |
| Best F1 Score      |               0.615 |         0.620 |
| ROC-AUC            |               0.842 |         0.823 |

Accuracy measures overall correct predictions. Churn Recall measures how many actual churn customers were identified. F1 balances precision and recall, while ROC-AUC measures the model's ability to distinguish between churn and non-churn customers.

Threshold selection can change precision, recall, F1, and business cost. The evaluation is based on the test set, and model selection depends on the specific business objective.

## Threshold Analysis

Different probability thresholds were tested to understand the trade-off between precision, recall, F1 score, and estimated business cost.

For the Logistic Regression model:

* Best business-cost threshold: `0.20`
* Minimum estimated cost: `$7,060`
* Best F1 threshold: `0.30`
* Best F1 score: `0.615`

For the Random Forest model:

* Best business-cost threshold: `0.10`
* Minimum estimated cost: `$8,060`
* Best F1 threshold: `0.35`
* Best F1 score: `0.620`

The business-cost analysis uses hypothetical costs of `$10` for a false positive and `$100` for a false negative. These values are project assumptions and do not represent actual telecom business costs.

## Business Insights

The dataset shows clear churn risk patterns among certain customer segments.

Observed churn rates were:

* Month-to-month contracts: `42.71%`
* One-year contracts: `11.27%`
* Two-year contracts: `2.83%`
* Fiber Optic internet: `41.89%`
* Electronic check payment: `45.29%`
* No OnlineSecurity: `41.77%`
* No TechSupport: `41.64%`
* No OnlineBackup: `39.93%`

A particularly high-risk group was identified consisting of customers with:

* Month-to-month contract
* Fiber Optic internet
* Electronic check payment
* No OnlineSecurity
* No TechSupport

This combined group included 989 customers and had an observed churn rate of `65.82%`.

These findings represent observed associations in the dataset and should not be interpreted as proof of causation.

## Customer-Level Prediction

A new customer example was passed through the trained Logistic Regression pipeline.

The model produced:

* Churn Probability: `89.87%`
* Predicted Churn: `Yes`

This probability represents the model's estimated likelihood based on the customer's input features. It should not be interpreted as certainty.

## How to Run

1. Clone or download the project to your local machine.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment in the project directory.
4. Install the required dependencies:

```
pip install -r requirements.txt
```


```

6. Run the project script:

```bash
python "Real ML Project.py"
```

## Project Structure

```text
day_01/
├── data/
│   └── raw/
├── notebooks/
├── src/
├── tests/
├── .gitignore
├── requirements.txt
├── README.md
└── Real ML Project.py
```

### Folder Description

* `data/raw` contains the raw dataset used for analysis and modeling.
* `notebooks` is for exploratory notebooks and experiments.
* `src` is for source code and reusable project logic.
* `tests` is for testing and validation scripts.
* `requirements.txt` contains the Python dependencies required to run the project.
* `Real ML Project.py` contains the main machine learning workflow.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## Key Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Data Preprocessing
* Classification
* Logistic Regression
* Random Forest
* Class Imbalance Handling
* Threshold Optimization
* Model Evaluation
* ROC-AUC Analysis
* Precision-Recall Analysis
* Business-Oriented ML Analysis
* Customer-Level Prediction
* Model Saving and Loading

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn prediction, from data exploration and preprocessing to model evaluation, threshold analysis, business insights, and customer-level prediction.

The project also demonstrates how machine learning results can be connected to practical business questions rather than focusing only on model accuracy.

