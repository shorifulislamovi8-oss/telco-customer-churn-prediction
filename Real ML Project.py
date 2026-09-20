from pathlib import Path

import pandas as pd


# Dataset location (the CSV file is kept unchanged in data/raw).
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_dataset():
    """Load the original Telco Customer Churn dataset without modifying it."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    return pd.read_csv(DATASET_PATH)


if __name__ == "__main__":
    dataset = load_dataset()
    print("Dataset loaded successfully.")
    print(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")

    print(dataset.info())

    print(dataset["TotalCharges"].unique()[:20])
print(dataset["TotalCharges"].str.strip().eq("").sum())
print(dataset.loc[dataset["TotalCharges"].str.strip() == "", ["tenure", "MonthlyCharges", "TotalCharges"]])

dataset["TotalCharges"] = dataset["TotalCharges"].str.strip()
dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)

print(dataset["TotalCharges"].isna().sum())
print(dataset["TotalCharges"].dtype)

print(dataset.isnull().sum())

print(dataset["Churn"].value_counts())

print(dataset["Churn"].value_counts(normalize=True) * 100)

print(pd.crosstab(dataset["Contract"], dataset["Churn"], normalize="index") * 100)

print(pd.crosstab(
    dataset["InternetService"],
    dataset["Churn"],
    normalize="index"
) * 100)

print(dataset["tenure"].describe())

dataset["TenureGroup"] = pd.cut(
    dataset["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

print(dataset["TenureGroup"].value_counts().sort_index())

print(pd.crosstab(
    dataset["TenureGroup"],
    dataset["Churn"],
    normalize="index"
) * 100)

print(dataset["MonthlyCharges"].describe())

dataset["ChargeGroup"] = pd.cut(
    dataset["MonthlyCharges"],
    bins=[0, 40, 80, float("inf")],
    labels=["Low", "Medium", "High"]
)

print(dataset["ChargeGroup"].value_counts())

print(pd.crosstab(
    dataset["ChargeGroup"],
    dataset["Churn"],
    normalize="index"
) * 100)

print(dataset["PaymentMethod"].value_counts())

print(pd.crosstab(
    dataset["PaymentMethod"],
    dataset["Churn"],
    normalize="index"
) * 100)

print(dataset["SeniorCitizen"].value_counts())

print(pd.crosstab(
    dataset["SeniorCitizen"],
    dataset["Churn"],
    normalize="index"
) * 100)

print(dataset.groupby("Churn")["tenure"].mean())
print(dataset.groupby("Churn")["MonthlyCharges"].mean())
print(dataset.groupby("Churn")["TotalCharges"].mean())

import matplotlib.pyplot as plt

#dataset.boxplot(column="tenure", by="Churn")

#plt.title("Tenure vs Churn")
#plt.suptitle("")
#plt.xlabel("Churn")
#plt.ylabel("Tenure (Months)")


#dataset.boxplot(column="MonthlyCharges", by="Churn")

#plt.title("Monthly Charges vs Churn")
#plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")


print(dataset[["tenure", "MonthlyCharges", "TotalCharges"]].corr())

X = dataset.drop(columns=["Churn", "customerID"])
y = dataset["Churn"]

print("X shape:", X.shape)
print("y shape:", y.shape)

X = dataset.drop(
    columns=["Churn", "customerID", "TenureGroup", "ChargeGroup"]
)

y = dataset["Churn"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print(X.dtypes)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

categorical_features = X.select_dtypes(include=["str"]).columns

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

print(preprocessor)

X_encoded = preprocessor.fit_transform(X)

print("Encoded shape:", X_encoded.shape)

#model training and evaluation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
       ("model", LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
))
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

print(pipeline)



pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", (y_pred == y_test).mean())

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))


print(numeric_features)

from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred, pos_label="Yes")

print("Recall:", recall)

from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, pos_label="Yes")

print("Precision:", precision)

from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred, pos_label="Yes")

print("Recall:", recall)

from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, pos_label="Yes")

print("F1-score:", f1)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

y_proba = pipeline.predict_proba(X_test)

print(y_proba[:5])

yes_proba = y_proba[:, 1]

print(yes_proba[:5])

y_pred_30 = ["Yes" if p >= 0.30 else "No" for p in yes_proba]
print(y_pred_30[:5])

from sklearn.metrics import recall_score

recall_30 = recall_score(
    y_test,
    y_pred_30,
    pos_label="Yes"
)

print("Recall at 0.30:", recall_30)

precision_30 = precision_score(
    y_test,
    y_pred_30,
    pos_label="Yes"
)

print("Precision at 0.30:", precision_30)

f1_30 = f1_score(
    y_test,
    y_pred_30,
    pos_label="Yes"
)

print("F1-score at 0.30:", f1_30)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("Thresholds:", thresholds)


threshold = 0.9

y_pred_threshold = [
    "Yes" if p >= threshold else "No"
    for p in yes_proba
]

print(y_pred_threshold[:40])

precision_90 = precision_score(
    y_test,
    y_pred_threshold,
    pos_label="Yes"
)

recall_90 = recall_score(
    y_test,
    y_pred_threshold,
    pos_label="Yes"
)

f1_90 = f1_score(
    y_test,
    y_pred_threshold,
    pos_label="Yes"
)

print("Precision at 0.90:", precision_90)
print("Recall at 0.90:", recall_90)
print("F1-score at 0.90:", f1_90)


results = pd.DataFrame({
    "Threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "Precision": [0.405963, 0.467153, 0.519337, 0.568182, 0.657233, 0.717703, 0.741935, 0.769231, 0.0],
    "Recall": [0.946524, 0.855615, 0.754011, 0.668449, 0.558824, 0.401070, 0.184492, 0.026738, 0.0],
    "F1": [0.568218, 0.604344, 0.615049, 0.614251, 0.604046, 0.514580, 0.295503, 0.051680, 0.0]
})

print(results)

import matplotlib.pyplot as plt

#plt.plot(results["Threshold"], results["Precision"], marker="o", label="Precision")
#plt.plot(results["Threshold"], results["Recall"], marker="o", label="Recall")
#plt.plot(results["Threshold"], results["F1"], marker="o", label="F1")

#plt.xlabel("Threshold")
#plt.ylabel("Score")
#plt.title("Precision, Recall and F1 vs Threshold")
#plt.legend()
#plt.grid()

#plt.show()

cm_30 = confusion_matrix(y_test, y_pred_30)

print("Confusion Matrix at 0.30:")
print(cm_30)

threshold = 0.2

y_pred_20 = [
    "Yes" if p >= threshold else "No"
    for p in yes_proba
]

cm_20 = confusion_matrix(y_test, y_pred_20)

print("Confusion Matrix at 0.20:")
print(cm_20)

threshold = 0.1

y_pred_10 = [
    "Yes" if p >= threshold else "No"
    for p in yes_proba
]

cm_10 = confusion_matrix(y_test, y_pred_10)

print("Confusion Matrix at 0.10:")
print(cm_10)

fp_cost = 10
fn_cost = 100

#results["FP"] = [
    #518, 365, 261, 217, 109, 0, 0, 0, 0
#]

#results["FN"] = [
 #   20, 54, 92, 125, 165, 224, 305, 364, 374
#]

#results["TotalCost"] = (
 #   results["FP"] * fp_cost
  #  + results["FN"] * fn_cost
#)

#print(results[["Threshold", "FP", "FN", "TotalCost"]])

fp_list = []
fn_list = []

for threshold in results["Threshold"]:

    y_pred = [
        "Yes" if p >= threshold else "No"
        for p in yes_proba
    ]

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    fp_list.append(fp)
    fn_list.append(fn)


results["FP"] = fp_list
results["FN"] = fn_list

print(results[["Threshold", "FP", "FN"]])

fp_cost = 10
fn_cost = 100

results["TotalCost"] = (
    results["FP"] * fp_cost
    + results["FN"] * fn_cost
)

print(results[["Threshold", "FP", "FN", "TotalCost"]])

best_threshold = results.loc[
    results["TotalCost"].idxmin()
]

print(best_threshold)

plt.plot(
    results["Threshold"],
    results["TotalCost"],
    marker="o"
)

plt.xlabel("Threshold")
plt.ylabel("Total Cost")
plt.title("Business Cost vs Threshold")
plt.grid()

#plt.show()

plt.plot(
    results["Threshold"],
    results["TotalCost"],
    marker="o"
)

print("Best Threshold:", best_threshold["Threshold"])
print("Minimum Cost:", best_threshold["TotalCost"])

best_f1 = results.loc[
    results["F1"].idxmax()
]

print("Best F1 Threshold:", best_f1["Threshold"])
print("Highest F1:", best_f1["F1"])


#RandomForestClassifier import
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
rf_model.fit(X_train_encoded, y_train)

y_pred_rf = rf_model.predict(X_test_encoded)

print(classification_report(y_test, y_pred_rf))

y_prob_rf = rf_model.predict_proba(X_test_encoded)[:, 1]

threshold = 0.30

#y_pred_rf_03 = (y_prob_rf >= threshold).astype(int)
import numpy as np
y_pred_rf_03 = np.where(y_prob_rf >= threshold, "Yes", "No")
print(classification_report(y_test, y_pred_rf_03))


print(classification_report(y_test, y_pred_rf_03))

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf_03,
    labels=["No", "Yes"]
)

print("Confusion Matrix:")
print(cm_rf)

tn, fp, fn, tp = cm_rf.ravel()

total_cost_rf = (fp * 10) + (fn * 100)

print("FP:", fp)
print("FN:", fn)
print("Total Cost:", total_cost_rf)



for threshold in np.arange(0.1, 1.0, 0.1):

    y_pred = np.where(
        y_prob_rf >= threshold,
        "Yes",
        "No"
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=["No", "Yes"]
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    cost = (fp * 10) + (fn * 100)

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f} | "
        f"FP: {fp} | FN: {fn} | "
        f"Cost: ${cost}"
    )

    best_f1 = 0
best_threshold = 0

for threshold in np.arange(0.1, 1.0, 0.01):

    y_pred = np.where(
        y_prob_rf >= threshold,
        "Yes",
        "No"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    f1 = report["Yes"]["f1-score"]

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("Best Threshold:", round(best_threshold, 2))
print("Highest F1:", round(best_f1, 4))

threshold = 0.35

y_pred_rf_best = np.where(
    y_prob_rf >= threshold,
    "Yes",
    "No"
)

print(classification_report(y_test, y_pred_rf_best))

cm_rf_best = confusion_matrix(
    y_test,
    y_pred_rf_best,
    labels=["No", "Yes"]
)

tn, fp, fn, tp = cm_rf_best.ravel()

cost_rf_best = (fp * 10) + (fn * 100)

print("FP:", fp)
print("FN:", fn)
print("Business Cost:", cost_rf_best)

y_pred_lr_default = pipeline.predict(X_test)

print(classification_report(y_test, y_pred_lr_default))

feature_importance = rf_model.feature_importances_

print(feature_importance)


feature_names = preprocessor.get_feature_names_out()

print("Number of features:", len(feature_names))
print(feature_names)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(10))

contract_churn = pd.crosstab(
    dataset["Contract"],
    dataset["Churn"],
    normalize="index"
) * 100

print(contract_churn)

security_churn = pd.crosstab(
    dataset["OnlineSecurity"],
    dataset["Churn"],
    normalize="index"
) * 100

print(security_churn)

techsupport_churn = pd.crosstab(
    dataset["TechSupport"],
    dataset["Churn"],
    normalize="index"
) * 100

print(techsupport_churn)

payment_churn = pd.crosstab(
    dataset["PaymentMethod"],
    dataset["Churn"],
    normalize="index"
) * 100

print(payment_churn)

internet_churn = pd.crosstab(
    dataset["InternetService"],
    dataset["Churn"],
    normalize="index"
) * 100

print(internet_churn)

backup_churn = pd.crosstab(
    dataset["OnlineBackup"],
    dataset["Churn"],
    normalize="index"
) * 100

print(backup_churn)

risk_group = dataset[
    (dataset["Contract"] == "Month-to-month") &
    (dataset["OnlineSecurity"] == "No") &
    (dataset["TechSupport"] == "No") &
    (dataset["PaymentMethod"] == "Electronic check") &
    (dataset["InternetService"] == "Fiber optic")
]

print("Customers:", len(risk_group))
print("Churn rate:", (risk_group["Churn"] == "Yes").mean() * 100)

normal_churn = (dataset["Churn"] == "Yes").mean() * 100
risk_churn = (risk_group["Churn"] == "Yes").mean() * 100

print("Overall churn rate:", normal_churn)
print("Combined risk group churn rate:", risk_churn)



import matplotlib.pyplot as plt

labels = ["Overall Customers", "Combined Risk Group"]
churn_rates = [normal_churn, risk_churn]

plt.bar(labels, churn_rates)
plt.ylabel("Churn Rate (%)")
plt.title("Overall vs Combined Risk Group Churn Rate")
plt.ylim(0, 100)
#plt.show()

contract_summary = dataset.groupby("Contract").agg(
    Customers=("Churn", "count"),
    Churn_Rate=("Churn", lambda x: (x == "Yes").mean() * 100)
)

print(contract_summary)

insight_summary = pd.DataFrame({
    "Feature": [
        "Contract",
        "OnlineSecurity",
        "TechSupport",
        "PaymentMethod",
        "InternetService",
        "OnlineBackup"
    ],
    "High_Churn_Group": [
        "Month-to-month",
        "No",
        "No",
        "Electronic check",
        "Fiber optic",
        "No"
    ],
    "Churn_Rate": [
        42.71,
        41.77,
        41.64,
        45.29,
        41.89,
        39.93
    ]
})

print(insight_summary)


from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test, yes_proba)

print("Logistic Regression ROC-AUC:", lr_auc)


rf_auc = roc_auc_score(y_test, y_prob_rf)

print("Random Forest ROC-AUC:", rf_auc)


from sklearn.metrics import ConfusionMatrixDisplay

y_pred_lr = pipeline.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr
)

#plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf
)

#plt.show()

rf_pred_default = rf_model.predict(X_test_encoded)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred_default
)

#plt.show()

from sklearn.metrics import precision_recall_curve

precision_lr, recall_lr, thresholds_lr = precision_recall_curve(
    y_test,
    yes_proba,
    pos_label="Yes"
)

plt.figure()
plt.plot(recall_lr, precision_lr)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Logistic Regression - Precision-Recall Curve")
plt.show()

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test,
    y_prob_rf,
    pos_label="Yes"
)

plt.figure()
plt.plot(recall_rf, precision_rf)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest - Precision-Recall Curve")
plt.show()



# nwe castomar 
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0,
    "TotalCharges": 450.0
}])

print(new_customer)

new_probability = pipeline.predict_proba(new_customer)[0, 1]

print("Churn Probability:", new_probability)

new_prediction = pipeline.predict(new_customer)[0]

print("Predicted Churn:", new_prediction)

import joblib

joblib.dump(
    pipeline,
    "telco_churn_model.pkl"
)

print("Model saved successfully!")

import joblib

loaded_model = joblib.load("telco_churn_model.pkl")

print("Model loaded successfully!")

loaded_probability = loaded_model.predict_proba(new_customer)[0, 1]

loaded_prediction = loaded_model.predict(new_customer)[0]

print("Churn Probability:", loaded_probability)
print("Predicted Churn:", loaded_prediction)