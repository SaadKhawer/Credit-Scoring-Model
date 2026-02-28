# ==============================
# CREDIT SCORING MODEL PROJECT
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("credit_data.csv")

print("First 5 rows:")
print(df.head())

# ==============================
# 2. DATA PREPROCESSING
# ==============================

# Handle missing values
df = df.dropna()

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Feature Engineering (Debt-to-Income Ratio)
if "Debt" in df.columns and "Income" in df.columns:
    df["DTI"] = df["Debt"] / df["Income"]

# ==============================
# 3. SPLIT FEATURES & TARGET
# ==============================

X = df.drop("Credit_Status", axis=1)
y = df["Credit_Status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. FEATURE SCALING
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 5. TRAIN MODELS
# ==============================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob)
    }

# ==============================
# 6. PRINT RESULTS
# ==============================

for model_name, metrics in results.items():
    print("\n==============================")
    print(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# ==============================
# 7. CONFUSION MATRIX (Random Forest)
# ==============================

best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 8. ROC CURVE
# ==============================

y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ==============================
# 9. FEATURE IMPORTANCE
# ==============================

if isinstance(best_model, RandomForestClassifier):
    importances = best_model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Important Features:")
    print(feature_importance_df.head())