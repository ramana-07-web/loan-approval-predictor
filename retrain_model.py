"""
Retrain script — Fixes the notebook issues and exports the best model.

Issues fixed:
1. Correct cell execution order (Dependents '3+' replaced BEFORE encoding/splitting)
2. Uses modern pandas syntax (no deprecated inplace on column slices)
3. Uses separate LabelEncoder per column (so inverse_transform works)
4. Exports the GridSearchCV best model (not the manually tuned one)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =============================================================================
# 1. Load data
# =============================================================================
df = pd.read_csv("loan_data.csv")
print("Dataset shape:", df.shape)
print(df.head())
print()

# =============================================================================
# 2. Handle missing values (modern syntax, no deprecated inplace)
# =============================================================================
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

print("Missing values after fill:")
print(df.isnull().sum())
print()

# =============================================================================
# 3. Fix Dependents '3+' BEFORE encoding (this was the notebook's bug)
# =============================================================================
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = pd.to_numeric(df['Dependents'])

# =============================================================================
# 4. Encode categorical variables (separate encoder per column)
# =============================================================================
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print()

# =============================================================================
# 5. Split features and target
# =============================================================================
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

print("Feature dtypes:")
print(X.dtypes)
print()

# =============================================================================
# 6. Train/test split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================================
# 7. Feature scaling
# =============================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# 8. Train Logistic Regression (baseline)
# =============================================================================
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_lr)}")

# =============================================================================
# 9. Train Random Forest with GridSearchCV (best model)
# =============================================================================
params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print("=" * 60)
print("RANDOM FOREST (GridSearchCV) RESULTS")
print("=" * 60)
print(f"Best Params: {grid.best_params_}")
print(f"Best CV Score: {grid.best_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_best)}")

# =============================================================================
# 10. Export the BEST model and scaler
# =============================================================================
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("=" * 60)
print("✅ Exported best_model → model.pkl")
print("✅ Exported scaler → scaler.pkl")
print("=" * 60)
