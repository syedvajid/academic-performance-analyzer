"""
Random Forest Regression
Academic Performance Analyzer - Dataset V2
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("data/student_performance_v2.csv")


# ============================================================
# Features and Target
# ============================================================

features = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation"
]

X = df[features]
y = df["total_score"]


# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# Train Random Forest
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# Evaluate
# ============================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# Display Results
# ============================================================

print("=" * 65)
print("Random Forest Regression - Dataset V2")
print("=" * 65)

print(f"\nTraining Samples : {len(X_train):,}")
print(f"Testing Samples  : {len(X_test):,}")

print("\nEvaluation:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# Feature Importance
# ============================================================

print("\nFeature Importance:")

for feature, importance in zip(
    features,
    model.feature_importances_
):
    print(
        f"{feature:30s}: "
        f"{importance:.4f}"
    )


# ============================================================
# Save Model
# ============================================================

model_path = "models/random_forest_model_v2.pkl"

joblib.dump(
    model,
    model_path
)

print(f"\n✅ Model saved:")
print(model_path)


# ============================================================
# Prediction Sanity Tests
# ============================================================

test_cases = pd.DataFrame({
    "weekly_self_study_hours": [
        1,
        1,
        40
    ],

    "attendance_percentage": [
        0,
        100,
        100
    ],

    "class_participation": [
        0,
        10,
        10
    ]
})


test_predictions = model.predict(
    test_cases
)


print("\nPrediction Sanity Tests:")

for i, prediction in enumerate(
    test_predictions,
    start=1
):

    print(
        f"Test {i}: "
        f"Study={test_cases.iloc[i - 1]['weekly_self_study_hours']}, "
        f"Attendance={test_cases.iloc[i - 1]['attendance_percentage']}, "
        f"Participation={test_cases.iloc[i - 1]['class_participation']} "
        f"→ Score={prediction:.2f}"
    )


print("\n" + "=" * 65)
print("Training completed successfully! ✅")
print("=" * 65)