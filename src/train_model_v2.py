import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------
# Load New Dataset
# ---------------------------------------

df = pd.read_csv("data/student_performance_v2.csv")


# ---------------------------------------
# Features and Target
# ---------------------------------------

features = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation"
]

X = df[features]
y = df["total_score"]


# ---------------------------------------
# Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ---------------------------------------
# Train Linear Regression
# ---------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ---------------------------------------
# Evaluate
# ---------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


print("=" * 60)
print("Linear Regression - Dataset V2")
print("=" * 60)

print(f"\nTraining Samples : {len(X_train):,}")
print(f"Testing Samples  : {len(X_test):,}")

print("\nModel Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature:30s}: {coefficient:.6f}")

print(f"\nIntercept: {model.intercept_:.6f}")

print("\nEvaluation:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ---------------------------------------
# Save V2 Model Separately
# ---------------------------------------

joblib.dump(
    model,
    "models/student_model_v2.pkl"
)

print("\n✅ Model saved:")
print("models/student_model_v2.pkl")


# ---------------------------------------
# Test Extreme / Important Cases
# ---------------------------------------

test_cases = pd.DataFrame({
    "weekly_self_study_hours": [1, 1, 40],
    "attendance_percentage": [0, 100, 100],
    "class_participation": [0, 10, 10]
})

test_predictions = model.predict(test_cases)

print("\nPrediction Sanity Tests:")

for i, prediction in enumerate(test_predictions, start=1):
    print(
        f"Test {i}: "
        f"Study={test_cases.iloc[i-1, 0]}, "
        f"Attendance={test_cases.iloc[i-1, 1]}, "
        f"Participation={test_cases.iloc[i-1, 2]} "
        f"→ Score={prediction:.2f}"
    )