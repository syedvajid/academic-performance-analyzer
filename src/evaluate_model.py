import pandas as pd
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ---------------------------------------
# Load Test Dataset
# ---------------------------------------
df = pd.read_csv("data/test_data.csv")

X_test = df[
    [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation"
    ]
]

y_test = df["total_score"]

# ---------------------------------------
# Load Model
# ---------------------------------------
model = joblib.load("models/student_model.pkl")

# ---------------------------------------
# Predictions
# ---------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------
# Metrics
# ---------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")