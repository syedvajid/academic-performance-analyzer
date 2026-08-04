import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("data/student_performance.csv")

# ---------------------------------------
# Features and Target
# ---------------------------------------
X = df[
    [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation"
    ]
]

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
# Train Model
# ---------------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# ---------------------------------------
# Save Model
# ---------------------------------------
joblib.dump(model, "models/student_model.pkl")

# ---------------------------------------
# Save Test Dataset
# ---------------------------------------
test_data = X_test.copy()
test_data["total_score"] = y_test

test_data.to_csv("data/test_data.csv", index=False)

print("✅ Model trained successfully!")
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")