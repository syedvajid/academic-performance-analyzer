import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/student_performance.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = df[
    [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation",
    ]
]

y = df["total_score"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# -----------------------------
# Train Decision Tree
# -----------------------------
model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
)

model.fit(X_train, y_train)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "models/decision_tree_model.pkl")

print("✅ Decision Tree trained successfully!")
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print("\nModel Parameters")
print("---------------------------")
print(f"Max Depth         : {model.max_depth}")
print(f"Min Samples Split : {model.min_samples_split}")
print(f"Min Samples Leaf  : {model.min_samples_leaf}")

