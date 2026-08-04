import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# ---------------------------------------------
# Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="Model Analysis",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------
df = pd.read_csv("data/student_performance_sample.csv")

X = df[
    [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation",
    ]
]

y = df["total_score"]

# ---------------------------------------------
# Load Trained Model
# ---------------------------------------------
model = joblib.load("models/student_model.pkl")

# ---------------------------------------------
# Predictions
# ---------------------------------------------
y_pred = model.predict(X)

# ---------------------------------------------
# Evaluation Metrics
# ---------------------------------------------
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5

# ---------------------------------------------
# Title
# ---------------------------------------------
st.title("📊 Model Analysis Dashboard")

st.write(
    "Performance evaluation of the trained **Linear Regression** model."
)

st.divider()

# ---------------------------------------------
# Metrics
# ---------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "🎯 R² Score",
        f"{r2:.4f}"
    )

    st.metric(
        "📉 Mean Absolute Error (MAE)",
        f"{mae:.4f}"
    )

with col2:
    st.metric(
        "📈 Root Mean Squared Error (RMSE)",
        f"{rmse:.4f}"
    )

    st.metric(
        "📊 Mean Squared Error (MSE)",
        f"{mse:.4f}"
    )

st.divider()

# ---------------------------------------------
# Actual vs Predicted Plot
# ---------------------------------------------
st.subheader("📈 Actual vs Predicted Scores")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    y,
    y_pred,
    alpha=0.4
)

ax.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--",
    linewidth=2
)

ax.set_xlabel("Actual Score")
ax.set_ylabel("Predicted Score")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)

st.info(
    "The closer the points are to the red dashed line, the better the model predicts."
)

st.divider()

# ---------------------------------------------
# Residual Plot
# ---------------------------------------------
st.subheader("📉 Residual Plot")

residuals = y - y_pred

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(
    y_pred,
    residuals,
    alpha=0.4
)

ax2.axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2
)

ax2.set_xlabel("Predicted Score")
ax2.set_ylabel("Residual")
ax2.set_title("Residual Plot")

st.pyplot(fig2)

st.info(
    "Residuals should be randomly distributed around zero. Random scatter indicates that the regression model is fitting the data reasonably well."
)

st.divider()

# ---------------------------------------------
# Model Summary
# ---------------------------------------------
st.subheader("📋 Model Summary")

summary = pd.DataFrame({
    "Metric": [
        "R² Score",
        "MAE",
        "MSE",
        "RMSE",
        "Model"
    ],
    "Value": [
        f"{r2:.4f}",
        f"{mae:.4f}",
        f"{mse:.4f}",
        f"{rmse:.4f}",
        "Linear Regression"
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

st.success("✅ Model evaluation completed successfully.")