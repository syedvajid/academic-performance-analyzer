import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Analysis",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================

DATASET_PATH = "data/student_performance_v2.csv"

LINEAR_MODEL_PATH = "models/student_model_v2.pkl"

RANDOM_FOREST_MODEL_PATH = "models/random_forest_model_v2.pkl"

FEATURES = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation"
]

TARGET = "total_score"

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

X = df[FEATURES]
y = df[TARGET]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# LOAD MODELS
# ============================================================

linear_model = joblib.load(
    LINEAR_MODEL_PATH
)

random_forest_model = joblib.load(
    RANDOM_FOREST_MODEL_PATH
)

# ============================================================
# TEST SET PREDICTIONS
# ============================================================

linear_pred = linear_model.predict(X_test)

rf_pred = random_forest_model.predict(X_test)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse
    }


# ============================================================
# CALCULATE MODEL METRICS
# ============================================================

linear_metrics = calculate_metrics(
    y_test,
    linear_pred
)

rf_metrics = calculate_metrics(
    y_test,
    rf_pred
)

# ============================================================
# DETERMINE BEST MODEL
# ============================================================

# Higher R² is better.
# Lower MAE and RMSE are better.
#
# R² is used as the primary selection criterion.

if linear_metrics["R²"] > rf_metrics["R²"]:

    best_model_name = "Linear Regression"

    best_model = linear_model

    best_pred = linear_pred

    best_metrics = linear_metrics

else:

    best_model_name = "Random Forest"

    best_model = random_forest_model

    best_pred = rf_pred

    best_metrics = rf_metrics


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Model Analysis Dashboard"
)

st.write(
    "Evaluation and comparison of the machine learning models "
    "used for academic performance prediction."
)

st.divider()

# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader(
    "📚 Dataset Information"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "👨‍🎓 Students",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "🧪 Training Samples",
        f"{len(X_train):,}"
    )

with col3:

    st.metric(
        "🔬 Testing Samples",
        f"{len(X_test):,}"
    )

with col4:

    st.metric(
        "🎯 Features",
        len(FEATURES)
    )

st.divider()

# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader(
    "🏆 Model Comparison"
)

comparison = pd.DataFrame({

    "Metric": [
        "R² Score",
        "MAE",
        "RMSE",
        "MSE"
    ],

    "Linear Regression": [
        linear_metrics["R²"],
        linear_metrics["MAE"],
        linear_metrics["RMSE"],
        linear_metrics["MSE"]
    ],

    "Random Forest": [
        rf_metrics["R²"],
        rf_metrics["MAE"],
        rf_metrics["RMSE"],
        rf_metrics["MSE"]
    ]

})

st.dataframe(
    comparison.style.format({
        "Linear Regression": "{:.4f}",
        "Random Forest": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ============================================================
# BEST MODEL PERFORMANCE
# ============================================================

st.subheader(
    f"🥇 Best Model: {best_model_name}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🎯 R² Score",
        f"{best_metrics['R²']:.4f}"
    )

with col2:

    st.metric(
        "📉 MAE",
        f"{best_metrics['MAE']:.4f}"
    )

with col3:

    st.metric(
        "📈 RMSE",
        f"{best_metrics['RMSE']:.4f}"
    )

with col4:

    st.metric(
        "📊 MSE",
        f"{best_metrics['MSE']:.4f}"
    )

st.success(
    f"🏆 {best_model_name} is the best-performing model "
    f"on the unseen test dataset."
)

st.divider()

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    f"🎯 Actual vs Predicted Scores - {best_model_name}"
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.scatter(
    y_test,
    best_pred,
    alpha=0.4
)

min_value = min(
    y_test.min(),
    best_pred.min()
)

max_value = max(
    y_test.max(),
    best_pred.max()
)

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

ax.set_xlabel(
    "Actual Score"
)

ax.set_ylabel(
    "Predicted Score"
)

ax.set_title(
    f"{best_model_name}: Actual vs Predicted"
)

st.pyplot(
    fig
)

st.info(
    "Points closer to the diagonal line indicate more accurate predictions."
)

st.divider()

# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

st.subheader(
    f"📉 Residual Analysis - {best_model_name}"
)

residuals = y_test - best_pred

fig2, ax2 = plt.subplots(
    figsize=(9, 6)
)

ax2.scatter(
    best_pred,
    residuals,
    alpha=0.4
)

ax2.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

ax2.set_xlabel(
    "Predicted Score"
)

ax2.set_ylabel(
    "Residual"
)

ax2.set_title(
    f"{best_model_name}: Residual Plot"
)

st.pyplot(
    fig2
)

st.info(
    "A good regression model should have residuals reasonably "
    "distributed around zero without a strong systematic pattern."
)

st.divider()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Feature Importance"
)

importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": random_forest_model.feature_importances_

})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

st.dataframe(
    importance_df.style.format({
        "Importance": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

fig3, ax3 = plt.subplots(
    figsize=(9, 5)
)

ax3.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax3.set_xlabel(
    "Feature"
)

ax3.set_ylabel(
    "Importance"
)

ax3.set_title(
    "Random Forest Feature Importance"
)

plt.xticks(
    rotation=20
)

st.pyplot(
    fig3
)

# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "🧠 Model Interpretation"
)

top_feature = importance_df.iloc[0]["Feature"]

top_importance = importance_df.iloc[0]["Importance"]

st.write(
    f"According to the Random Forest model, "
    f"**{top_feature}** is the most influential feature, "
    f"with a feature importance of **{top_importance:.4f}**."
)

st.write(
    "The model uses weekly self-study hours, attendance percentage, "
    "and class participation together to estimate a student's total score."
)

# ============================================================
# LINEAR REGRESSION COEFFICIENTS
# ============================================================

st.subheader(
    "📐 Linear Regression Coefficients"
)

if hasattr(linear_model, "coef_"):

    coefficient_df = pd.DataFrame({

        "Feature": FEATURES,

        "Coefficient": linear_model.coef_

    })

    coefficient_df["Absolute Impact"] = (
        coefficient_df["Coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    st.dataframe(
        coefficient_df.style.format({
            "Coefficient": "{:.4f}",
            "Absolute Impact": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Linear regression coefficients show how the predicted score "
        "changes with each feature while the other features are held constant."
    )

# ============================================================
# MODEL VERDICT
# ============================================================

st.divider()

st.subheader(
    "⚖️ Model Verdict"
)

if best_model_name == "Linear Regression":

    st.success(
        "Linear Regression wins because it provides higher predictive "
        "accuracy while remaining simpler and easier to interpret."
    )

    st.write(
        "Since the relationship between the academic features and "
        "total score is largely linear, the more complex Random Forest "
        "model does not provide an advantage on this dataset."
    )

else:

    st.success(
        "Random Forest wins because it provides better predictive "
        "performance on the unseen test dataset."
    )

    st.write(
        "The Random Forest model is able to capture relationships "
        "that the linear model cannot represent as effectively."
    )

# ============================================================
# FINAL MODEL SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📋 Final Model Summary"
)

summary = pd.DataFrame({

    "Property": [
        "Dataset",
        "Number of Students",
        "Features",
        "Target",
        "Best Model",
        "R² Score",
        "MAE",
        "RMSE"
    ],

    "Value": [
        "Student Performance V2",
        str(len(df)),
        str(len(FEATURES)),
        "total_score",
        str(best_model_name),
        f"{best_metrics['R²']:.4f}",
        f"{best_metrics['MAE']:.4f}",
        f"{best_metrics['RMSE']:.4f}"
    ]

})

# Explicitly force every Value to string.
# This prevents PyArrow/Streamlit mixed-type errors.

summary["Value"] = summary["Value"].astype(str)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

st.success(
    f"✅ Model analysis completed successfully! "
    f"Best model: {best_model_name}"
)