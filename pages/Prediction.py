import streamlit as st
import joblib
import pandas as pd

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Prediction",
    page_icon="🎯",
    layout="wide"
)

# ----------------------------------------
# Load Trained Model
# ----------------------------------------
model = joblib.load("models/student_model.pkl")

# ----------------------------------------
# Title
# ----------------------------------------
st.title("🎯 Student Score Prediction")

st.write(
    "Enter the student's details below and click **Predict Score**."
)

st.divider()

# ----------------------------------------
# Input Fields
# ----------------------------------------
study_hours = st.number_input(
    "📚 Weekly Self Study Hours",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.5
)

attendance = st.number_input(
    "📅 Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

participation = st.number_input(
    "🙋 Class Participation",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

st.write("")

# ----------------------------------------
# Prediction
# ----------------------------------------
if st.button("🚀 Predict Score", use_container_width=True):

    input_df = pd.DataFrame({
        "weekly_self_study_hours": [study_hours],
        "attendance_percentage": [attendance],
        "class_participation": [participation]
    })

    prediction = model.predict(input_df)[0]

    # Display Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="🎓 Predicted Score",
            value=f"{prediction:.2f}"
        )

    with col2:

        if prediction >= 90:
            performance = "🏆 Excellent"

        elif prediction >= 80:
            performance = "🥇 Very Good"

        elif prediction >= 70:
            performance = "👍 Good"

        elif prediction >= 60:
            performance = "🙂 Average"

        else:
            performance = "⚠ Needs Improvement"

        st.metric(
            label="Performance",
            value=performance
        )

    st.divider()

    st.subheader("📋 Input Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Weekly Self Study Hours",
            "Attendance Percentage",
            "Class Participation"
        ],
        "Value": [
            study_hours,
            attendance,
            participation
        ]
    })

    st.dataframe(summary, use_container_width=True)

    st.success("Prediction completed successfully! ✅")