import streamlit as st
import joblib
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Prediction",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/student_model_v2.pkl"

FEATURES = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🎯 Student Performance Prediction")

st.write(
    "Enter a student's academic information below to estimate "
    "their expected total score using the trained Linear Regression model."
)

st.divider()


# ============================================================
# STUDENT INPUT
# ============================================================

st.subheader("📚 Student Information")

col1, col2, col3 = st.columns(3)


with col1:

    study_hours = st.number_input(
        "📖 Weekly Self-Study Hours",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=0.5
    )


with col2:

    attendance = st.number_input(
        "📅 Attendance Percentage",
        min_value=50.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )


with col3:

    participation = st.number_input(
        "🙋 Class Participation",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🚀 Predict Student Score",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "weekly_self_study_hours": [
            study_hours
        ],

        "attendance_percentage": [
            attendance
        ],

        "class_participation": [
            participation
        ]

    })


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # Keep score within valid range
    prediction = max(
        0,
        min(100, prediction)
    )


    # --------------------------------------------------------
    # PERFORMANCE CLASSIFICATION
    # --------------------------------------------------------

    if prediction >= 90:

        performance = "🏆 Excellent"

        message = (
            "Outstanding predicted performance. "
            "The student is expected to perform exceptionally well."
        )

    elif prediction >= 75:

        performance = "🥇 Very Good"

        message = (
            "The student is expected to achieve a strong "
            "academic performance."
        )

    elif prediction >= 60:

        performance = "🥈 Good"

        message = (
            "The student is expected to achieve a satisfactory "
            "academic performance."
        )

    elif prediction >= 50:

        performance = "🥉 Average"

        message = (
            "The student may benefit from additional preparation "
            "and consistent academic engagement."
        )

    else:

        performance = "⚠️ Needs Improvement"

        message = (
            "The prediction suggests that the student should "
            "focus on improving study habits and academic engagement."
        )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("🎯 Prediction Result")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "🎓 Predicted Total Score",
            f"{prediction:.2f} / 100"
        )


    with col2:

        st.metric(
            "📊 Performance Level",
            performance
        )


    st.success(
        f"🎯 Predicted Score: **{prediction:.2f} / 100**"
    )


    st.info(message)


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📋 Student Input Summary")

    summary = pd.DataFrame({

        "Feature": [

            "Weekly Self-Study Hours",

            "Attendance Percentage",

            "Class Participation"

        ],

        "Value": [

            f"{study_hours:.1f} hours",

            f"{attendance:.1f}%",

            f"{participation:.1f} / 10"

        ]

    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    st.subheader("🧠 Model Information")

    st.write(
        "This prediction is generated using the "
        "**Linear Regression** model selected during "
        "the Model Analysis stage."
    )


    model_info = pd.DataFrame({

        "Metric": [

            "Model",

            "R² Score",

            "MAE",

            "RMSE"

        ],

        "Value": [

            "Linear Regression",

            "0.8289",

            "2.4895",

            "3.1146"

        ]

    })


    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Academic Performance Analyzer • "
    "Machine Learning Prediction System"
)