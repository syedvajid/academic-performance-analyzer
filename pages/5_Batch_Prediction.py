import streamlit as st
import joblib
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📦",
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

st.title("📦 Batch Student Prediction")

st.write(
    "Upload a CSV file containing multiple students and "
    "predict their expected academic scores using the "
    "Linear Regression model."
)

st.divider()


# ============================================================
# REQUIRED CSV FORMAT
# ============================================================

st.subheader("📋 Required CSV Format")

st.write(
    "Your CSV file must contain the following three columns:"
)

required_columns_df = pd.DataFrame({
    "Column": FEATURES,

    "Description": [
        "Weekly self-study hours",
        "Attendance percentage",
        "Class participation score"
    ]
})

st.dataframe(
    required_columns_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SAMPLE CSV
# ============================================================

st.subheader("🧪 Need a Sample CSV?")

sample_data = pd.DataFrame({

    "weekly_self_study_hours": [
        10,
        15,
        20,
        25,
        30
    ],

    "attendance_percentage": [
        70,
        78,
        85,
        90,
        95
    ],

    "class_participation": [
        4,
        5,
        7,
        8,
        9
    ]
})


sample_csv = sample_data.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Sample CSV",

    data=sample_csv,

    file_name="sample_students.csv",

    mime="text/csv",

    use_container_width=True
)


st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📤 Upload Student CSV")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"❌ Could not read the CSV file: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("👀 Dataset Preview")

    st.write(
        f"Found **{len(df):,} student records**."
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = [

        column

        for column in FEATURES

        if column not in df.columns

    ]


    if missing_columns:

        st.error(
            "❌ Missing required columns: "
            + ", ".join(missing_columns)
        )

        st.info(
            "Please make sure your CSV contains "
            "weekly_self_study_hours, "
            "attendance_percentage, and "
            "class_participation."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK NUMERIC VALUES
    # --------------------------------------------------------

    invalid_columns = []

    for column in FEATURES:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.isna().any():

            invalid_columns.append(
                column
            )


    if invalid_columns:

        st.error(
            "❌ The following columns contain "
            "missing or non-numeric values:"
        )

        for column in invalid_columns:

            st.write(
                f"- `{column}`"
            )

        st.stop()


    # --------------------------------------------------------
    # CONVERT FEATURES TO NUMERIC
    # --------------------------------------------------------

    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column]
        )


    # --------------------------------------------------------
    # RANGE WARNINGS
    # --------------------------------------------------------

    if (
        (df["weekly_self_study_hours"] < 0).any()
        or
        (df["weekly_self_study_hours"] > 40).any()
    ):

        st.warning(
            "⚠️ Some self-study hour values are "
            "outside the dataset's expected range of 0–40 hours."
        )


    if (
        (df["attendance_percentage"] < 50).any()
        or
        (df["attendance_percentage"] > 100).any()
    ):

        st.warning(
            "⚠️ Some attendance values are "
            "outside the dataset's expected range of 50–100%."
        )


    if (
        (df["class_participation"] < 0).any()
        or
        (df["class_participation"] > 10).any()
    ):

        st.warning(
            "⚠️ Some participation values are "
            "outside the dataset's expected range of 0–10."
        )


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🚀 Predict All Students",
        use_container_width=True
    )


    if predict_button:

        # ====================================================
        # PREPARE INPUT FEATURES
        # ====================================================

        X = df[FEATURES].copy()


        # ====================================================
        # GENERATE PREDICTIONS
        # ====================================================

        predictions = model.predict(
            X
        )


        # Keep predictions inside 0–100
        predictions = predictions.clip(
            0,
            100
        )


        # ====================================================
        # CREATE RESULTS
        # ====================================================

        results = df.copy()

        results["predicted_total_score"] = (
            predictions.round(2)
        )


        # ====================================================
        # PERFORMANCE CLASSIFICATION
        # ====================================================

        def classify_performance(score):

            if score >= 90:

                return "🏆 Excellent"

            elif score >= 75:

                return "🥇 Very Good"

            elif score >= 60:

                return "🥈 Good"

            elif score >= 50:

                return "🥉 Average"

            else:

                return "⚠️ Needs Improvement"


        results["performance_level"] = (

            results["predicted_total_score"]

            .apply(
                classify_performance
            )

        )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            f"✅ Successfully predicted "
            f"{len(results):,} students!"
        )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Prediction Results"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SUMMARY STATISTICS
        # ====================================================

        st.subheader(
            "📊 Batch Prediction Summary"
        )


        average_score = (
            results[
                "predicted_total_score"
            ].mean()
        )


        highest_score = (
            results[
                "predicted_total_score"
            ].max()
        )


        lowest_score = (
            results[
                "predicted_total_score"
            ].min()
        )


        student_count = len(
            results
        )


        # ====================================================
        # SUMMARY METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "📊 Average Score",
                f"{average_score:.2f}"
            )


        with col2:

            st.metric(
                "🏆 Highest Score",
                f"{highest_score:.2f}"
            )


        with col3:

            st.metric(
                "📉 Lowest Score",
                f"{lowest_score:.2f}"
            )


        with col4:

            st.metric(
                "👨‍🎓 Students",
                f"{student_count:,}"
            )


        # ====================================================
        # PERFORMANCE DISTRIBUTION
        # ====================================================

        st.divider()

        st.subheader(
            "📈 Performance Distribution"
        )


        performance_order = [

            "🏆 Excellent",

            "🥇 Very Good",

            "🥈 Good",

            "🥉 Average",

            "⚠️ Needs Improvement"

        ]


        distribution_counts = (

            results[
                "performance_level"
            ]

            .value_counts()

            .reindex(
                performance_order,
                fill_value=0
            )

        )


        distribution = pd.DataFrame({

            "Performance Level":
                performance_order,

            "Students":
                distribution_counts.values

        })


        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # PERFORMANCE CHART
        # ====================================================

        chart_data = (

            distribution

            .set_index(
                "Performance Level"
            )

        )


        st.bar_chart(
            chart_data,
            use_container_width=True
        )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.divider()

        st.subheader(
            "💾 Download Predictions"
        )


        output_csv = results.to_csv(
            index=False
        )


        st.download_button(

            label="⬇️ Download Prediction Results",

            data=output_csv,

            file_name="student_predictions.csv",

            mime="text/csv",

            use_container_width=True

        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.subheader(
            "🧠 Model Information"
        )


        st.write(
            "Batch predictions are generated using the "
            "**Linear Regression** model selected during "
            "model evaluation."
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
    "Batch Machine Learning Prediction System"
)