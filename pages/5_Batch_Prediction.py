import streamlit as st
import pandas as pd
import joblib

st.title("📂 Batch Prediction")

st.write(
    "Upload a CSV file to predict scores for multiple students."
)

model = joblib.load("models/student_model.pkl")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")

    st.dataframe(df.head())

    required_columns = [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation"
    ]

    if all(col in df.columns for col in required_columns):

        predictions = model.predict(df[required_columns])

        df["Predicted Score"] = predictions

        st.subheader("Predictions")

        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

    else:

        st.error(
            "Uploaded CSV does not contain the required columns."
        )