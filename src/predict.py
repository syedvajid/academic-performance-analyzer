"""
Prediction Module
"""

import joblib
import pandas as pd


def predict():

    # Load the trained model
    model = joblib.load("models/student_model.pkl")

    print("\n===== Student Performance Predictor =====")

    study_hours = float(input("Weekly Self Study Hours: "))
    attendance = float(input("Attendance Percentage: "))
    participation = float(input("Class Participation (0-10): "))

    sample = pd.DataFrame(
        [[study_hours, attendance, participation]],
        columns=[
            "weekly_self_study_hours",
            "attendance_percentage",
            "class_participation",
        ],
    )

    prediction = model.predict(sample)

    print("\n🎯 Predicted Total Score:", round(prediction[0], 2))