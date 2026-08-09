"""
Prediction Module
"""

import joblib
import pandas as pd


def predict():

    # Load the trained model
    model = joblib.load("models/student_model.pkl")

    print("\n===== Student Performance Predictor =====")

    # Get user input
    study_hours = float(input("Weekly Self Study Hours: "))
    attendance = float(input("Attendance Percentage: "))
    participation = float(input("Class Participation (0-10): "))

    # Validate inputs
    if study_hours < 0:
        print("❌ Study hours cannot be negative.")
        return

    if not 0 <= attendance <= 100:
        print("❌ Attendance must be between 0 and 100.")
        return

    if not 0 <= participation <= 10:
        print("❌ Class participation must be between 0 and 10.")
        return

    # Create prediction DataFrame
    sample = pd.DataFrame(
        {
            "weekly_self_study_hours": [study_hours],
            "attendance_percentage": [attendance],
            "class_participation": [participation],
        }
    )

    # Display exactly what is being sent to the model
    print("\n===== Input Sent to Model =====")
    print(sample.to_string(index=False))

    # Make prediction
    prediction = model.predict(sample)[0]

    # Keep score within the valid total-score range
    prediction = max(0, min(100, prediction))

    print("\n🎯 Predicted Total Score:", round(prediction, 2))

    # Performance classification
    if prediction >= 75:
        performance = "Good"
    elif prediction >= 60:
        performance = "Average"
    else:
        performance = "Needs Improvement"

    print("📊 Performance:", performance)