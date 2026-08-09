"""
Synthetic Academic Performance Dataset Generator

Feature Contribution:
- Self-study: 30 points
- Attendance: 18 points
- Class participation: 12 points
- Baseline: 40 points

Total possible score = 100
"""

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

N_STUDENTS = 10_000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# Generate Student IDs
# ============================================================

student_id = np.arange(1, N_STUDENTS + 1)


# ============================================================
# Generate Features
# ============================================================

# ------------------------------------------------------------
# Weekly Self-Study Hours
# Range: 0-40
# ------------------------------------------------------------

study_hours = np.clip(
    rng.normal(
        loc=15,
        scale=7,
        size=N_STUDENTS
    ),
    0,
    40
)


# ------------------------------------------------------------
# Attendance Percentage
# Range: 50-100
# ------------------------------------------------------------

attendance = np.clip(
    rng.normal(
        loc=85,
        scale=12,
        size=N_STUDENTS
    ),
    0,
    100
)


# ------------------------------------------------------------
# Class Participation
# Range: 0-10
# ------------------------------------------------------------

participation = np.clip(
    rng.normal(
        loc=6,
        scale=2,
        size=N_STUDENTS
    ),
    0,
    10
)


# ============================================================
# Normalize Features
# ============================================================

# Study:
# 0 hours  -> 0
# 40 hours -> 1

study_norm = study_hours / 40


# Attendance:
# 50%  -> 0
# 100% -> 1

attendance_norm = (attendance - 50) / 50


# Participation:
# 0 -> 0
# 10 -> 1

participation_norm = participation / 10


# ============================================================
# Academic Score
# ============================================================

# Baseline = 40
#
# Study contribution        = 30 points
# Attendance contribution   = 18 points
# Participation contribution = 12 points
#
# Maximum:
#
# 40 + 30 + 18 + 12 = 100


base_score = 48

study_contribution = 30 * study_norm

attendance_contribution = 18 * attendance_norm

participation_contribution = 12 * participation_norm


total_score = (
    base_score
    + study_contribution
    + attendance_contribution
    + participation_contribution
)


# ============================================================
# Add Realistic Noise
# ============================================================

noise = rng.normal(
    loc=0,
    scale=3,
    size=N_STUDENTS
)

total_score = total_score + noise


# ============================================================
# Keep Scores Valid
# ============================================================

total_score = np.clip(
    total_score,
    0,
    100
)

total_score = np.round(
    total_score,
    1
)


# ============================================================
# Generate Grades
# ============================================================

def assign_grade(score):
    """
    Convert total score into a letter grade.
    """

    if score >= 90:
        return "A"

    elif score >= 80:
        return "B"

    elif score >= 70:
        return "C"

    elif score >= 60:
        return "D"

    else:
        return "F"


grades = np.array([
    assign_grade(score)
    for score in total_score
])


# ============================================================
# Create DataFrame
# ============================================================

df = pd.DataFrame({
    "student_id": student_id,

    "weekly_self_study_hours": np.round(
        study_hours,
        1
    ),

    "attendance_percentage": np.round(
        attendance,
        1
    ),

    "class_participation": np.round(
        participation,
        1
    ),

    "total_score": total_score,

    "grade": grades
})


# ============================================================
# Save Dataset
# ============================================================

output_path = "data/student_performance_v2.csv"

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# Dataset Summary
# ============================================================

print("=" * 65)
print("Synthetic Academic Performance Dataset Generated")
print("=" * 65)

print(f"\nRows generated : {len(df):,}")
print(f"Saved to       : {output_path}")


# ============================================================
# Statistical Summary
# ============================================================

print("\nDataset Summary:")

print(
    df[
        [
            "weekly_self_study_hours",
            "attendance_percentage",
            "class_participation",
            "total_score"
        ]
    ]
    .describe()
    .round(2)
)


# ============================================================
# Correlation Analysis
# ============================================================

print("\nCorrelation with total_score:")

correlations = (
    df[
        [
            "weekly_self_study_hours",
            "attendance_percentage",
            "class_participation",
            "total_score"
        ]
    ]
    .corr()["total_score"]
    .sort_values(ascending=False)
    .round(4)
)

print(correlations)


# ============================================================
# Grade Distribution
# ============================================================

print("\nGrade Distribution:")

grade_distribution = (
    df["grade"]
    .value_counts(normalize=True)
    .sort_index()
    .round(3)
)

print(grade_distribution)


# ============================================================
# First 10 Rows
# ============================================================

print("\nFirst 10 Rows:")

print(
    df.head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 65)
print("Dataset generation completed successfully! ✅")
print("=" * 65)