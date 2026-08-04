"""
Data Preprocessing Module
"""

import pandas as pd


def load_data():
    df = pd.read_csv("data/student_performance.csv")

    print("\nDataset Loaded Successfully!")
    print(df.head())

    return df