"""
Exploratory Data Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df):

    print("\n========== DATASET INFORMATION ==========\n")

    print(df.info())

    print("\n========== STATISTICS ==========\n")

    print(df.describe())

    print("\n========== MISSING VALUES ==========\n")

    print(df.isnull().sum())

    # Correlation Heatmap
    plt.figure(figsize=(8,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="Blues"
    )

    plt.title("Correlation Heatmap")

    plt.show()