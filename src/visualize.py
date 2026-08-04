"""
Visualization Module
"""

import matplotlib.pyplot as plt


def visualize(df):

    plt.figure(figsize=(8,5))

    plt.scatter(
        df["weekly_self_study_hours"],
        df["total_score"],
        alpha=0.3
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Total Score")
    plt.title("Study Hours vs Total Score")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))

    plt.hist(df["total_score"], bins=30)

    plt.title("Distribution of Scores")

    plt.xlabel("Score")

    plt.ylabel("Students")

    plt.grid(True)

    plt.show()