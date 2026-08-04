"""
Academic Performance Analyzer

Main Entry Point
"""

from data_preprocessing import load_data
from eda import perform_eda
from visualize import visualize
from train_model import train_model
from evaluate_model import evaluate_model
from predict import predict


def main():

    print("=" * 60)
    print("        Academic Performance Analyzer")
    print("=" * 60)

    # Step 1: Load Dataset
    df = load_data()

    # Display Dataset Information
    print("\nDataset Shape :", df.shape)
    print("\nColumns :")
    print(df.columns.tolist())

    # Step 2: Exploratory Data Analysis (EDA)
    perform_eda(df)

    # Step 3: Data Visualization
    visualize(df)

    # Step 4: Train and Compare Models
    model, X_test, y_test = train_model(df)

    # Step 5: Evaluate the Best Model
    evaluate_model(model, X_test, y_test)

    # Step 6: Make Predictions
    predict()


if __name__ == "__main__":
    main()
    