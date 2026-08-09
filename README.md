# 🎓 Academic Performance Analyzer

An end-to-end Machine Learning application for analyzing student academic data and predicting academic performance using **Python, Scikit-learn, and Streamlit**.

The project demonstrates a complete machine learning workflow, from data preprocessing and exploratory analysis to model training, evaluation, and deployment through an interactive web application.
---

## 🌐 Live Demo

🚀 **Try the deployed application:**  
[Launch Academic Performance Analyzer](https://academic-performance-analyzer-dseci2jzslks7nysrkrhc3.streamlit.app/)

The application is deployed using **Streamlit Community Cloud**.

---


## 📌 Project Overview

Academic performance can be influenced by multiple factors such as study habits, attendance, and classroom participation.

This project uses Machine Learning to analyze these factors and predict a student's academic score.

The application provides both individual and batch predictions through an interactive Streamlit interface.

---

## ✨ Features

- 🧹 Data preprocessing
- 🔍 Exploratory Data Analysis (EDA)
- 🛠️ Feature engineering
- ✂️ Train/Test data splitting
- 📈 Linear Regression model
- 🌳 Decision Tree Regression model
- ⚙️ Decision Tree hyperparameter tuning
- 📊 Model evaluation
- 🎯 Individual student score prediction
- 📂 Batch prediction using CSV files
- 📋 Dataset explorer
- 📉 Actual vs Predicted visualization
- 📉 Residual analysis
- 🖥️ Interactive multipage Streamlit application

---

## 🧠 Machine Learning Models

### Linear Regression

Linear Regression is used as the primary baseline regression model for predicting student academic scores.

### Decision Tree Regressor

A Decision Tree Regressor was also trained and evaluated to investigate whether a non-linear model could improve prediction performance.

The Decision Tree was regularized using parameters such as:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

This helps reduce overfitting and improve generalization.

---

## 📊 Model Evaluation

The models are evaluated using standard regression metrics:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

The Linear Regression model achieved approximately:

| Metric | Score |
|---|---:|
| R² Score | 0.6600 |
| MAE | 7.1613 |
| MSE | 80.9352 |
| RMSE | 8.9964 |

These results are calculated using a held-out test dataset rather than the training data.

---

## 🎯 Prediction Features

The prediction system currently uses:

- 📚 Weekly Self Study Hours
- 📅 Attendance Percentage
- 🙋 Class Participation

The trained model uses these features to estimate the student's total academic score.

---

## 🖥️ Streamlit Application

The project includes an interactive multipage Streamlit application.

### Pages

- 🏠 **Home**  
  Overview of the project and technology stack.

- 🎯 **Prediction**  
  Enter student information and generate an individual score prediction.

- 📊 **Model Analysis**  
  View regression metrics, Actual vs Predicted plots, residual analysis, and model information.

- 📂 **Dataset Explorer**  
  Explore the dataset, columns, shape, and descriptive statistics.

- 📦 **Batch Prediction**  
  Upload a CSV file and generate predictions for multiple students.

- ℹ️ **About**  
  Information about the project and its objectives.

---

## 🏗️ Project Structure

```text
academic-performance-analyzer/
│
├── app.py
│
├── pages/
│   ├── Prediction.py
│   ├── Model_Analysis.py
│   ├── Dataset_Explorer.py
│   ├── 5_Batch_Prediction.py
│   └── About.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── train_decision_tree.py
│   ├── evaluate_decision_tree.py
│   ├── predict.py
│   ├── visualize.py
│   └── main.py
│
├── models/
│   ├── student_model.pkl
│   └── decision_tree_model.pkl
│
├── data/
│
├── notebooks/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Joblib

### Visualization

- Matplotlib

### Web Application

- Streamlit

### Version Control

- Git
- GitHub

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd academic-performance-analyzer
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

Then open the local Streamlit address displayed in the terminal.

---

## 📂 Dataset

The dataset contains student academic information used for training and evaluating the regression models.

Large dataset files are excluded from this repository using `.gitignore` to keep the repository lightweight.

The ML workflow separates the dataset into:

- **80% Training Data**
- **20% Testing Data**

using a fixed random state for reproducibility.

---

## 🔄 Machine Learning Workflow

```text
Student Dataset
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Saved Model
      ↓
Streamlit Application
      ↓
Student Score Prediction
```

---
## 🚀 Future Improvements

Possible future extensions include:

- Random Forest Regression
- Gradient Boosting / XGBoost
- Automated hyperparameter optimization
- Additional academic and behavioral features
- Improved model interpretability

---

## 📌 Project Status

✅ **Completed and Deployed**

🌐 Live application available through Streamlit Community Cloud.

The current version includes the complete ML pipeline, trained regression models, model evaluation, individual prediction, batch prediction, and an interactive Streamlit dashboard.

---

## 👨‍💻 Author

**Syed Vajid**

Engineering Student | Machine Learning & AI Enthusiast

---

## 📜 License

This project is licensed under the **MIT License**
