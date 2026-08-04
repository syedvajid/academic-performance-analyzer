import streamlit as st

st.set_page_config(
    page_title="Academic Performance Analyzer",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Academic Performance Analyzer")

st.markdown(
"""
### Welcome 👋

This application predicts a student's academic performance using Machine Learning.

---

## 🚀 Features

- 📈 Student Score Prediction
- 📊 Model Comparison
- 📉 Feature Importance
- 📂 Dataset Explorer
- 🤖 AI Recommendations

---

## 🧠 Machine Learning Algorithms

- Linear Regression
- Decision Tree
- Random Forest

---

## 📊 Dataset

| Attribute | Value |
|------------|-------|
| Records | 1,000,000 |
| Features | 3 |
| Target | Total Score |

---

### 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib

---
"""
)

st.success(
    "👈 Use the sidebar to navigate through the application."
)