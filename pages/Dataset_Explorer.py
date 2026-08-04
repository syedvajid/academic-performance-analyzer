import streamlit as st
import pandas as pd

st.title("📂 Dataset Explorer")

df = pd.read_csv("data/student_performance_sample.csv")

st.subheader("Dataset")

st.dataframe(df.head(100))

st.subheader("Dataset Shape")

st.write(df.shape)

st.subheader("Columns")

st.write(df.columns.tolist())

st.subheader("Statistics")

st.dataframe(df.describe())