import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN

st.title("Data Mining Project")
st.write("## Issam Falih")

st.header("Part I: Initial Data Exploration")
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    
    if file_type == 'csv':
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Preview of the first and last lines of the data:")
    st.write(data.head(), data.tail())
    
    st.write("A Summary of the data")
    st.write(data.describe(include='all'))
    st.write(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")
    st.write("Number of missing values per column:")
    st.write(data.isnull().sum())

