import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# Page configuration
st.set_page_config(page_title="Data Mining Project", page_icon=":bar_chart:", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
    }
    .header {
        color: #2E7D32;
        font-size: 28px;
    }
    .subheader {
        color: #388E3C;
        font-size: 24px;
    }
    .description {
        color: #757575;
    }
    .footer {
        color: #BDBDBD;
        font-size: 14px;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="title">Data Mining Project</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">By Issam Falih</p>', unsafe_allow_html=True)

# Header for data exploration section
st.markdown('<h2 class="header">Part I: Initial Data Exploration</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]

    # Header and separator options
    header_option = st.selectbox("Does your file have a header?", ("Yes", "No"))
    separator_option = st.selectbox("Select the separator used in your file", (",", ";", "\t", " "))

    header = 0 if header_option == "Yes" else None

    if file_type == 'csv':
        data = pd.read_csv(uploaded_file, header=header, sep=separator_option)
    else:
        data = pd.read_excel(uploaded_file, header=header)

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.replace(',', '.').astype(float)
        
    # Using columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subheader">Preview of the data</h3>', unsafe_allow_html=True)
        st.write(data.head())
        st.write(data.tail())

    with col2:
        st.markdown('<h3 class="subheader">Summary of the data</h3>', unsafe_allow_html=True)
        st.write(data.describe(include='all'))

    st.markdown('<h3 class="subheader">Dataset Information</h3>', unsafe_allow_html=True)
    st.write(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")
    st.write("Number of missing values per column:")
    st.write(data.isnull().sum())

    # Part II: Data Pre-processing and Cleaning
    st.markdown('<h2 class="header">Part II: Data Pre-processing and Cleaning</h2>', unsafe_allow_html=True)
    
    # Managing missing values
    st.markdown('<h3 class="subheader">Managing Missing Values</h3>', unsafe_allow_html=True)
    missing_values_strategy = st.selectbox("Select a strategy for handling missing values:", 
                                           ("Delete rows", "Delete columns", "Impute with mean", "Impute with median", "Impute with mode", "KNN Imputation"))

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = data.select_dtypes(exclude=['int64', 'float64']).columns

    if missing_values_strategy == "Delete rows":
        data_cleaned = data.dropna()
    elif missing_values_strategy == "Delete columns":
        data_cleaned = data.dropna(axis=1)
    elif missing_values_strategy == "Impute with mean":
        imputer = SimpleImputer(strategy='mean')
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        data_cleaned = data
    elif missing_values_strategy == "Impute with median":
        imputer = SimpleImputer(strategy='median')
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        data_cleaned = data
    elif missing_values_strategy == "Impute with mode":
        imputer_num = SimpleImputer(strategy='most_frequent')
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])
        data[non_numeric_cols] = imputer_cat.fit_transform(data[non_numeric_cols])
        data_cleaned = data
    elif missing_values_strategy == "KNN Imputation":
        imputer = KNNImputer(n_neighbors=5)
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        data_cleaned = data

    st.write(f"Dataset after handling missing values using '{missing_values_strategy}':")
    st.write(data_cleaned.head())

    # Data normalization
    st.markdown('<h3 class="subheader">Data Normalization</h3>', unsafe_allow_html=True)
    normalization_strategy = st.selectbox("Select a normalization method:", 
                                          ("None", "Min-Max Normalization", "Z-score Standardization"))

    if normalization_strategy == "Min-Max Normalization":
        scaler = MinMaxScaler()
        data_cleaned[numeric_cols] = scaler.fit_transform(data_cleaned[numeric_cols])
    elif normalization_strategy == "Z-score Standardization":
        scaler = StandardScaler()
        data_cleaned[numeric_cols] = scaler.fit_transform(data_cleaned[numeric_cols])

    st.write(f"Dataset after applying '{normalization_strategy}':")
    st.write(data_cleaned.head())

    # Part III: Visualization of the cleaned data
    st.markdown('<h2 class="header">Part III: Visualization of the cleaned data</h2>', unsafe_allow_html=True)

    # Histogram visualization
    st.markdown('<h3 class="subheader">Histograms</h3>', unsafe_allow_html=True)
    if len(numeric_cols) > 0:
        selected_column_hist = st.selectbox("Select a column for histogram visualization", numeric_cols)
        
        if selected_column_hist:
            fig, ax = plt.subplots()
            sns.histplot(data_cleaned[selected_column_hist].astype(float), kde=True, ax=ax)
            st.pyplot(fig)

    # Box plot visualization
    st.markdown('<h3 class="subheader">Box Plots</h3>', unsafe_allow_html=True)
    if len(numeric_cols) > 0:
        selected_column_box = st.selectbox("Select a column for box plot visualization", numeric_cols, key='box_plot')

        if selected_column_box:
            fig, ax = plt.subplots()
            sns.boxplot(x=data_cleaned[selected_column_box].astype(float), ax=ax)
            st.pyplot(fig)

    # Bar plot visualization for non-numeric columns
    st.markdown('<h3 class="subheader">Bar Plots for Categorical Data</h3>', unsafe_allow_html=True)
    if len(non_numeric_cols) > 0:
        selected_column_bar = st.selectbox("Select a column for bar plot visualization", non_numeric_cols)

        if selected_column_bar:
            fig, ax = plt.subplots()
            sns.countplot(y=selected_column_bar, data=data_cleaned, ax=ax)
            st.pyplot(fig)