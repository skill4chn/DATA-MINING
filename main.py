from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
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

st.markdown("""
    <style>
    .title {
        color: white;
        font-size: 36px;
        font-weight: bold;
    }
    .header {
        color: white;
        font-size: 28px;
    }
    .subheader {
        color: white;
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

st.markdown('<h2 class="header">Part I: Initial Data Exploration</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    separator_option = st.selectbox("Select the separator used in your file", (",", ";", "\t", " "))
    
    if file_type == 'csv':
        data = pd.read_csv(uploaded_file, sep=separator_option)
    else:
        data = pd.read_excel(uploaded_file)

    for col in data.columns:
        try:
            data[col] = data[col].str.replace(',', '.').astype(float)
        except:
            pass

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h2 class="subheader">Preview of the data</h2>', unsafe_allow_html=True)
        st.markdown('<h4 class="subheader">10 first rows</h4>', unsafe_allow_html=True)
        st.write(data.head())
        st.markdown('<h4 class="subheader">10 last rows</h4>', unsafe_allow_html=True)
        st.write(data.tail())

    with col2:
        st.markdown('<h2 class="subheader">Summary of the data</h2>', unsafe_allow_html=True)
        st.write(data.describe(include='all'))

    st.markdown('<h3 class="subheader">Dataset Information</h3>', unsafe_allow_html=True)
    st.write(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")
    st.write("Number of missing values per column:")
    st.write(data.isnull().sum())

    # Part II: Data Pre-processing and Cleaning
    st.markdown('<h2 class="header">Part II: Data Pre-processing and Cleaning</h2>', unsafe_allow_html=True)
    
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subheader">Histograms</h3>', unsafe_allow_html=True)
        if len(numeric_cols) > 0:
            selected_column_hist = st.selectbox("Select a column for histogram visualization", numeric_cols)
        
        if selected_column_hist:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(data_cleaned[selected_column_hist], kde=True, ax=ax)
            st.pyplot(fig)

    with col2:
        st.markdown('<h3 class="subheader">Box Plots</h3>', unsafe_allow_html=True)
        if len(numeric_cols) > 0:
            selected_column_box = st.selectbox("Select a column for box plot visualization", numeric_cols, key='box_plot')

        if selected_column_box:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=data_cleaned[selected_column_box], ax=ax)
            st.pyplot(fig)
            
            
    # Part IV: Clustering or Prediction
    st.markdown('<h2 class="header">Part IV: Clustering or Prediction</h2>', unsafe_allow_html=True)
    task = st.selectbox("Choose a task", ("Clustering", "Prediction"))

    if task == "Clustering":
        st.markdown('<h3 class="subheader">Clustering</h3>', unsafe_allow_html=True)
        clustering_algorithm = st.selectbox("Select a clustering algorithm", ("KMeans", "DBSCAN"))
        
        if clustering_algorithm == "KMeans":
            n_clusters = st.number_input("Number of clusters (K)", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(data_cleaned[numeric_cols])
            data_cleaned['Cluster'] = labels
            st.write(f"KMeans Clustering with {n_clusters} clusters")
            st.write(data_cleaned.head())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=data_cleaned[numeric_cols[0]], y=data_cleaned[numeric_cols[1]], hue='Cluster', data=data_cleaned, palette='viridis', ax=ax)
            st.pyplot(fig)
        
        elif clustering_algorithm == "DBSCAN":
            eps = st.number_input("Epsilon (eps)", min_value=0.1, max_value=10.0, value=0.5)
            min_samples = st.number_input("Minimum samples", min_value=1, max_value=10, value=5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data_cleaned[numeric_cols])
            data_cleaned['Cluster'] = labels
            st.write(f"DBSCAN Clustering with eps={eps} and min_samples={min_samples}")
            st.write(data_cleaned.head())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=data_cleaned[numeric_cols[0]], y=data_cleaned[numeric_cols[1]], hue='Cluster', data=data_cleaned, palette='viridis', ax=ax)
            st.pyplot(fig)

    elif task == "Prediction":
        st.markdown('<h3 class="subheader">Prediction</h3>', unsafe_allow_html=True)
        prediction_algorithm = st.selectbox("Select a prediction algorithm", ("Linear Regression", "Logistic Regression"))
        
        target_column = st.selectbox("Select the target column", data_cleaned.columns)
        
        X = data_cleaned.drop(columns=[target_column])
        y = data_cleaned[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if prediction_algorithm == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            st.write("Linear Regression Results")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
            st.write(f"R^2 Score: {model.score(X_test, y_test)}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=predictions, ax=ax)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            st.pyplot(fig)
        
        elif prediction_algorithm == "Logistic Regression":
            model = LogisticRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            st.write("Logistic Regression Results")
            st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pd.crosstab(y_test, predictions), annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            st.pyplot(fig)

            
st.markdown('<div class="footer">© 2024 Issam Falih. All rights reserved.</div>', unsafe_allow_html=True)