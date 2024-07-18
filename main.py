import streamlit as st
import pandas as pd
import numpy as np
from data_processing import handle_missing_values, normalize_data, load_data
from visualization import plot_histogram, plot_boxplot, plot_2d_correlation, plot_3d_correlation, plot_correlation_heatmap
from clustering import perform_clustering, evaluate_clustering, plot_3d_clusters
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

st.set_page_config(page_title="Data Mining Project", page_icon=":bar_chart:", layout="wide")

st.sidebar.title(":male-technologist: Credits :writing_hand:")

st.sidebar.image("efrei_logo.png", use_column_width=True)

st.sidebar.markdown("""
    <style>
    .sidebar-content {
        font-family: 'Arial', sans-serif;
        color: #FFFFFF;
        padding: 10px;
    }
    .sidebar-content h3 {
        font-size: 20px;
        margin-bottom: 10px;
    }
    .sidebar-content a {
        color: #1E90FF;
        text-decoration: none;
    }
    .sidebar-content a:hover {
        text-decoration: underline;
    }
    .sidebar-content .class-info {
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    <div class="sidebar-content">
        <h3>Name: Jefferson LIN</h3>
        <p>Linkedin: <a href="https://www.linkedin.com/in/jefferson-lin-b718711b7/">Jefferson LIN</a></p>
        <p>Github: <a href="https://github.com/skill4chn">Jefferson LIN</a></p>
        <h3>Name: Nathanaël RAKOTO</h3>
        <p>Linkedin: <a href="https://www.linkedin.com/in/nathanael-rakoto">Nathanaël RAKOTO</a></p>
        <p>Github: <a href="https://github.com/Clutchboyyyy">Nathanaël RAKOTO</a></p>
        <p class="class-info">Class: BIA2</p>
    </div>
""", unsafe_allow_html=True)

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

st.markdown('<h1 class="title">Data Mining Project</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">By Issam Falih</p>', unsafe_allow_html=True)

st.markdown('<h2 class="header">Part I: Initial Data Exploration</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    data, numeric_cols, non_numeric_cols = load_data(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h2 class="subheader">Preview of the data</h2>', unsafe_allow_html=True)
        st.markdown('<h4 class="subheader">5 first rows</h4>', unsafe_allow_html=True)
        st.write(data.head())
        st.markdown('<h4 class="subheader">5 last rows</h4>', unsafe_allow_html=True)
        st.write(data.tail())
    
    with col2:
        st.markdown('<h2 class="subheader">Summary of the data</h2>', unsafe_allow_html=True)
        st.write(data.describe(include='all'))
    
    st.markdown('<h3 class="subheader">Dataset Information</h3>', unsafe_allow_html=True)
    st.write(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")
    st.write("Number of missing values per column:")
    st.write(data.isnull().sum())

    st.markdown('<h2 class="header">Part II: Data Pre-processing and Cleaning</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subheader">Managing Missing Values</h3>', unsafe_allow_html=True)
    missing_values_strategy = st.selectbox("Select a strategy for handling missing values:", 
                                           ("Delete rows", "Delete columns", "Impute with mean", "Impute with median", "Impute with mode", "KNN Imputation"))
    data_cleaned = handle_missing_values(data, numeric_cols, non_numeric_cols, missing_values_strategy)
    st.write(f"Dataset after handling missing values using '{missing_values_strategy}':")
    st.write(data_cleaned.head())

    st.markdown('<h3 class="subheader">Data Normalization</h3>', unsafe_allow_html=True)
    normalization_strategy = st.selectbox("Select a normalization method:", 
                                          ("None", "Min-Max Normalization", "Z-score Standardization"))
    data_cleaned = normalize_data(data_cleaned, numeric_cols, normalization_strategy)
    st.write(f"Dataset after applying '{normalization_strategy}':")
    st.write(data_cleaned.head())

    st.markdown('<h2 class="header">Part III: Visualization of the cleaned data</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3 class="subheader">Histograms</h3>', unsafe_allow_html=True)
        if len(numeric_cols) > 0:
            selected_column_hist = st.selectbox("Select a column for histogram visualization", numeric_cols)
            if selected_column_hist:
                plot_histogram(data_cleaned, selected_column_hist)

    with col2:
        st.markdown('<h3 class="subheader">Box Plots</h3>', unsafe_allow_html=True)
        if len(numeric_cols) > 0:
            selected_column_box = st.selectbox("Select a column for box plot visualization", numeric_cols, key='box_plot')
            if selected_column_box:
                plot_boxplot(data_cleaned, selected_column_box)
    st.markdown('<h3 class="subheader">Correlation Heatmap for All Columns</h3>', unsafe_allow_html=True)
    plot_correlation_heatmap(data_cleaned)

    st.markdown('<h3 class="subheader">2D Correlation Visualization</h3>', unsafe_allow_html=True)
    if len(numeric_cols) > 1:
        x_col_2d = st.selectbox("Select the X-axis column for the 2D plot", numeric_cols, key='2d_x')
        y_col_2d = st.selectbox("Select the Y-axis column for the 2D plot", numeric_cols, key='2d_y')
        plot_2d_correlation(data_cleaned, x_col_2d, y_col_2d)
            
    st.markdown('<h2 class="header">Part IV: Clustering or Prediction</h2>', unsafe_allow_html=True)
    task = st.selectbox("Choose a task", ("Clustering", "Prediction"))

    if task == "Clustering":
        st.markdown('<h3 class="subheader">Clustering</h3>', unsafe_allow_html=True)
        clustering_algorithm = st.selectbox("Select a clustering algorithm", ("KMeans", "DBSCAN"))
        model = perform_clustering(data_cleaned, numeric_cols, clustering_algorithm)

        st.markdown('<h2 class="header">Part V: Learning Evaluation</h2>', unsafe_allow_html=True)
        evaluate_clustering(data_cleaned, clustering_algorithm, model)
        if len(numeric_cols) >= 3:
            x_col_3d = st.selectbox("Select the X-axis column for the 3D plot", numeric_cols, key='3d_x')
            y_col_3d = st.selectbox("Select the Y-axis column for the 3D plot", numeric_cols, key='3d_y')
            z_col_3d = st.selectbox("Select the Z-axis column for the 3D plot", numeric_cols, key='3d_z')
            plot_3d_correlation(data_cleaned, x_col_3d, y_col_3d, z_col_3d)

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
            sns.scatterplot(x=y_test, y=predictions, ax=ax, alpha=0.3)
            slope, intercept, r_value, p_value, std_err = linregress(y_test, predictions)
            x_vals = np.array(ax.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, color='red')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.legend()
            st.pyplot(fig)
        
        elif prediction_algorithm == "Logistic Regression":
            target_cols = [col for col in data_cleaned.columns if set(data_cleaned[col].dropna().unique()) <= {0, 1}]
            if len(target_cols) == 0:
                st.error("No appropriate target column found for the selected algorithm.")
            else:
                target_column = st.selectbox("Select the target column", target_cols)
            
            X = data_cleaned.drop(columns=[target_column])
            y = data_cleaned[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions_proba = model.predict_proba(X_test)[:, 1]
            
            st.write("Logistic Regression Results")
            st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")

            fpr, tpr, thresholds = roc_curve(y_test, predictions_proba)
            roc_auc = roc_auc_score(y_test, predictions_proba)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)


st.markdown('<div class="footer">© 2024 Data Mining Issam Falih. All rights reserved.</div>', unsafe_allow_html=True)