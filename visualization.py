import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_histogram(data, column):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(data[column], kde=True, ax=ax)
    st.pyplot(fig)

def plot_boxplot(data, column):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=data[column], ax=ax)
    st.pyplot(fig)

def plot_2d_correlation(data, x_column, y_column):
    correlation = data[[x_column, y_column]].corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=data[x_column], y=data[y_column], ax=ax)
    ax.set_title(f'Correlation: {correlation:.2f}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)
    
    # Display correlation value
    st.markdown(f"### Correlation between {x_column} and {y_column}:")
    st.markdown(f"<h4 style='color: green;'>{correlation:.2f}</h4>", unsafe_allow_html=True)

def plot_3d_correlation(data, x_column, y_column, z_column):
    correlation_matrix = data[[x_column, y_column, z_column]].corr()
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[x_column], data[y_column], data[z_column], c=data['Cluster'], cmap='viridis')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_zlabel(z_column)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.title('3D Scatter Plot with Clusters')
    st.pyplot(fig)
    
def plot_correlation_heatmap(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)