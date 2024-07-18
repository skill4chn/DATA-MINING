from sklearn.cluster import KMeans, DBSCAN
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def perform_clustering(data, numeric_cols, algorithm):
    model = None
    if algorithm == "KMeans":
        n_clusters = st.number_input("Number of clusters (K)", min_value=2, max_value=10, value=3)
        x_column = st.selectbox("Select the X-axis column for the plot", numeric_cols, key='kmeans_x')
        y_column = st.selectbox("Select the Y-axis column for the plot", numeric_cols, key='kmeans_y')
        
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(data[numeric_cols])
        data['Cluster'] = labels
        st.write(f"KMeans Clustering with {n_clusters} clusters")
        st.write(data.head())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data[x_column], y=data[y_column], hue='Cluster', data=data, palette='viridis', ax=ax)
        ax.set_xlabel(x_column, fontsize=12)  
        ax.set_ylabel(y_column, fontsize=12)  
        st.pyplot(fig)
    
    elif algorithm == "DBSCAN":
        eps = st.number_input("Epsilon (eps)", min_value=0.1, max_value=10.0, value=0.5)
        min_samples = st.number_input("Minimum samples", min_value=1, max_value=10, value=5)
        x_column = st.selectbox("Select the X-axis column for the plot", numeric_cols, key='dbscan_x')
        y_column = st.selectbox("Select the Y-axis column for the plot", numeric_cols, key='dbscan_y')
        z_column = st.selectbox("Select the Z-axis column for the plot", numeric_cols, key='dbscan_z')
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data[numeric_cols])
        data['Cluster'] = labels
        st.write(f"DBSCAN Clustering with eps={eps} and min_samples={min_samples}")
        st.write(data.head())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data[x_column], y=data[y_column], hue='Cluster', data=data, palette='viridis', ax=ax)
        ax.set_xlabel(x_column, fontsize=12) 
        ax.set_ylabel(y_column, fontsize=12) 
        st.pyplot(fig)
    return model

def evaluate_clustering(data, algorithm, model):
    if algorithm == "KMeans":
        st.markdown('<h3 class="subheader">Cluster Statistics for KMeans</h3>', unsafe_allow_html=True)
        cluster_centers = model.cluster_centers_
        st.write("Cluster Centers:")
        st.write(cluster_centers)
        
        cluster_counts = data['Cluster'].value_counts().sort_index()
        st.write("Number of data points in each cluster:")
        st.write(cluster_counts)
    
    elif algorithm == "DBSCAN":
        st.markdown('<h3 class="subheader">Cluster Statistics for DBSCAN</h3>', unsafe_allow_html=True)
        cluster_counts = data['Cluster'].value_counts().sort_index()
        st.write("Number of data points in each cluster:")
        st.write(cluster_counts)

def plot_3d_clusters(data, numeric_cols):
    st.markdown('<h3 class="subheader">3D Scatter Plot of Clusters</h3>', unsafe_allow_html=True)
    x_col_3d = st.selectbox("Select the X-axis column for the 3D plot", numeric_cols, key='3d_x')
    y_col_3d = st.selectbox("Select the Y-axis column for the 3D plot", numeric_cols, key='3d_y')
    z_col_3d = st.selectbox("Select the Z-axis column for the 3D plot", numeric_cols, key='3d_z')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[x_col_3d], data[y_col_3d], data[z_col_3d], c=data['Cluster'], cmap='viridis')
    ax.set_xlabel(x_col_3d)
    ax.set_ylabel(y_col_3d)
    ax.set_zlabel(z_col_3d)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
