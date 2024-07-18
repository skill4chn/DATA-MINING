import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_histogram(data, column):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(data[column], kde=True, ax=ax)
    st.pyplot(fig)

def plot_boxplot(data, column):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=data[column], ax=ax)
    st.pyplot(fig)
