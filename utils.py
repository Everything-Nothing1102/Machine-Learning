import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def load_csv_data(filepath):
    """Load CSV data from the given filepath."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None

def normalize_column(df, column):
    """Normalize a single column in a DataFrame (z-score normalization)."""
    if column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

def plot_distribution(df, column):
    """Plot a histogram with seaborn for the given column."""
    if column in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

def display_summary(df):
    """Display basic summary statistics."""
    st.write("Data Preview:")
    st.dataframe(df.head())

    st.write("Summary Statistics:")
    st.dataframe(df.describe())