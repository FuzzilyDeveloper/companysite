import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page config
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Title
st.title("Data Analysis Dashboard")

# Sidebar
st.sidebar.header("Dashboard Controls")

# Sample data loading
@st.cache_data
def load_data():
    # Replace this with your actual data loading logic
    return pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'Sales': np.random.randn(100).cumsum() + 100,
        'Traffic': np.random.randint(100, 1000, 100)
    })

# Load data
data = load_data()

# Create visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Trend")
    fig1 = px.line(data, x='Date', y='Sales')
    st.plotly_chart(fig1)

with col2:
    st.subheader("Traffic Analysis")
    fig2 = px.bar(data, x='Date', y='Traffic')
    st.plotly_chart(fig2)

# Data table
st.subheader("Raw Data")
st.dataframe(data)