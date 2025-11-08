import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px

# Page config
st.set_page_config(page_title="Time Series Analyzer", layout="wide")

# Title
st.title("Time Series Analyzer")

# Sidebar controls
st.sidebar.header("Analysis Parameters")
forecast_days = st.sidebar.slider("Forecast Days", 7, 365, 30)

# File upload
uploaded_file = st.file_uploader("Upload your time series data (CSV)", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Data preparation
    st.subheader("Data Preparation")
    date_column = st.selectbox("Select date column", data.columns)
    value_column = st.selectbox("Select value column", data.columns)
    
    if st.button("Analyze"):
        # Prepare data for Prophet
        df = data[[date_column, value_column]].copy()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Create and train model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Plot results
        st.subheader("Forecast Results")
        fig1 = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'])
        fig1.add_scatter(x=df['ds'], y=df['y'], name='Actual')
        st.plotly_chart(fig1)
        
        # Components
        st.subheader("Time Series Components")
        components = model.plot_components(forecast)
        st.pyplot(components)
        
        # Download forecast
        st.download_button(
            label="Download Forecast Data",
            data=forecast.to_csv(index=False),
            file_name="forecast_results.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file with your time series data")