import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Title of the app
st.title('SARIMA Sales Forecasting Tool')

# File upload section
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

# If a file is uploaded, process the data
if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file, sheet_name="data")
    
    # Show the first few rows to the user
    st.write(df.head())
    
    # Extract parameters from params tab
    params_df = pd.read_excel(uploaded_file, sheet_name="params")
    order_str = params_df.loc[params_df['parameter'] == 'order', 'value'].values[0]
    seasonal_order_str = params_df.loc[params_df['parameter'] == 'seasonal_order', 'value'].values[0]
    
    # Convert string to tuple
    order = tuple(map(int, order_str.strip('()').split(',')))
    seasonal_order = tuple(map(int, seasonal_order_str.strip('()').split(',')))
    forecast_weeks = int(params_df.loc[params_df['parameter'] == 'forecast_weeks', 'value'].values[0])
    
    # Preprocess the data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Group the data by week and size, aggregating sales
    df_weekly = df.groupby([pd.Grouper(freq='W'), 'Size']).agg({'Sales': 'sum'}).reset_index()
    
    # Forecasting for each size
    for size in df_weekly['Size'].unique():
        st.subheader(f'Forecast for {size}')
        
        # Filter data for the selected size
        size_data = df_weekly[df_weekly['Size'] == size]
        
        # Fit the SARIMA model
        model = SARIMAX(size_data['Sales'], order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)
        
        # Generate forecast
        forecast = results.get_forecast(steps=forecast_weeks)
        forecast_index = pd.date_range(start=size_data['Date'].max(), periods=forecast_weeks + 1, freq='W')[1:]
        forecast_values = forecast.predicted_mean
        
        # Plot the forecast
        plt.figure(figsize=(10, 6))
        plt.plot(size_data['Date'], size_data['Sales'], label='Observed', color='blue')
        plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
        plt.fill_between(forecast_index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color='pink', alpha=0.3)
        plt.title(f'Sales Forecast for {size}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        st.pyplot(plt)

        # Display forecast table
        forecast_table = pd.DataFrame({
            'Week': forecast_index,
            'Forecasted Sales': forecast_values
        })
        st.write(forecast_table) 
