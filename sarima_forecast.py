import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import openpyxl

# === Load Excel input ===
input_file = 'sarima_forecast_input.xlsx'
data_df = pd.read_excel(input_file, sheet_name='data')
params_df = pd.read_excel(input_file, sheet_name='params', index_col=0)

# === Clean and prepare data ===
data_df = data_df.dropna()
data_df.columns = ['date', 'sales']
data_df['date'] = pd.to_datetime(data_df['date'])
data_df.set_index('date', inplace=True)
data_df = data_df.asfreq('W')  # Weekly frequency

# === Get forecast length ===
forecast_steps = int(params_df.loc['Forecast Weeks', 'Value'])

# === SARIMA parameters (hardcoded per your model) ===
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)

# === Fit model ===
model = SARIMAX(data_df['sales'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# === Forecast ===
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data_df.index[-1] + timedelta(weeks=1), periods=forecast_steps, freq='W')
forecast_df = pd.DataFrame({'date': forecast_index, 'forecast': forecast.predicted_mean.values})

# === Save forecast to new sheet ===
with pd.ExcelWriter(input_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    forecast_df.to_excel(writer, sheet_name='forecast', index=False)

print(f"✅ Forecast saved to 'forecast' tab in {input_file}")
