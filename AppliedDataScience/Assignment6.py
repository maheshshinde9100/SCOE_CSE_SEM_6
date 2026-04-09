# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 2. Load Dataset
file_path = "Yamana_Gold_Inc._AUY.csv.xlsx"  # update path if needed
df = pd.read_excel(file_path)

# 3. Preprocess Data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select Close Price
data = df['Close'].dropna()


# 4. Plot Original Data
plt.figure()
plt.plot(data)
plt.title("Stock Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()




# 5. Build ARIMA Model
# (p=1, d=1, q=1)
model = ARIMA(data, order=(1,1,1))
model_fit = model.fit()

# 6. Print Model Summary
print(model_fit.summary())

# 7. Forecast Future Values
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

print("\nForecasted Values:")
print(forecast)

# 8. Plot Forecast
plt.figure()
plt.plot(data, label="Original Data")
plt.plot(range(len(data), len(data)+forecast_steps), forecast, label="Forecast")
plt.legend()
plt.title("ARIMA Forecast")
plt.show()
