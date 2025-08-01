import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # Suppress warnings for simplicity

# Sample Time Series Data Generation
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = np.cumsum(np.random.randn(100)) + np.sin(np.linspace(0, 20, 100)) * 5 # Trend + seasonality-like
ts = pd.Series(data, index=dates)

# Split data (train on 80%, test on 20%)
train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:]

# Fit ARIMA model (p, d, q) - example values, typically determined by ACF/PACF plots
# For proper ARIMA, you'd analyze ACF/PACF plots to determine p, d, q
# This is a simplified example.
order = (5, 1, 0) # p=5 (AR order), d=1 (differencing order), q=0 (MA order)
model = ARIMA(train, order=order)
model_fit = model.fit()

print("\nARIMA Model Summary:")
print(model_fit.summary())

# Make predictions
forecast_steps = len(test)
forecast = model_fit.forecast(steps=forecast_steps)

# Evaluate
print(f"\nMean Squared Error (ARIMA): {mean_squared_error(test, forecast):.2f}")

# Plotting (optional)
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Actual Test')
plt.plot(test.index, forecast, label='Forecast')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()
