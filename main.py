# AI-Powered Stock Price Movement Prediction & Signal Generator

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Step 1: Fetch Historical Stock Data
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Step 2: Feature Engineering
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data['Close'])
data.dropna(inplace=True)

# Step 3: Labeling for Classification (Buy/Sell/Hold)
future_days = 1
data['Target'] = data['Close'].shift(-future_days) - data['Close']
data['Target'] = data['Target'].apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))
data.dropna(inplace=True)

# Step 4: Train-Test Split
features = ['MA_10', 'MA_50', 'RSI']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Train Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Time Series Forecasting with ARIMA
print("\nARIMA Forecast (Next 5 Days):")
model_arima = ARIMA(data['Close'], order=(5, 1, 0))
model_fit = model_arima.fit()
forecast_arima = model_fit.forecast(steps=5)
print(forecast_arima)

# Step 7: Forecasting with Prophet
prophet_data = data.reset_index()[['Date', 'Close']]
prophet_data.columns = ['ds', 'y']

model_prophet = Prophet()
model_prophet.fit(prophet_data)
future = model_prophet.make_future_dataframe(periods=7)
forecast_prophet = model_prophet.predict(future)

fig = model_prophet.plot(forecast_prophet)
plt.title('7-Day Forecast using Prophet')
plt.show()

# Step 8: Visualization of Buy/Sell Signals
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, data['MA_10'], label='MA 10')
plt.plot(data.index, data['MA_50'], label='MA 50')
plt.legend()
plt.title(f'{ticker} - Price & Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()
