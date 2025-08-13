# AI-Powered-Stock-Price-Movement-Prediction-Signal-Generator

This repository implements an end-to-end machine learning pipeline for analyzing and predicting stock price movements, generating trading signals, and forecasting future prices using classical and deep learning time series models.


🚀 Features
📥 Fetches historical stock data using yfinance
⚙️ Performs technical feature engineering (Moving Averages, RSI)
🤖 Builds a Random Forest Classifier for Buy/Sell/Hold signal generation
⏳ Implements time series forecasting using:
ARIMA (classical statistical model)
Prophet (robust trend modeling by Meta)
📊 Visualizes price trends, indicators, and forecasts using matplotlib
🛠️ Tech Stack
Python 3.x
Libraries:
yfinance, pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels, prophet
📁 Project Structure
AI-Powered-Stock-Price-Movement-Prediction-Signal-Generator/ ├── main.py # Core script with modeling pipeline ├── requirements.txt # All dependencies └── README.md # Project documentation

🧠 Models Used
Model	Purpose
Random Forest	Classification (Buy/Sell/Hold)
ARIMA	Univariate time series forecasting
Prophet	Trend & seasonality-based forecasting
