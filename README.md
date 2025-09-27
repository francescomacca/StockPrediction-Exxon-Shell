This project focuses on analyzing and predicting the historical stock prices of ExxonMobil (XOM) and Royal Dutch Shell (SHELL) using different Machine Learning approaches.

📊 Project Overview

Data Source: Nasdaq.com historical stock data

Companies: ExxonMobil and Shell, two leading global energy companies strongly influenced by oil price fluctuations, geopolitical decisions (e.g., OPEC), and global economic cycles.

Goal: Predict closing prices using multiple ML techniques and compare their performance.

🔍 Exploratory Data Analysis

Strong positive correlation between Exxon and Shell stock prices.

Correlation matrix shows values close to 1 for opening and closing prices.

Autocorrelation (ACF/PACF): Both stocks are highly dependent on their previous day’s values.

🤖 Machine Learning Models Implemented

Lasso Regression

Provides good predictive accuracy but struggles with extreme values.

XGBoost

Achieves the best overall performance, capturing both stable trends and sudden price peaks.

Performed slightly better on Shell compared to Exxon.

LSTM (Long Short-Term Memory)

Deep learning model trained on historical sequences (time windows).

Captures general patterns but underperforms compared to XGBoost.

Achieved R² ≈ 68.2% for Shell and 65.5% for Exxon.

📈 Results & Key Insights

XGBoost outperformed other models, delivering the most reliable predictions.

LSTM showed moderate accuracy, but demonstrated the potential of deep learning in time-series forecasting.

Exxon stock proved more volatile, while Shell exhibited slightly more predictable behavior.

📂 Repository Contents

project → source code (data processing, ML models, evaluation)

report.pdf → detailed project report and methodology

HistoricalData → sample datasets (from Nasdaq.com)
