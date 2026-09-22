# Stock Prediction: ExxonMobil vs Shell

A comparative machine learning project for **financial time-series analysis and stock-price forecasting**, using historical market data from **ExxonMobil (XOM)** and **Shell**.

The project explores statistical relationships between the two energy companies and compares different machine learning approaches, from regularized linear regression to gradient boosting and deep learning.

---

## Project Overview

### Objective

The objective is to investigate whether historical market information can be used to model and forecast stock prices, while comparing the performance of different machine learning techniques.

The analysis focuses on:

* Exploratory data analysis
* Cross-company correlation
* Time-series dependence
* Machine learning regression
* Gradient boosting
* Deep learning with LSTM networks
* Model performance evaluation

### Companies

| Company    | Ticker | Sector |
| ---------- | ------ | ------ |
| ExxonMobil | XOM    | Energy |
| Shell      | SHEL   | Energy |

Both companies operate in the global energy sector, making them useful subjects for a comparative analysis of financial time series.

---

## Dataset

The project uses historical daily market data obtained from **Nasdaq historical stock data**.

Each dataset contains:

* Date
* Close/Last
* Open
* High
* Low
* Volume

The datasets contain **1,255 observations per company**, covering approximately July 2020 – July 2025.

---

## Methodology

The project is structured into four main stages.

### 1. Data Preparation

The raw datasets are loaded with Pandas and prepared for analysis.

Main preprocessing steps include:

* Converting price variables from strings to numerical values
* Removing currency symbols
* Converting dates to `datetime`
* Checking data types and descriptive statistics
* Merging ExxonMobil and Shell observations by date

```python
HistoricalDataExxon = pd.read_csv("HistoricalDataExxon.csv")
HistoricalDataShell = pd.read_csv("HistoricalDataShell.csv")
```

---

### 2. Exploratory Data Analysis

The analysis begins by examining the historical behavior of both stocks.

#### Price comparison

Historical closing prices are visualized to compare the evolution of ExxonMobil and Shell over time.

#### Correlation analysis

A correlation matrix is used to investigate relationships between:

* Open
* High
* Low
* Closing prices
* Trading volume

The analysis shows a **strong positive relationship between the two stocks' price variables**.

#### Time-series analysis

Autocorrelation and partial autocorrelation functions (ACF/PACF) are calculated for both companies to investigate temporal dependence in closing prices.

---

## Machine Learning Models

### Lasso Regression

A regularized linear regression model is used as a baseline machine learning approach.

The model is evaluated using the **R² score**.

**ExxonMobil test R²: 0.899**

---

### XGBoost

The project then applies **XGBoost regression** to the merged ExxonMobil/Shell dataset.

The implementation includes:

* Chronological ordering of observations
* A 4-day Simple Moving Average (SMA)
* 300 boosting estimators
* Learning rate: `0.03`
* Maximum tree depth: `6`
* Row subsampling: `0.8`
* Feature subsampling: `0.8`

The model is evaluated using:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R²

### XGBoost Results

| Model Target |    MSE |    MAE |         R² |
| ------------ | -----: | -----: | ---------: |
| ExxonMobil   | 2.3245 | 1.1788 | **0.9279** |
| Shell        | 0.7243 | 0.6284 | **0.9112** |

Within the models tested in this project, XGBoost produced the strongest reported performance.

---

### LSTM

A Long Short-Term Memory neural network is used to model sequential price information.

The network uses:

* 60-day historical sequences
* MinMax normalization
* 2 LSTM layers
* 64 units per layer
* Dropout: 20%
* Dense output layer
* Adam optimizer
* Mean Squared Error loss
* Early stopping

The model is trained separately for ExxonMobil and Shell.

### LSTM Results

| Company    |     MSE |    MAE |         R² |
| ---------- | ------: | -----: | ---------: |
| ExxonMobil | 10.4756 | 2.5074 | **0.6737** |
| Shell      |  2.8663 | 1.1708 | **0.6936** |

---

## Results

The main findings from the implemented experiments are:

1. ExxonMobil and Shell exhibit strong positive relationships across several market variables.
2. Both stocks show significant temporal dependence in their historical closing prices.
3. **XGBoost achieved higher reported test performance than the LSTM models** in this implementation.
4. The LSTM models were able to capture part of the underlying temporal structure, but produced lower R² scores than XGBoost.
5. Shell produced lower forecast errors than ExxonMobil in the reported XGBoost and LSTM experiments.

---

## Technologies

### Programming & Data Analysis

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

### Statistics & Time Series

* Statsmodels
* ACF / PACF analysis

### Machine Learning

* Scikit-learn
* Lasso Regression
* Linear Regression
* XGBoost

### Deep Learning

* TensorFlow
* Keras
* LSTM
* Dropout
* Early Stopping

---

## Repository Structure

```text
StockPrediction-Exxon-Shell/
│
├── HistoricalDataExxon.csv
├── HistoricalDataShell.csv
├── project.ipynb
├── relazione.pdf
├── README.md
└── .gitattributes
```

### Files

**`project.ipynb`**
Complete Python implementation of data preparation, exploratory analysis, machine learning models and evaluation.

**`HistoricalDataExxon.csv`**
Historical ExxonMobil market data.

**`HistoricalDataShell.csv`**
Historical Shell market data.

**`relazione.pdf`**
Detailed project report and methodology.

---

## Key Takeaways

This project combines **financial analysis, statistics and machine learning** to study two major energy companies.

Rather than focusing on a single forecasting technique, the analysis compares different approaches and evaluates how they perform on the same financial time-series problem.

The project therefore provides practical experience in:

> **Data Preparation → Exploratory Analysis → Time-Series Analysis → Machine Learning → Deep Learning → Model Evaluation**

---

## Limitations

The results should be interpreted as an **academic machine learning experiment**, not as a trading system or investment strategy.

In particular:

* The dataset covers a limited historical period and only two companies.
* Stock prices are affected by many variables that are not included in the dataset.
* The different models use different preprocessing and validation approaches.
* The reported metrics measure predictive performance within the experimental setup and should not be interpreted as evidence of reliable future investment returns.

Future extensions could include:

* Lagged financial features
* Returns instead of raw prices
* Rolling/expanding time-series validation
* Hyperparameter optimization
* Additional market and macroeconomic variables
* Broader company and sector coverage

---

## Author

**Francesco Macca**
Economics & Big Data — Roma Tre University

[GitHub](https://github.com/francescomacca)
