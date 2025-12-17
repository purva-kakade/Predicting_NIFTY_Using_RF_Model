# Predicting NIFTY Using a Random Forest Classifier

A machine learning project that predicts the directional movement (Up/Down) of the Nifty 50 index using Random Forest Classification.

## 📝 Overview

This project was built to explore the application of Machine Learning in financial markets. It uses historical data from Yahoo Finance to train a **Random Forest Classifier** that predicts whether the Nifty 50 closing price will be higher tomorrow compared to today.

Unlike simple time-series forecasting, this model incorporates **global macro-indicators** (S&P 500 and Singapore STI) and technical rolling averages to improve prediction precision.

## ✨ Key Features

* **Global Market Correlation:** Incorporates S&P 500 and STI (Singapore) data to capture global market sentiment influencing the Indian market.
* **Feature Engineering:** Uses rolling averages (2, 5, 60, 250, 1000 days) and trend ratios to capture short-term momentum and long-term trends.
* **Robust Backtesting:** Includes a custom backtesting engine that simulates trading over historical data with a sliding window approach to prevent look-ahead bias.
* **Precision Focused:** The model is tuned and evaluated using `precision_score` to minimize false positives.

## 🛠️ Technologies Used

* **Python 3.x**
* **yfinance:** For fetching real-time and historical market data.
* **scikit-learn:** For the Random Forest Classifier and metrics.
* **pandas:** For data manipulation and time-series analysis.

## 🚀 How to Run

1.  **Install Dependencies:**
    You need to install the required libraries first:
    ```bash
    pip install pandas yfinance scikit-learn
    ```

2.  **Run the Script:**
    Download `stock_predictor.py` and run it in your Python environment.

## 📊 How it Works

1.  **Data Ingestion:** Fetches `^NSEI` (Nifty 50), `^GSPC` (S&P 500), and `^STI` (Straits Times Index).
2.  **Preprocessing:** Aligns timezones, fills missing data using forward filling, and calculates daily percentage changes.
3.  **Training:** The model trains on a rolling basis (e.g., using the first 10 years to predict the next year, then sliding forward).
4.  **Thresholding:** A custom probability threshold (60%) is used—the model only "bets" if it is more than 60% confident.

## ⚠️ Disclaimer

**For Educational Purposes Only.** This code is a learning project and should not be used for actual financial trading. Stock market prediction is volatile and risky.
