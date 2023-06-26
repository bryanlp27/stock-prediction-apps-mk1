# stock-prediction-apps-mk1

# Stock Prediction App

This is a web application built using Streamlit that utilizes the FBProphet machine learning library for stock prediction. The app provides predictions for the stocks of popular companies such as Google, Apple, Microsoft, and Tesla.

## Installation

1. Clone the repository:

   git clone <repository_url>

2. Navigate to the project directory:

   cd stock-prediction-app

3. Install the required dependencies using pip:

   pip install -r requirements.txt

## Usage

1. Run the Streamlit app:

   streamlit run app.py

2. The web application will open in your browser.

3. Select the stock you want to predict from the dropdown menu.

4. Adjust the parameters (Years of Prediction, etc.) using the sliders.

5. The predicted stock prices will be displayed in a line chart, along with the historical stock prices.

6. You can experiment with different stocks and parameter settings to observe the predictions.

## About FBProphet

FBProphet is a popular open-source library developed by Facebook for time series forecasting. It is designed to handle the inherent characteristics of time series data, such as trends, seasonality, and holidays. By utilizing Prophet's forecasting capabilities, this app provides users with predictions for stock prices based on historical data.

Please note that the predictions generated by this app are for informational purposes only and should not be considered as financial advice. Always do your own research and consult with a qualified financial professional before making any investment decisions.

## Acknowledgements

- The FBProphet library: [Prophet](https://facebook.github.io/prophet/)
- Stock data: [Yahoo Finance](https://finance.yahoo.com/)

Streamlit Model Deploy Link = https://stock-prediction-apps-mk1.streamlit.app/
