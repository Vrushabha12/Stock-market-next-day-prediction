📈 Stock Market Next-Day Close Price Prediction
This project builds a machine learning model to predict the next-day closing price of stocks using historical market data. The dataset includes multiple stock attributes such as open price, volume traded, market cap, and more.

🔍 Project Overview
The goal is to forecast the next day's close price using a Random Forest Regression model, based on engineered features from a historical dataset.

📁 Dataset
File Name: stock_market_june2025.csv

Columns Used:

Date

Ticker

Open Price

High Price

Low Price

Close Price

Volume Traded

Market Cap

PE Ratio

Dividend Yield

EPS

52 Week High

52 Week Low

Sector

The dataset is expected to be uploaded as a .zip file and automatically extracted during runtime.

🧠 Features Used
Open Price

High Price

Low Price

Volume Traded (log-transformed)

Market Cap (log-transformed)

PE Ratio

Dividend Yield

EPS

52 Week High

52 Week Low

High-Low Spread (Engineered)

Close-Open Difference (Engineered)

Target variable: Next-Day Close Price

🧪 Model & Evaluation
Model: RandomForestRegressor from sklearn

Train/Test Split: 80% / 20%

Evaluation Metrics:

R² Score

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

📊 Example Output
yaml
Copy
Edit
📊 Next-Day Close Price Prediction:
R² Score : 0.8123
RMSE     : 12.3456
MAE      : 9.8765
📈 Visualization
A scatter plot compares the actual vs. predicted next-day close prices:

Blue dots: individual predictions

Red dashed line: ideal prediction line (perfect prediction)

🛠 Requirements
Python 3.x

pandas

numpy

scikit-learn

matplotlib

Google Colab environment (for file upload handling)

🚀 How to Run
Upload the .zip file containing stock_market_june2025.csv to Colab.

Run all cells from top to bottom.

Observe evaluation metrics and prediction plot.

📌 Notes
Missing values are handled.

The script performs log-transformation to normalize skewed features.

Feature engineering is applied to boost prediction performance.

