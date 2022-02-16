# Kaggle-Scripts
It contains the scripts I used in Kaggle competition "G-Research Crypto Forecasting", https://www.kaggle.com/c/g-research-crypto-forecasting, but they are also generally applicable into various Kaggle competitions. 

The LGBMRegressor.py is the main script calling the specially constructed time-series CV splitting function and one feature engineering class of moving average. 



## (1) Data Description


timestamp: All timestamps are returned as second Unix timestamps (the number of seconds elapsed since 1970-01-01 00:00:00.000 UTC). Timestamps in this dataset are multiple of 60, indicating minute-by-minute data.

Asset_ID: The asset ID corresponding to one of the crytocurrencies (e.g. Asset_ID = 1 for Bitcoin). The mapping from Asset_ID to crypto asset is contained in asset_details.csv.

Count: Total number of trades in the time interval (last minute).

Open: Opening price of the time interval (in USD).

High: Highest price reached during time interval (in USD).

Low: Lowest price reached during time interval (in USD).

Close: Closing price of the time interval (in USD).

Volume: Quantity of asset bought or sold, displayed in base currency USD.

VWAP: The average price of the asset over the time interval, weighted by volume. VWAP is an aggregated form of trade data.

Target: Residual log-returns for the asset over a 15 minute horizon.
supplemental_train.csv After the submission period is over this file's data will be replaced with cryptoasset prices from the submission period. In the Evaluation phase, the train, train supplement, and test set will be contiguous in time, apart from any missing data. The current copy, which is just filled approximately the right amount of data from train.csv is provided as a placeholder.

asset_details.csv Provides the real name and of the cryptoasset for each Asset_ID and the weight each cryptoasset receives in the metric. Weights are determined by the logarithm of each product's market cap (in USD), of the cryptocurrencies at a fixed point in time. Weights were assigned to give more relevance to cryptocurrencies with higher market volumes to ensure smaller cryptocurrencies do not disproportionately impact the models.

example_sample_submission.csv An example of the data that will be delivered by the time series API. The data is just copied from train.csv.

example_test.csv An example of the data that will be delivered by the time series API.

