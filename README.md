# Model Training

The script contains that schedules model training.

## Customizable Parameters

You can change them in the config.py file

- `TIME_MINUTE_REPEAT`: Interval expressed in minutes of the schedule
- `N_REPEAT`: Number of repetitions

## Args Parameters

- `SYMBOL`: The Name of Stock Exchange Symbol separated by comma for multi-currency training (--symbol) *REQUIRED
- `NOTIFY`: If True send the predictions to Telegram Channel
- `SEND_SERVER_SIGNAL`: If True send signal to MT5 Server

## Run

- Start Training : `python3 scripts/training.py --symbol AUDJPY,AUDNZD,AUDCHF,SGDJPY,AUDCAD --notify True --sendSignal True`

# DataSet Generator

This Python script retrieves historical Forex data for a specified symbol using the `yfinance` library
The data can be visualized in a candlestick chart and saved to a CSV file

## Features

- Retrieves historical Forex data for a specified symbol
- Visualizes the data in a candlestick chart if enabled
- Saves the data to a CSV file named `DATASET_{SYMBOL}.csv`

## Customizable Parameters

The script contains several parameters that you can modify to suit your needs:

- `SHOW_PLOT`: If True Show the plot
- `RETRY_LIMIT`: Times to retry in case of error

## Args Parameters

- `SYMBOL`: The Name of Stock Exchange Symbol (--symbol) *REQUIRED
- `INTERVAL`: Dataset Range (--interval)
- `GENERATE_PLOT`: If True Make the plot (--plot)

## Run

- Create DataSet: `python3 scripts/create_dataSet.py --symbol AUDJPY`

# Forecast BOT

This script implements a Long Short-Term Memory (LSTM) neural network for predicting trading signals in Forex markets. It leverages historical price data to generate buy/sell signals, calculate potential profits and losses, and save predictions to a CSV file. The model also includes functionality for plotting results and managing previous model states. Can you see running on https://www.mql5.com/it/users/fziviello87

## Features

- Data Processing: Loads and preprocesses Forex historical data
- Model Training: Trains an LSTM model to predict trading signals based on historical data
- Predictions: Generates buy/sell signals and calculates potential profits and losses
- CSV Management: Saves predictions to a CSV file with options to overwrite existing data
- Visualization: Generates plots to visualize trading signals against historical prices
- Early Stopping: Implements early stopping to prevent overfitting during model training

## Usage

Use the dataset created with the script `create_dataSet`

## Dynamic Parameters

- `REPEAT_TRAINING`: If True restarts model training

## Customizable Parameters

- `GENERATE_PLOT`: If True Make the plot
- `SHOW_PLOT`: If True Show the plot

### Business Parameters

You can change them in the config.py file

- `MAX_MARGIN`: Maximum margin on price
- `MIN_MARGIN`: Minimum margin on the price
- `LOT_SIZE`: The number of lots
- `CONTRACT_SIZE`: Standard volume for one Forex lot
- `EXCHANGE_RATE`: The exchange rate for profit calculations
- `FAVORITE_RATE`: Preferred conversion currency (EUR)
- `N_PREDICTIONS`: The maximum number of predictions to generate
- `VALIDATION_THRESHOLD`: Model Validation Threshold
- `INTERVAL_MINUTES`: Dataset interval in minutes
- `RETRY_LIMIT`: Maximum number of retry
- `INTERVAL_DATASET`: Dataset interval in desired format
- `FORECAST_VALIDITY_MINUTES`: Validity of the forecast
- `TIME_MINUTE_REPEAT`: Repeat Training time
- `N_REPEAT`: Number of Repeat Training
- `BOT_TOKEN`: Token API Bot Telagram
- `CHANNEL_TELEGRAM`: Telegram Channel Name with @
- `PARAM_GRID`: Neural Network Parameters 
  - `units`: The number of neurons in the LSTM layers
  - `dropout`: The dropout rate to prevent overfitting
  - `epochs`: The number of training epochs
  - `batch_size`: The size of the batches used during training
  - `learning_rate`: The learning rate to optimize the weights
  - `optimizer`: The optimization algorithm (e.g. adam, rmsprop)

For env management you can create the `secret.env` file in the project root where you can create the production keys

## Args Parameters

- `SYMBOL`: The Name of Stock Exchange Symbol (--symbol) *REQUIRED
- `SEND_SERVER_SIGNAL`: If True send signal to MT5 Server
- `NOTIFY`: If True send the predictions to Telegram Channel
- `FAVORITE_RATE`: Favorite conversion rate (--favoriteRate) (default EUR)
- `INTERVAL_MINUTES`: Interval expressed in minutes to align with the dataset (--interval)
- `GENERATE_PLOT`: If True Make the plot (--plot)

## Run

- Start Forecast: `python3 scripts/forecast_bot.py --symbol AUDJPY --notify True --sendSignal True`

# Calculate Statistics

This script calculates the statistics obtained by the model during its training.

## Customizable Parameters

- `PREFIX_VALIDATION`: Validation file name prefix 

## Args Parameters

- `SYMBOL`: The Name of Stock Exchange Symbol (--symbol) *REQUIRED
- `ALL`: Analyze all available validation files (--ALL) *REQUIRED
- `NOTIFY`: If True send the predictions to Telegram Channel
- `GENERATE_PLOT`: If True Make the plot (--plot)

## Run

- Start Get Statistics for ALL symbol availables: `python3 scripts/get_statistics.py --ALL --notify True`
- Start Get Statistics for single symbol: `python3 scripts/get_statistics.py --symbol AUDJPY --notify True`

### Use Venv

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `.venv/bin/python`

## Requirements

`pip3 install -r requirements.txt`

if you have problems installing ta-lib proceed as follows
- `brew install ta-lib`
- `TA_INCLUDE_PATH=$(brew --prefix ta-lib)/include`
- `TA_LIBRARY_PATH=$(brew --prefix ta-lib)/lib`
- `CFLAGS="-I$TA_INCLUDE_PATH" LDFLAGS="-L$TA_LIBRARY_PATH" pip install ta-lib`

if you use windows:
- Download ta-lib precompiled `https://sourceforge.net/projects/talib-whl` and move into venv dir
- Run `pip install path\to\ta_lib‑0.4.0‑cp310‑cp310‑win_amd64.whl`


To use the `SEND_SERVER_SIGNAL` functionality you can use APIM MT5 
- `https://github.com/fziviello/APIM_MT5`


![screenshot1](/Screenshot_1.png?raw=true)

![screenshot1](/Screenshot_2.png?raw=true)
