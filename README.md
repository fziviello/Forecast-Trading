# Model Training

The script contains that schedules model training

## Customizable Parameters

The script contains several parameters that you can modify to suit your needs:

- `SCRIPT_CREATE_DATASET`: File path script create dataset
- `SCRIPT_FORECAST`: File path script forecast
- `LOG_FOLDER`:  Dir path script create logs
- `LOG_FILE_PATH`: File path where Logs
- `TIME_MINUTE_REPEAT`: Interval expressed in minutes of the schedule
- `N_REPEAT`: Number of repetitions

## Args Parameters

- `SYMBOL`: The Name of Stock Exchange Symbol separated by comma for multi-currency training (--symbol) *REQUIRED

## Run

- Start Training : `python3 training.py --symbol AUDJPY,AUDCHF`

# DataSet Generator

This Python script retrieves historical Forex data for a specified symbol using the `yfinance` library. 
The data can be visualized in a candlestick chart and saved to a CSV file.

## Features

- Retrieves historical Forex data for a specified symbol.
- Visualizes the data in a candlestick chart if enabled.
- Saves the data to a CSV file named `DATASET_{SYMBOL}.csv`

## Customizable Parameters

The script contains several parameters that you can modify to suit your needs:

- `LOG_FOLDER`:  Dir path script create logs
- `LOG_FILE_PATH`: File path where Logs
- `DATA_FOLDER`:  Dir path script create dataset
- `PLOT_FOLDER`: 'Dir path script create plot
- `SHOW_PLOT`: If True Show the plot
- `RETRY_LIMIT`: Times to retry in case of error

## Args Parameters

- `SYMBOL`: The Name of Stock Exchange Symbol (--symbol) *REQUIRED
- `INTERVAL`: Dataset Range (--interval)
- `GENERATE_PLOT`: If True Make the plot (--plot)

## Run

- Create DataSet: `python3 create_dataSet.py --symbol AUDJPY`

# Forecast BOT

This script implements a Long Short-Term Memory (LSTM) neural network for predicting trading signals in Forex markets. It leverages historical price data to generate buy/sell signals, calculate potential profits and losses, and save predictions to a CSV file. The model also includes functionality for plotting results and managing previous model states.

## Features

- Data Processing: Loads and preprocesses Forex historical data.
- Model Training: Trains an LSTM model to predict trading signals based on historical data.
- Predictions: Generates buy/sell signals and calculates potential profits and losses.
- CSV Management: Saves predictions to a CSV file with options to overwrite existing data.
- Visualization: Generates plots to visualize trading signals against historical prices.
- Early Stopping: Implements early stopping to prevent overfitting during model training.

## Usage
Use the dataset created with the script `create_dataSet`

## Dynamic Parameters

- `MODEL_PATH`: File path where the trained LSTM
- `SCALER_PATH`: File path where the MinMaxScaler object is saved
- `FORECAST_RESULTS_PATH`: File path where Forecast Result
- `DATASET_PATH`: File path where DataSet
- `VALIDATION_RESULTS_PATH`: File path where Validation Result
- `LOG_FILE_PATH`: File path where Logs
- `PLOT_FILE_PATH`: Path to the file where the graphs are located
- `REPEAT_TRAINING`: If True restarts model training

## Customizable Parameters

- `MODELS_FOLDER`:  Dir path script create models
- `DATA_FOLDER`:  Dir path script import dataset
- `RESULTS_FOLDER`: Dir path script save predictions and validation
- `LOG_FOLDER`:  Dir path script create logs
- `PLOT_FOLDER`: 'Dir path script create plot
- `OVERWRITE_FORECAST_CSV`: Set to True to overwrite the existing predictions CSV; False to append.
- `VALIDATION_THRESHOLD`: Error threshold to enable retraining
- `GENERATE_PLOT`: If True Make the plot
- `SHOW_PLOT`: If True Show the plot

### Business Parameters

- `MARGIN_PROFIT`: The profit margin to calculate take profit levels.
- `LEVERAGE`: The leverage to apply to trades.
- `UNIT`: The number of units traded.
- `EXCHANGE_RATE`: The exchange rate for profit calculations.
- `DATA_MODEL_RATE`: Model reference currency
- `FAVORITE_RATE`: Preferred conversion currency
- `N_PREDICTIONS`: The maximum number of predictions to generate.

## Args Parameters

- `SYMBOL`: The Name of Stock Exchange Symbol (--symbol) *REQUIRED
- `FAVORITE_RATE`: Favorite conversion rate (--favoriteRate) (default EUR)
- `INTERVAL_MINUTES`: Interval expressed in minutes to align with the dataset (--interval)
- `GENERATE_PLOT`: If True Make the plot (--plot)

## Run

- Start Forecast: `python3 forecast_bot.py --symbol AUDJPY`

## Requirements

Make sure to have the following libraries installed:

- `yfinance`
- `finplot`
- `pandas`
- `numpy`
- `joblib`  
- `scikit-learn`  
- `matplotlib`  
- `tensorflow`  

### Use Venv

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `.venv/bin/python`

You can install them using pip:

`pip3 install -r requirements.txt`

## Use Notebook

- The notebook can be modified to analyze other forex pairs by simply changing the value of SYMBOL.

- To periodically update the dataset, the notebook can be scheduled to run automatically with a task scheduler or other automation tools.

![screenshot1](/Screenshot_1.png?raw=true)

![screenshot1](/Screenshot_2.png?raw=true)