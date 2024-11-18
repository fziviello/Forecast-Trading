MARGIN_PROFIT = 0.002
LEVERAGE = 0.01
UNIT = 1000
EXCHANGE_RATE = 1.0
FAVORITE_RATE = "EUR"
N_PREDICTIONS = 10
VALIDATION_THRESHOLD = 0.1
INTERVAL_MINUTES = 2
RETRY_LIMIT = 3
INTERVAL = '2m'
FORECAST_VALIDITY_MINUTES = 30
TIME_MINUTE_REPEAT = 5
N_REPEAT = 60
BOT_TOKEN = ""
CHANNEL_TELEGRAM = "@SignalFaz"

PARAM_GRID = {
    'units': [50, 75, 100],
    'dropout': [0.2, 0.3],
    'epochs': [50, 75, 100],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.005, 0.01],
    'optimizer': ['adam', 'rmsprop']
}