MAX_MARGIN = 0.03  # 3% del prezzo
MIN_MARGIN = 0.007 # 0.7% del prezzo
LOT_SIZE = 0.01
CONTRACT_SIZE = 100_000
EXCHANGE_RATE = 0.1
FAVORITE_RATE = "EUR"
N_PREDICTIONS = 10
VALIDATION_THRESHOLD = 0.1
INTERVAL_MINUTES = 2
RETRY_LIMIT = 3
INTERVAL_DATASET = '2m'
FORECAST_VALIDITY_MINUTES = 30
TIME_MINUTE_REPEAT = 5
N_REPEAT = 60
BOT_TOKEN = ""
CHANNEL_TELEGRAM = ""

PARAM_GRID = {
    'units': [50, 75, 100],
    'dropout': [0.2],
    'epochs': [50, 75, 100],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.005, 0.01],
    'optimizer': ['adam', 'rmsprop']
}