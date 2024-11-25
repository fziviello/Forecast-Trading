from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'secret.env'))

MAX_MARGIN = 0.03 # 3% del prezzo
MIN_MARGIN = 0.007 # 0.7% del prezzo
LOT_SIZE = 0.01
CONTRACT_SIZE = 100_000
EXCHANGE_RATE = 0.1
FAVORITE_RATE = "EUR"
N_PREDICTIONS = 10
VALIDATION_THRESHOLD = 0.1
INTERVAL_MINUTES = 10
RETRY_LIMIT = 3
INTERVAL_DATASET = '2m'
FORECAST_VALIDITY_MINUTES = 30
TIME_MINUTE_REPEAT = 5
N_REPEAT = 60
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_TELEGRAM = os.getenv("CHANNEL_TELEGRAM")
IP_SERVER_TRADING = os.getenv("IP_SERVER_TRADING")
PORT_SERVER_TRADING = os.getenv("PORT_SERVER_TRADING")

PARAM_GRID = {
    'units': [50, 75, 100],
    'dropout': [0.2],
    'epochs': [50, 75, 100],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.005, 0.01],
    'optimizer': ['adam', 'rmsprop']
}