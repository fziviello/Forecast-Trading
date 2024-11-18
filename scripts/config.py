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
BOT_TOKEN = "7439330565:AAFNP8E91iIaHfbs-F0apaYXrMCMz2hivqQ"
CHANNEL_TELEGRAM = "@SignalFaz"

PARAM_GRID = {
    'units': [50, 75, 100],
    'dropout': [0.2, 0.3],
    'epochs': [50, 75, 100],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.005, 0.01],
    'optimizer': ['adam', 'rmsprop']
}

#Tempo di elaborazione migliore configurazione per AUDJPY: 13:21:49
#Migliori parametri trovati per AUDJPY: {'units': 75, 'dropout': 0.2, 'epochs': 50, 'batch_size': 32, 'learning_rate': 0.001, 'optimizer': 'rmsprop'} con accuracy=1.000%à#
#Migliori parametri trovati per AUDJPY: {'units': 50, 'dropout': 0.2, 'epochs': 50, 'batch_size': 16, 'learning_rate': 0.01, 'optimizer': 'adam'} con accuracy=0.700%

#Tempo di elaborazione migliore configurazione per AUDNZD: 15:15:36
#Migliori parametri trovati per AUDNZD: {'units': 50, 'dropout': 0.2, 'epochs': 50, 'batch_size': 16, 'learning_rate': 0.001, 'optimizer': 'adam'} con accuracy=0.700%