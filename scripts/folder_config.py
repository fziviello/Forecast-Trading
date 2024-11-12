import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG_DATASET_FILE_PATH = 'create_dataSet.log'
LOG_FORECAST_FILE_PATH = 'forecast_bot.log'
LOG_TRAINING_FILE_PATH = 'training.log'
LOG_STATISTICS_FILE_PATH = 'get_statistic.log'

MODELS_FOLDER = BASE_PATH + '/models'
DATA_FOLDER = BASE_PATH + '/dataset'
RESULTS_FOLDER = BASE_PATH + '/results'
PLOTS_FOLDER = BASE_PATH + '/plots'
LOGS_FOLDER = BASE_PATH + '/logs'

def setup_folders():
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
        print(f"\033[92mCartella '{LOGS_FOLDER}' creata con successo.\033[0m")
        
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
        print(f"\033[92mCartella '{MODELS_FOLDER}' creata con successo.\033[0m")    

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print(f"\033[92mCartella '{RESULTS_FOLDER}' creata con successo.\033[0m")    

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"\033[92mCartella '{DATA_FOLDER}' creata con successo.\033[0m")
    
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)
        print(f"\033[92mCartella '{PLOTS_FOLDER}' creata con successo.\033[0m")      
