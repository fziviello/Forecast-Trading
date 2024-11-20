import os
import pandas as pd
import argparse
import logging
from utilities.folder_config import setup_folders, LOGS_FOLDER, LOG_STATISTICS_FILE_PATH, RESULTS_FOLDER

PREFIX_VALIDATION = 'forecast_validation'

setup_folders()

logging.basicConfig(filename=os.path.join(LOGS_FOLDER, LOG_STATISTICS_FILE_PATH), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_validation_results(symbol):
    validation_file_path = os.path.join(RESULTS_FOLDER, f'{PREFIX_VALIDATION}_{symbol}.csv')
    if os.path.exists(validation_file_path):
        validation_df = pd.read_csv(validation_file_path)
        logging.info(f"Risultati di validazione per {symbol} caricati")
        return validation_df
    else:
        print(f"\033[91mFile di validazione {symbol} non trovato.\033[0m")
        logging.error(f"File di validazione {symbol} non trovato.")
        return None

def print_validation_statistics(symbol):
    validation_df = get_validation_results(symbol)
    if validation_df is not None:
        total_predictions = len(validation_df)
        successful_predictions = validation_df[validation_df['Risultato'].str.contains('Successo')].shape[0]
        failure_predictions = total_predictions - successful_predictions
        failure_rate = failure_predictions / total_predictions if total_predictions > 0 else 0

        print(f"Statistiche di validazione per il simbolo \033[93m{symbol}\033[0m:\n")
        print(f"Totale Previsioni: \033[94m{total_predictions}\033[0m")
        print(f"Soddisfatte: \033[92m{successful_predictions}\033[0m")
        print(f"Insoddisfatte: \033[91m{failure_predictions}\033[0m")
        print(f"Fallimento: \033[91m{failure_rate:.2f}%\n\033[0m")

        logging.info(f"Statistiche di validazione per {symbol}:\n"
                     f"Totale previsioni: {total_predictions}, Successo: {successful_predictions}, Fallimento: {failure_predictions}, Tasso di fallimento: {failure_rate:.2f}")
    else:
        print(f"\033[91mImpossibile caricare i risultati di validazione per {symbol}\033[0m")
        logging.error(f"\033[91mImpossibile caricare i risultati di validazione per {symbol}\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza le statistche in base al simbolo")
    parser.add_argument('--symbol', type=str, required=True, help="Simbolo per il quale effettuare le statistiche")
    args = parser.parse_args()

    SYMBOL = args.symbol.upper()
    
    print_validation_statistics(SYMBOL)
