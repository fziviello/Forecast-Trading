import os
import pandas as pd
import argparse
import logging
from datetime import datetime
from config import BOT_TOKEN, CHANNEL_TELEGRAM
from utilities.utility import str_to_bool
from utilities.telegram_sender import TelegramSender
from utilities.folder_config import setup_folders, LOGS_FOLDER, LOG_STATISTICS_FILE_PATH, RESULTS_FOLDER, PLOTS_FOLDER
from utilities.plots import plot_statistics

PREFIX_VALIDATION = 'forecast_validation'
SEND_TELEGRAM = False
GENERATE_PLOT = False
SHOW_PLOT = False

setup_folders()

logging.basicConfig(
    filename=os.path.join(LOGS_FOLDER, LOG_STATISTICS_FILE_PATH),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def sendNotify(msg):
    if SEND_TELEGRAM is True:
        telegramSender = TelegramSender(BOT_TOKEN)
        telegramSender.sendMsg(msg, CHANNEL_TELEGRAM)

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
        placed_predictions = validation_df[pd.to_numeric(validation_df['Ticket'], errors='coerce').isna()].shape[0]
        successful_predictions = validation_df[validation_df['Risultato'].str.contains('Successo')].shape[0]
        failure_predictions = total_predictions - successful_predictions

        success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
        failure_rate = (failure_predictions / total_predictions * 100) if total_predictions > 0 else 0
        placed_rate = (placed_predictions / total_predictions * 100) if total_predictions > 0 else 0

        print(f"Statistiche di validazione per il simbolo \033[93m{symbol}\033[0m:\n")
        print(f"Totale Previsioni: \033[94m{total_predictions}\033[0m")
        print(f"Soddisfatte: \033[92m{successful_predictions} \033[92m({success_rate:.2f}%)\033[0m")
        print(f"Ordini Piazzati: \033[93m{placed_predictions} \033[93m({placed_rate:.2f}%)\033[0m")
        print(f"Insoddisfatte: \033[91m{failure_predictions} \033[91m({failure_rate:.2f}%)\033[0m\n")

        validation_stats = (
            f"Statistiche di validazione per {symbol}:\n"
            f"Totale Previsioni: {total_predictions}\n"
            f"Piazzate: {placed_predictions} ({placed_predictions:.2f}%)\n"
            f"Soddisfatte: {successful_predictions} ({success_rate:.2f}%)\n"
            f"Insoddisfatte: {failure_predictions} ({failure_rate:.2f}%)"
        )
        
        sendNotify(validation_stats)
        
        logging.info(validation_stats)
        return {'symbol': symbol, 'success_rate': success_rate, 'failure_rate': failure_rate}
    else:
        print(f"\033[91mImpossibile caricare i risultati di validazione per {symbol}\033[0m")
        logging.error(f"Impossibile caricare i risultati di validazione per {symbol}")
        return None

def process_all_symbol():
    global PLOTS_FOLDER, GENERATE_PLOT, SHOW_PLOT
    validation_files = [f for f in os.listdir(RESULTS_FOLDER) if f.startswith(PREFIX_VALIDATION) and f.endswith('.csv')]
    if not validation_files:
        print("\033[91mNessun file di validazione trovato.\033[0m")
        return

    statistics = []
    for file in validation_files:
        symbol = file.replace(f"{PREFIX_VALIDATION}_", "").replace(".csv", "").upper()
        logging.info(f"Trovato simbolo {symbol} da validare")
        stats = print_validation_statistics(symbol)
        if stats:
            statistics.append(stats)
    
    if statistics and GENERATE_PLOT is True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file_path = os.path.join(PLOTS_FOLDER, f"validation_statistics_{timestamp}.png")
        plot_statistics(statistics, plot_file_path, SHOW_PLOT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza le statistiche in base al simbolo")
    parser.add_argument("--notify", type=str, required=False, help="Invia notifica al canale telegram")
    parser.add_argument("--plot", type=str, required=False, help="Generare il grafico")
    parser.add_argument('--symbol', type=str, required=False, help="Simbolo per il quale effettuare le statistiche")
    parser.add_argument('--ALL', action='store_true', help="Analizza tutti i file di validazione disponibili")
    args = parser.parse_args()

    if args.notify is not None:
        SEND_TELEGRAM = str_to_bool(args.notify)

    if args.plot is not None:
        GENERATE_PLOT = str_to_bool(args.plot)
        
    if args.ALL:
        process_all_symbol()
    elif args.symbol:
        SYMBOL = args.symbol.upper()
        print_validation_statistics(SYMBOL)
    else:
        print("\033[91mErrore: Specificare '--symbol' o '--ALL'.\033[0m")