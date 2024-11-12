import os
import logging
import argparse
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
from config import RETRY_LIMIT, INTERVAL
from folder_config import setup_folders, DATA_FOLDER, PLOTS_FOLDER, LOGS_FOLDER, LOG_DATASET_FILE_PATH

GENERATE_PLOT = False
SHOW_PLOT = False

setup_folders()
logging.basicConfig(filename=os.path.join(LOGS_FOLDER, LOG_DATASET_FILE_PATH), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def getForexData(interval,symbol):
    
    if interval == "1m":
        ndays = 7
    elif interval == "2m":
        ndays = 59
    elif interval == "5m":
        ndays = 59
    elif interval == "15m":
        ndays = 59   
    elif interval == "30m":
        ndays = 59
    elif interval == "60m":
        ndays = 730
    elif interval == "90m":
        ndays = 59      
    elif interval == "1h":
        ndays = 730
    else:
        logging.error(f"Errore: Intevallo '{interval}' non gestito")
        exit (f"\033[91mErrore: Intevallo '{interval}' non gestito\033[0m")
        
    today = datetime.now()

    startDate = today - timedelta(days=ndays)
    symbol += "=X"
    attempt = 0
    
    while attempt < RETRY_LIMIT:
        try:
            sym = yf.Ticker(symbol)
            data = sym.history(start=startDate, end=today, interval=interval)
            if data.empty:
                logging.error(f"Nessun dato recuperato per {symbol}")
                raise ValueError(f"\033[93mNessun dato recuperato per {symbol}\033[0m")
            
            data.reset_index(inplace=True)
            return data[['Datetime', 'Open', 'High', 'Low', 'Close']]
        
        except Exception as e:
            attempt += 1
            logging.error(f"Errore durante il recupero dei dati (tentativo {attempt}/{RETRY_LIMIT}): {e}")
            print(f"\033[91mErrore durante il recupero dei dati (tentativo {attempt}/{RETRY_LIMIT}): {e}\033[0m")
            if attempt >= RETRY_LIMIT:
                logging.error(f"Fallimento dopo {RETRY_LIMIT} tentativi. Impossibile recuperare i dati.")
                print(f"\033[91mFallimento dopo {RETRY_LIMIT} tentativi. Impossibile recuperare i dati.\033[0m")
                return None

def run_create_dataSet(interval,symbol):
     
    forex_data = getForexData(interval,symbol)

    if forex_data is not None:
        if GENERATE_PLOT:
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plotName = os.path.join(PLOTS_FOLDER, f"{symbol}_{timestamp}.png")
            
            if SHOW_PLOT:
                mpf.plot(
                    forex_data.set_index('Datetime'),
                    type='candle', 
                    style='charles',
                    title=f"{symbol} Forex Data",
                    ylabel='Prezzo',
                    volume=False,
                    warn_too_much_data=len(forex_data) + 100
                )
                mpf.show()
            
            mpf.plot(
                forex_data.set_index('Datetime'),
                type='candle',
                style='charles',
                title=f"{symbol} Forex Data",
                ylabel='Prezzo',
                volume=False,
                savefig=plotName,
                warn_too_much_data=len(forex_data) + 100
            )
            print(f"\033[92mGrafico salvato con successo in '{plotName}'\033[0m")
        
        nameCSV = os.path.join(DATA_FOLDER, "DATASET_" + symbol + ".csv")
        
        forex_data.to_csv(nameCSV, index=False)
        logging.info(f"Dataset generato con successo con intervallo '{interval}' in '{nameCSV}'")
        print(f"\033[92mDataset generato con successo con intervallo '{interval}' in '{nameCSV}'\033[0m")
    else:
        logging.error(f"Impossibile generare il dataset a causa di un errore nel recupero dei dati.")
        print(f"\033[91mImpossibile generare il dataset a causa di un errore nel recupero dei dati.\033[0m")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Inserire l' intervallo temporale ed il symbol da esaminare")
    parser.add_argument("--plot", type=str, required=False, help="Generare il grafico")
    parser.add_argument("--interval", type=str, required=False, help="Inserire l'intervallo temporale")
    parser.add_argument("--symbol", type=str, required=True, help="Inserire il symbol")
    args = parser.parse_args()
    SYMBOL = (args.symbol).upper()
        
    if args.interval is not None :
        INTERVAL = args.interval
    
    if args.plot is not None:
        GENERATE_PLOT = args.plot
            
    run_create_dataSet(INTERVAL,SYMBOL)