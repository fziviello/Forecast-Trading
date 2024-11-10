import os
import logging
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

LOG_FOLDER = 'LOGS'
LOG_FILE_PATH = 'create_dataSet.log'
DATA_FOLDER = 'DATASET'
PLOT_FOLDER = 'PLOT'
GENERATE_PLOT = False
SHOW_PLOT = False
SYMBOL = 'AUDJPY'
INTERVAL = '30m'
RETRY_LIMIT = 3 

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
    print(f"\033[92mCartella '{LOG_FOLDER}' creata con successo.\033[0m")
    
logging.basicConfig(filename=os.path.join(LOG_FOLDER, LOG_FILE_PATH), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

today = datetime.now()
if INTERVAL == "30m":
    ndays = 59
else:
    ndays = 730

startDate = today - timedelta(days=ndays)

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"\033[92mCartella '{DATA_FOLDER}' creata con successo.\033[0m")

if not os.path.exists(PLOT_FOLDER) and GENERATE_PLOT:
    os.makedirs(PLOT_FOLDER)
    print(f"\033[92mCartella '{PLOT_FOLDER}' creata con successo.\033[0m")    

def getForexData(symbol):
    symbol += "=X"
    attempt = 0
    while attempt < RETRY_LIMIT:
        try:
            sym = yf.Ticker(symbol)
            data = sym.history(start=startDate, end=today, interval=INTERVAL)
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

forex_data = getForexData(SYMBOL)

if forex_data is not None:
    if GENERATE_PLOT:
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plotName = os.path.join(PLOT_FOLDER, f"{SYMBOL}_{timestamp}.png")
        
        if SHOW_PLOT:
            mpf.plot(
                forex_data.set_index('Datetime'),
                type='candle', 
                style='charles',
                title=f"{SYMBOL} Forex Data",
                ylabel='Prezzo',
                volume=False,
                warn_too_much_data=len(forex_data) + 100
            )
            mpf.show()
        
        mpf.plot(
            forex_data.set_index('Datetime'),
            type='candle',
            style='charles',
            title=f"{SYMBOL} Forex Data",
            ylabel='Prezzo',
            volume=False,
            savefig=plotName,
            warn_too_much_data=len(forex_data) + 100
        )
        print(f"\033[92mGrafico salvato con successo in '{plotName}'\033[0m")
    
    nameCSV = os.path.join(DATA_FOLDER, "DATASET_" + SYMBOL + ".csv")
    
    forex_data.to_csv(nameCSV, index=False)
    logging.info(f"Dataset generato con successo in '{nameCSV}'")
    print(f"\033[92mDataset generato con successo in '{nameCSV}'\033[0m")
else:
    logging.error(f"Impossibile generare il dataset a causa di un errore nel recupero dei dati.")
    print(f"\033[91mImpossibile generare il dataset a causa di un errore nel recupero dei dati.\033[0m")
