import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

GENERATE_PLOT = False
SYMBOL = 'AUDJPY'
INTERVAL = '30m'
RETRY_LIMIT = 3

today = datetime.now()
if INTERVAL == "30m":
    ndays = 59
else:
    ndays = 730

startDate = today - timedelta(days=ndays)

def getForexData(symbol):
    symbol += "=X"
    attempt = 0
    while attempt < RETRY_LIMIT:
        try:
            sym = yf.Ticker(symbol)
            data = sym.history(start=startDate, end=today, interval=INTERVAL)
            if data.empty:
                raise ValueError(f"\033[93mNessun dato recuperato per {symbol}\033[0m")
            
            data.reset_index(inplace=True)
            return data[['Datetime', 'Open', 'High', 'Low', 'Close']]
        
        except Exception as e:
            attempt += 1
            print(f"\033[91mErrore durante il recupero dei dati (tentativo {attempt}/{RETRY_LIMIT}): {e}\033[0m")
            if attempt >= RETRY_LIMIT:
                print(f"\033[91mFallimento dopo {RETRY_LIMIT} tentativi. Impossibile recuperare i dati.\033[0m")
                return None

forex_data = getForexData(SYMBOL)

if forex_data is not None:
    if GENERATE_PLOT:
        
        plotName = "PLOT_" + SYMBOL + ".png" 
        
        mpf.plot(
            forex_data.set_index('Datetime'),
            type='candle', 
            style='charles',
            title=f"{SYMBOL} Forex Data",
            ylabel='Prezzo',
            volume=False
        )
        mpf.show()
        
        mpf.plot(
            forex_data.set_index('Datetime'),
            type='candle',
            style='charles',
            title=f"{SYMBOL} Forex Data",
            ylabel='Prezzo',
            volume=False,
            savefig=plotName
        )
        print(f"\033[92m{'Grafico salvato con successo in ' + plotName}\033[0m")
    
    nameCSV = "DATASET_" + SYMBOL + ".csv" 
    
    forex_data.to_csv(nameCSV, index=False)
    print(f"\033[92m{'Dataset generato con successo'}\033[0m")
else:
    print(f"\033[91m{'Impossibile generare il dataset a causa di un errore nel recupero dei dati.'}\033[0m")
