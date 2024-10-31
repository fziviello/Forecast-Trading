import yfinance as yf
import finplot as fplt
from datetime import datetime, timedelta

# Configurazioni
GENERATE_PLOT = False
SYMBOL = 'AUDJPY=X'
CSVNAME = 'forex_data.csv'

today = datetime.now()
startDate = today - timedelta(days=730)

def getForexData(symbol):
    try:
        sym = yf.Ticker(symbol)
        data = sym.history(start=startDate, end=today, interval='1h')
        if data.empty:
            raise ValueError(f"Nessun dato restituito per {symbol}")
        
        data.reset_index(inplace=True)
        return data[['Datetime', 'Open', 'High', 'Low', 'Close']]
    except Exception as e:
        print(f"Errore durante il recupero dei dati: {e}")
        return None

forex_data = getForexData(SYMBOL)

if forex_data is not None:
    if GENERATE_PLOT:
        fplt.candlestick_ochl(forex_data[['Datetime', 'Open', 'Close', 'High', 'Low']])
        fplt.show()
    
    forex_data.to_csv(CSVNAME, index=False)
    print(f"\033[92m{'Dataset generato con successo'}\033[0m")
else:
    print(f"\033[91m{'Impossibile generare il dataset a causa di un errore nel recupero dei dati.'}\033[0m")