import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

GENERATE_PLOT = False
SYMBOL = 'AUDJPY=X'
CSVNAME = 'forex_data.csv'
PLOTNAME = 'forex_chart.png'
INTERVAL = '30m'

today = datetime.now()
if INTERVAL == "30m" :
   ndays = 60
else:
   ndays = 730

startDate = today - timedelta(days=ndays)

def getForexData(symbol):
    try:
        sym = yf.Ticker(symbol)
        data = sym.history(start=startDate, end=today, interval=INTERVAL)
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
            savefig=PLOTNAME
        )
        print(f"\033[92m{'Grafico salvato con successo in ' + PLOTNAME}\033[0m")
    
    forex_data.to_csv(CSVNAME, index=False)
    print(f"\033[92m{'Dataset generato con successo'}\033[0m")
else:
    print(f"\033[91m{'Impossibile generare il dataset a causa di un errore nel recupero dei dati.'}\033[0m")
