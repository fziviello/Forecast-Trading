import pandas as pd
import mplfinance as mpf
from config import N_PREDICTIONS

def plot_forex_candlestick(df, predictions, plot_file_path, show_plot=False):
    df_plot = df[-N_PREDICTIONS:].copy()
    
    df_plot.index = pd.to_datetime(df_plot.index)

    df_plot['Date'] = df_plot.index
    df_plot.index.name = 'Date'
    df_plot = df_plot[['Open', 'High', 'Low', 'Close']]

    buy_signals = df_plot.iloc[[i for i, x in enumerate(predictions) if x == 1]]
    sell_signals = df_plot.iloc[[i for i, x in enumerate(predictions) if x == 0]]

    add_plot = []
    if not buy_signals.empty:
        add_plot.append(mpf.make_addplot(buy_signals['Close'], type='scatter', marker='^', color='blue', markersize=100, label='Segnali Buy'))
    if not sell_signals.empty:
        add_plot.append(mpf.make_addplot(sell_signals['Close'], type='scatter', marker='v', color='magenta', markersize=100, label='Segnali Sell'))

    add_plot.append(mpf.make_addplot(df_plot['Close'].iloc[-N_PREDICTIONS:], color='red', label='Prezzo di Chiusura'))

    mpf.plot(df_plot, type='candle', style='charles', addplot=add_plot, title='Forecast',
             ylabel='Prezzo', savefig=plot_file_path, volume=False, show_nontrading=False)

    if show_plot:
        mpf.plot(df_plot, type='candle', style='charles', addplot=add_plot, title='Forecast',
             ylabel='Prezzo', volume=False, show_nontrading=False)
        mpf.show()