import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from config import N_PREDICTIONS

def plot_statistics(statistics, plot_file_path, show_plot=False):
    symbols = [stat['symbol'] for stat in statistics]
    success_rates = [stat['success_rate'] for stat in statistics]
    failure_rates = [stat['failure_rate'] for stat in statistics]

    x = range(len(symbols))

    plt.figure(figsize=(10, 6))
    plt.bar(x, success_rates, color='green', alpha=0.7, label='Success Rate (%)')
    plt.bar(x, failure_rates, bottom=success_rates, color='red', alpha=0.7, label='Failure Rate (%)')

    plt.xticks(x, symbols, rotation=45, ha='right')
    plt.xlabel('Symbols')
    plt.ylabel('Percentage')
    plt.title('Statistiche di validazione per tutti i simboli')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file_path)
    print(f"\n\033[92mGrafico salvato in: {plot_file_path}\033[0m")
    if show_plot:
        plt.show()
        
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