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
        
def plot_model_performance(accuracy_list, plot_file_path, show_plot=False):    
    units = [config['units'] for config in accuracy_list]
    dropout = [config['dropout'] for config in accuracy_list]
    epochs = [config['epochs'] for config in accuracy_list]
    batch_size = [config['batch_size'] for config in accuracy_list]
    learning_rate = [config['learning_rate'] for config in accuracy_list]
    optimizer = [config['optimizer'] for config in accuracy_list]
    accuracies = [config['accuracy'] for config in accuracy_list]

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(range(len(accuracies)), accuracies, c=accuracies, cmap='viridis', edgecolors='k', s=100)

    ax.set_title("Performance dei Modelli durante la Ricerca a Griglia", fontsize=14)
    ax.set_xlabel("Configurazione dei Parametri", fontsize=12)
    ax.set_ylabel("Accuratezza", fontsize=12)
    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels([
        f"Units={u}\nDropout={d}\nEpochs={e}\nBatch={b}\nLR={lr}\nOpt={opt}"
        for u, d, e, b, lr, opt in zip(units, dropout, epochs, batch_size, learning_rate, optimizer)
    ], rotation=90, fontsize=8)

    plt.colorbar(scatter, ax=ax, label="Accuratezza")
    plt.tight_layout()
    plt.savefig(plot_file_path)
    
    if show_plot is True:
        plt.show()