import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import sys
from datetime import datetime,timedelta
import pytz
import mplfinance as mpf
import logging
import argparse

from folder_config import setup_folders, MODELS_FOLDER, DATA_FOLDER, RESULTS_FOLDER, PLOTS_FOLDER, LOGS_FOLDER, LOG_FORECAST_FILE_PATH
from config import MARGIN_PROFIT, LEVERAGE, UNIT, EXCHANGE_RATE, FAVORITE_RATE, N_PREDICTIONS, VALIDATION_THRESHOLD, INTERVAL_MINUTES

GENERATE_PLOT = False
SHOW_PLOT = False
OVERWRITE_FORECAST_CSV = False
REPEAT_TRAINING = False

setup_folders()
    
logging.basicConfig(filename=os.path.join(LOGS_FOLDER, LOG_FORECAST_FILE_PATH), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculator_profit(predicted_take_profit, predicted_entry_price):
    global UNIT, LEVERAGE, EXCHANGE_RATE
    return round(abs((predicted_take_profit - predicted_entry_price) * UNIT * LEVERAGE * EXCHANGE_RATE), 2)
            
def calculator_loss(predicted_stop_loss, predicted_entry_price):
    global UNIT, LEVERAGE, EXCHANGE_RATE
    return round((predicted_entry_price - predicted_stop_loss) * UNIT * LEVERAGE * EXCHANGE_RATE, 2)

def exchange_currency(base, target):
    ticker = f"{base}{target}=X"
    try:
        data = yf.Ticker(ticker)
        exchange_rate = (data.history(period="1d")['Close'].iloc[-1])
        exchange_rate = round(exchange_rate, 2)
        print(f"\033[94m\nIl tasso di cambio da {base} a {target} è: \033[92m{exchange_rate}€\033[0m\n")
        logging.info(f"Il tasso di cambio da {base} a {target} è: {exchange_rate}")
        return exchange_rate
    except Exception as e:
        print(f"\033[91m'Errore nel recuperare il tasso di cambio, verrà utilizzato il suo valore di default'\033[0m")
        logging.error(f"Errore nel recuperare il tasso di cambio: {e}")
        return None

def load_and_preprocess_data():
    global DATASET_PATH
    df = pd.read_csv(DATASET_PATH, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Upper Bollinger'] = df['MA20'] + 2 * df['Volatility']
    df['Lower Bollinger'] = df['MA20'] - 2 * df['Volatility']
    
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    
    return df

def create_sequences(X, y, time_steps=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def validate_predictions():
    global REPEAT_TRAINING
    if not os.path.exists(FORECAST_RESULTS_PATH):
        logging.info("Nessuna previsione precedente da validare.")
        return

    prev_predictions = pd.read_csv(FORECAST_RESULTS_PATH)
    df = pd.read_csv(DATASET_PATH, parse_dates=['Datetime'])
    validation_results = []
    unsuccessful_count = 0

    for i, row in prev_predictions.iterrows():
        pred_datetime = pd.to_datetime(row['Data Previsione'])
        actual_data = df.loc[df['Datetime'] >= pred_datetime]

        if not actual_data.empty:
            actual_close = actual_data.iloc[0]['Close']
            predicted_entry_price = float(row['Prezzo'])
            predicted_take_profit = float(row['Take Profit'])
            predicted_stop_loss = float(row['Stop Loss'])

            if actual_close >= predicted_take_profit:
                result = "Successo - Take Profit raggiunto"
            elif actual_close <= predicted_stop_loss:
                result = "Fallimento - Stop Loss raggiunto"
                unsuccessful_count += 1
            else:
                result = "Previsione non avvenuta"
            
            validation_results.append({
                'Data Previsione': pred_datetime,
                'Tipo': row['Tipo'],
                'Risultato': result,
                'Prezzo': predicted_entry_price,
                'Close Attuale': actual_close,
                'Take Profit': predicted_take_profit,
                'Stop Loss': predicted_stop_loss,
                'Guadagno': f"{calculator_profit(predicted_take_profit, predicted_entry_price):.2f}€",
                'Perdita': f"{calculator_loss(predicted_stop_loss, predicted_entry_price):.2f}€"
            })

    if validation_results:
        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv(VALIDATION_RESULTS_PATH, index=False)
        logging.info(f"Risultati di validazione salvati in {VALIDATION_RESULTS_PATH}.")

    total_predictions = len(prev_predictions)
    if total_predictions > 0:
        failure_rate = unsuccessful_count / total_predictions
        if failure_rate > 0:
            print(f"\033[91mTasso di previsioni non riuscite: {failure_rate:.2f}\033[0m")
        logging.info(f"Tasso di previsioni non riuscite: {failure_rate:.2f}")
        REPEAT_TRAINING = failure_rate > VALIDATION_THRESHOLD

def plot_forex_candlestick(df, predictions):
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
             ylabel='Prezzo', savefig=PLOT_FILE_PATH, volume=False, show_nontrading=False)

    if SHOW_PLOT:
        mpf.plot(df_plot, type='candle', style='charles', addplot=add_plot, title='Forecast',
             ylabel='Prezzo', volume=False, show_nontrading=False)
        mpf.show()

def run_trading_model():
    global MODEL_PATH, SCALER_PATH, FORECAST_RESULTS_PATH, VALIDATION_RESULTS_PATH, LOG_FILE_PATH, PLOT_FILE_PATH, REPEAT_TRAINING, INTERVAL_MINUTES
    validate_predictions()
    df = load_and_preprocess_data()
    X = df[['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'Volatility', 'RSI', 'MACD', 'Upper Bollinger', 'Lower Bollinger']].values
    y = df['Target']

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = MinMaxScaler()
        scaler.fit(X)
        joblib.dump(scaler, SCALER_PATH)
    X_scaled = scaler.transform(X)

    time_steps = 20
    X_seq, y_seq = create_sequences(X_scaled, y.values, time_steps)
    X_train, X_test = X_seq[:-N_PREDICTIONS], X_seq[-N_PREDICTIONS:]
    y_train, y_test = y_seq[:-N_PREDICTIONS], y_seq[-N_PREDICTIONS:]

    if os.path.exists(MODEL_PATH) and not REPEAT_TRAINING:
        model = load_model(MODEL_PATH)
    else:
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=75, return_sequences=True),
            Dropout(0.2),
            LSTM(units=75, return_sequences=False),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=75, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        model.save(MODEL_PATH)

    predictions = (model.predict(X_test) > 0.5).astype(int)

    results = []
    for i, pred in enumerate(predictions.flatten()):
        entry_price = round(df['Close'].iloc[-N_PREDICTIONS + i], 3)
        order_type = "Buy" if pred == 1 else "Sell"
        order_class = "Limit" if pred == 1 else "Stop"

        stop_loss = round(entry_price * (0.98 if order_type == "Buy" else 1.02), 3)
        take_profit = round(entry_price * (1 + MARGIN_PROFIT if order_type == "Buy" else 1 - MARGIN_PROFIT), 3)
        
        hypothetical_profit = calculator_profit(take_profit, entry_price)
        hypothetical_loss = calculator_loss(stop_loss, entry_price)

        data_obj = datetime.now()
        data_utc = data_obj.replace(tzinfo=pytz.UTC)
        data_utc = data_utc - timedelta(minutes=INTERVAL_MINUTES)  # Sottraggo i minuti definiti nel dataset per essere allineato
        data_formatted = data_utc.strftime("%Y-%m-%d %H:%M:%S%z")

        results.append({
            'Data Previsione': data_formatted,
            'Tipo': f"{order_type} {order_class}",
            'Prezzo': f"{entry_price:.5f}",
            'Stop Loss': f"{stop_loss:.5f}",
            'Take Profit': f"{take_profit:.5f}",
            'Guadagno': f"{hypothetical_profit:.2f}€",
            'Perdita': f"{hypothetical_loss:.2f}€"
        })

    results = sorted(results, key=lambda x: float(x['Prezzo']))

    print("\nPrevisioni Generate:\n")
    row_index = 1
    for result in results:
        logging.info(f"Data: {result['Data Previsione']}, Tipo: {result['Tipo']}, Prezzo: {result['Prezzo']}, Stop Loss: {result['Stop Loss']}, Take Profit: {result['Take Profit']}, Guadagno: {result['Guadagno']}, Perdita: {result['Perdita']}")

        type_colored = f"\033[94m{result['Tipo']}\033[0m" if result['Tipo'] == "Buy" or result['Tipo'] == "Buy Limit" or result['Tipo'] == "Buy Stop" else f"\033[91m{result['Tipo']}\033[0m"
        entry_price_colored = f"\033[96m{result['Prezzo']}\033[0m"
        stop_loss_colored = f"\033[93m{result['Stop Loss']}\033[0m"
        take_profit_colored = f"\033[95m{result['Take Profit']}\033[0m"
        guadagno_colored = f"\033[92m{result['Guadagno']}\033[0m" if float(result['Guadagno'][:-1]) > 0 else f"\033[91m{result['Guadagno']}\033[0m"
        perdita_colored = f"\033[91m{result['Perdita']}\033[0m"

        print(
            f"{row_index:>2})  {type_colored:<8} Prezzo: {entry_price_colored:<8} Stop Loss: {stop_loss_colored:<8} "
            f"Take Profit: {take_profit_colored:<8} Guadagno: {guadagno_colored:<10} Perdita: {perdita_colored:<10}"
        )

        row_index += 1

    print("\n")
    results_df = pd.DataFrame(results)
    results_df = results_df[['Data Previsione', 'Tipo', 'Prezzo', 'Stop Loss', 'Take Profit', 'Guadagno', 'Perdita']]

    if OVERWRITE_FORECAST_CSV:
        results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)
    else:
        if os.path.isfile(FORECAST_RESULTS_PATH):
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='a', index=False, header=False)
        else:
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)

    if GENERATE_PLOT:
        plot_forex_candlestick(df, predictions)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Inserire il symbol da esaminare")
    parser.add_argument("--plot", type=str, required=False, help="Generare il grafico")
    parser.add_argument("--interval", type=str, required=False, help="Inserire l'intervallo temporale in minuti")
    parser.add_argument("--favoriteRate", type=str, required=False, help="Inserire il rate di conversione")
    parser.add_argument("--symbol", type=str, required=True, help="Inserire il symbol per avviare il bot")
    args = parser.parse_args()
    SYMBOL = (args.symbol).upper()
    
    if args.interval is not None :
        FAVORITE_RATE = args.favoriteRate
    
    if args.interval is not None :
        INTERVAL_MINUTES = args.interval
        
    if args.plot is not None:
        GENERATE_PLOT = args.plot
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    MODEL_PATH = os.path.join(MODELS_FOLDER, f"lstm_trading_model_{SYMBOL}.h5")
    SCALER_PATH = os.path.join(MODELS_FOLDER, f"scaler_{SYMBOL}.pkl")
    FORECAST_RESULTS_PATH = os.path.join(RESULTS_FOLDER, f"forecast_trading_{SYMBOL}.csv")
    VALIDATION_RESULTS_PATH = os.path.join(RESULTS_FOLDER, f"forecast_validation_{SYMBOL}.csv")
    PLOT_FILE_PATH = os.path.join(PLOTS_FOLDER, f"forecast_trading_{SYMBOL}_{timestamp}.png")
    DATASET_PATH = os.path.join(DATA_FOLDER, f"DATASET_{SYMBOL}.csv")
    DATA_MODEL_RATE = SYMBOL[:3]
    
    rate_exchange = exchange_currency(DATA_MODEL_RATE, FAVORITE_RATE)
    
    if rate_exchange:
        EXCHANGE_RATE = rate_exchange

    run_trading_model()
    
