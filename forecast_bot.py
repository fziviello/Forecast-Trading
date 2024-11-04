import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import logging

MODEL_PATH = 'lstm_trading_model.h5'
SCALER_PATH = 'scaler.pkl'
DATASET_PATH = 'forex_data.csv'
FORECAST_RESULTS_PATH = 'forecast_trading.csv'
LOG_FILE_PATH = 'forecast_trading.log'
PLOT_FILE_PATH = 'forecast_trading.png'

MARGIN_PROFIT = 0.005
LEVERAGE = 0.01
UNIT = 100
EXCHANGE_RATE = 1.0
N_PREDICTIONS = 5

REPEAT_TRAINING = False
GENERATE_PLOT = False
OVERWRITE_FORECAST_CSV = False

logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    df = pd.read_csv(DATASET_PATH, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    
    return df

def create_sequences(X, y, time_steps=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def run_trading_model():
    df = load_and_preprocess_data()
    X = df[['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'Volatility']]
    y = df['Target']

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = MinMaxScaler()
        scaler.fit(X)
        joblib.dump(scaler, SCALER_PATH)
    X_scaled = scaler.transform(X)

    time_steps = 30
    X_seq, y_seq = create_sequences(X_scaled, y.values, time_steps)
    X_train, X_test = X_seq[:-N_PREDICTIONS], X_seq[-N_PREDICTIONS:]
    y_train, y_test = y_seq[:-N_PREDICTIONS], y_seq[-N_PREDICTIONS:]

    if os.path.exists(MODEL_PATH) and not REPEAT_TRAINING:
        model = load_model(MODEL_PATH)
    else:
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        model.save(MODEL_PATH)

    predictions = (model.predict(X_test) > 0.5).astype(int)

    results = []
    for i, pred in enumerate(predictions.flatten()):
        entry_price = round(df['Close'].iloc[-N_PREDICTIONS + i], 3)
        order_type = "Buy" if pred == 1 else "Sell"
        
        stop_loss = round(entry_price * (0.98 if order_type == "Buy" else 1.02), 3)
        take_profit = round(entry_price * (1 + MARGIN_PROFIT if order_type == "Buy" else 1 - MARGIN_PROFIT), 3)
        
        hypothetical_profit = round((take_profit - entry_price) * UNIT * LEVERAGE * EXCHANGE_RATE, 2)
        hypothetical_loss = round((entry_price - stop_loss) * UNIT * LEVERAGE * EXCHANGE_RATE, 2)
        
        results.append({
            'Data Previsione': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Tipo': order_type,
            'Prezzo': f"{entry_price:.3f}",
            'Stop Loss': f"{stop_loss:.3f}",
            'Take Profit': f"{take_profit:.3f}",
            'Guadagno': f"{hypothetical_profit:.2f}€",
            'Perdita': f"{hypothetical_loss:.2f}€"
        })

    print("\nPrevisioni Generate:\n")
    for result in results:
        logging.info(f"Data: {result['Data Previsione']}, Tipo: {result['Tipo']}, Prezzo: {result['Prezzo']}, Stop Loss: {result['Stop Loss']}, Take Profit: {result['Take Profit']}, Guadagno: {result['Guadagno']}, Perdita: {result['Perdita']}")
        
        type_colored = f"\033[94m{result['Tipo']}\033[0m" if result['Tipo'] == "Buy" else f"\033[91m{result['Tipo']}\033[0m"
        entry_price_colored = f"\033[96m{result['Prezzo']}\033[0m"
        stop_loss_colored = f"\033[93m{result['Stop Loss']}\033[0m"
        take_profit_colored = f"\033[95m{result['Take Profit']}\033[0m"
        guadagno_colored = f"\033[92m{result['Guadagno']}\033[0m" if float(result['Guadagno'][:-1]) > 0 else f"\033[91m{result['Guadagno']}\033[0m"
        perdita_colored = f"\033[91m{result['Perdita']}\033[0m"

        print(
            f"Tipo: {type_colored}, Prezzo: {entry_price_colored}, Stop Loss: {stop_loss_colored}, "
            f"Take Profit: {take_profit_colored}, Guadagno: {guadagno_colored}, "
            f"Perdita: {perdita_colored}"
        )

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
        plt.figure(figsize=(14, 7))
        plt.plot(df.index[-N_PREDICTIONS:], df['Close'].iloc[-N_PREDICTIONS:], color='red', label='Prezzo di Chiusura')
        
        buy_signals = [i for i, x in enumerate(predictions) if x == 1]
        sell_signals = [i for i, x in enumerate(predictions) if x == 0]

        if buy_signals:
            plt.scatter(df.index[-N_PREDICTIONS:][buy_signals], df['Close'].iloc[-N_PREDICTIONS:].iloc[buy_signals], color='blue', label='Segnali Buy', marker='^')
        if sell_signals:
            plt.scatter(df.index[-N_PREDICTIONS:][sell_signals], df['Close'].iloc[-N_PREDICTIONS:].iloc[sell_signals], color='magenta', label='Segnali Sell', marker='v')

        plt.title('Forecast')
        plt.xlabel('Data')
        plt.ylabel('Prezzo')
        plt.legend()
        plt.savefig(PLOT_FILE_PATH, format='png')
        plt.show()

if __name__ == "__main__":
    run_trading_model()
