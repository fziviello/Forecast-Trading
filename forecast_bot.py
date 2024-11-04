import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
FORECAST_RESULTS_PATH = 'forecast_trading.csv'
DATASET_PATH = 'forex_data.csv'
LOG_FILE_PATH = 'forecast_trading.log'
PLOT_FILE_PATH = 'forecast_trading.png'

REPEAT_TRAINING = False
GENERATE_PLOT = False
OVERWRITE_FORECAST_CSV = False

MARGIN_PROFIT = 0.005
N_PREDICTIONS = 10

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

def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    model.save(MODEL_PATH)
    logging.info("Modello addestrato e salvato con successo.")

def run_trading_model():
    df = load_and_preprocess_data()

    X = df[['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'Volatility']]
    y = df['Target']

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)

    time_steps = 60
    X_seq, y_seq = create_sequences(X_scaled, y.values, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    if os.path.exists(MODEL_PATH) and not REPEAT_TRAINING:
        model = load_model(MODEL_PATH)
        logging.info("Modello caricato con successo.")
    else:
        train_model(X_train, y_train)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    logging.info(f"Accuratezza del test: {test_accuracy * 100:.2f}%")

    predictions = (model.predict(X_test) > 0.5).astype("int32")
    df_test = df.iloc[-len(y_test):].copy()
    df_test['Segnale'] = predictions

    results = []

    for i in range(len(df_test) - 1):
        if len(results) >= N_PREDICTIONS:
            break
        
        entry_price = round(df_test['Close'].iloc[i], 3)
        if df_test['Segnale'].iloc[i] == 1:
            order_type = "Buy"
            stop_loss = round(entry_price * 0.98, 3)
            take_profit = round(entry_price * (1 + MARGIN_PROFIT), 3)
            
            results.append({
                'Data Previsione': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Tipo': order_type,
                'Prezzo': f"{entry_price:.3f}",
                'Stop Loss': f"{stop_loss:.3f}",
                'Take Profit': f"{take_profit:.3f}",
            })

        elif df_test['Segnale'].iloc[i] == 0:
            order_type = "Sell"
            stop_loss = round(entry_price * 1.02, 3)
            take_profit = round(entry_price * (1 - MARGIN_PROFIT), 3)

            results.append({
                'Data Previsione': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Tipo': order_type,
                'Prezzo': f"{entry_price:.3f}",
                'Stop Loss': f"{stop_loss:.3f}",
                'Take Profit': f"{take_profit:.3f}",
            })

    results_df = pd.DataFrame(results)
    
    if OVERWRITE_FORECAST_CSV:
        results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)
    else:
        if os.path.isfile(FORECAST_RESULTS_PATH):
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='a', index=False, header=False)
        else:
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)

    logging.info("\nPrevisioni Generate:\n")
    for index, row in results_df.iterrows():
        logging.info(f"Tipo: {row['Tipo']}, Prezzo: {row['Prezzo']}, Stop Loss: {row['Stop Loss']}, Take Profit: {row['Take Profit']}")

    if GENERATE_PLOT:
        plt.figure(figsize=(14, 7))
        plt.plot(df_test.index, df_test['Close'], color='blue', label='Prezzo di Chiusura')
        buy_signals = df_test[df_test['Segnale'] == 1]
        if not buy_signals.empty:
            plt.scatter(buy_signals.index, buy_signals['Close'], color='green', label='Segnali Buy', marker='^')

        sell_signals = df_test[df_test['Segnale'] == 0]
        if not sell_signals.empty:
            plt.scatter(sell_signals.index, sell_signals['Close'], color='red', label='Segnali Sell', marker='v')

        plt.title('Previsioni di Trading')
        plt.xlabel('Data')
        plt.ylabel('Prezzo')
        plt.legend()
        
        plt.savefig(PLOT_FILE_PATH, format='png')
        
        plt.show()

if __name__ == "__main__":
    run_trading_model()