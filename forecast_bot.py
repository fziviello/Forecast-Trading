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

MODEL_PATH = 'lstm_trading_model.h5'
SCALER_PATH = 'scaler.pkl'
FORECAST_RESULTS_PATH = 'forecast_trading.csv'
DATASET_PATH = 'forex_data.csv'

REPEAT_TRAINING = False

GENERATE_PLOT = False
OVERWRITE_FORECAST_CSV = True

MARGIN_PROFIT = 0.005
LEVERAGE = 0.01
UNIT = 100
EXCHANGE_RATE = 1.0
N_PREDICTIONS = 5

def run_trading_model():
    def create_sequences(X, y, time_steps=60):
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])
        return np.array(X_seq), np.array(y_seq)

    df = pd.read_csv(DATASET_PATH, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'Volatility']]
    y = df['Target']

    # Verifica se esiste uno scaler salvato
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)

    # Creazione delle sequenze per LSTM
    time_steps = 60
    X_seq, y_seq = create_sequences(X_scaled, y.values, time_steps)

    # Divisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Verifica se esiste un modello addestrato salvato
    if os.path.exists(MODEL_PATH) and not REPEAT_TRAINING:
        model = load_model(MODEL_PATH)
        print("Modello caricato con successo.")
    else:
        # Creazione del modello LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compilazione del modello
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Aggiungiamo un early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Addestramento del modello
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Salvataggio del modello addestrato
        model.save(MODEL_PATH)
        print("Modello addestrato e salvato con successo.")

    # Valutazione del modello
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Accuratezza del test: {test_accuracy * 100:.2f}%")

    # Predizioni per il backtesting
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    df_test = df.iloc[-len(y_test):].copy()
    df_test['Segnale'] = predictions

    target_profit_margin = MARGIN_PROFIT
    leverage = LEVERAGE
    unit = UNIT
    exchange_rate = EXCHANGE_RATE
    num_predictions = N_PREDICTIONS
    
    results = []

    for i in range(len(df_test) - 1):
        if len(results) >= num_predictions:
            break
        
        entry_price = round(df_test['Close'].iloc[i], 3)
        if df_test['Segnale'].iloc[i] == 1:
            order_type = "Buy"
            stop_loss = round(entry_price * 0.98, 3)
            take_profit = round(entry_price * (1 + target_profit_margin), 3)
            
            hypothetical_profit = round((take_profit - entry_price) * unit * leverage * exchange_rate, 2)
            hypothetical_loss = round((entry_price - stop_loss) * unit * leverage * exchange_rate, 2)
            
            results.append({
                'Data Previsione': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Tipo': order_type,
                'Prezzo': f"{entry_price:.3f}",
                'Stop Loss': f"{stop_loss:.3f}",
                'Take Profit': f"{take_profit:.3f}",
                'Guadagno': f"{hypothetical_profit:.2f}€",
                'Perdita': f"{hypothetical_loss:.2f}€"
            })

        elif df_test['Segnale'].iloc[i] == 0:
            order_type = "Sell"
            stop_loss = round(entry_price * 1.02, 3)
            take_profit = round(entry_price * (1 - target_profit_margin), 3)
            
            hypothetical_profit = round((entry_price - take_profit) * unit * leverage * exchange_rate, 2)
            hypothetical_loss = round((stop_loss - entry_price) * unit * leverage * exchange_rate, 2)
            
            results.append({
                'Data Previsione': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Tipo': order_type,
                'Prezzo': f"{entry_price:.3f}",
                'Stop Loss': f"{stop_loss:.3f}",
                'Take Profit': f"{take_profit:.3f}",
                'Guadagno': f"{hypothetical_profit:.2f}€",
                'Perdita': f"{hypothetical_loss:.2f}€"
            })

    results_df = pd.DataFrame(results)
    results_df = results_df[['Data Previsione', 'Tipo', 'Prezzo', 'Stop Loss', 'Take Profit', 'Guadagno', 'Perdita']]
    
    if OVERWRITE_FORECAST_CSV:
        results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)
    else:
        if os.path.isfile(FORECAST_RESULTS_PATH):
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='a', index=False, header=False)
        else:
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)

    # Stampa finale delle previsioni
    print("\nPrevisioni Generate:\n")
    for index, row in results_df.iterrows():
        type_colored = f"\033[94m{row['Tipo']}\033[0m" if row['Tipo'] == "Buy" else f"\033[91m{row['Tipo']}\033[0m"
        entry_price_colored = f"\033[96m{row['Prezzo']}\033[0m"
        stop_loss_colored = f"\033[93m{row['Stop Loss']}\033[0m"
        take_profit_colored = f"\033[95m{row['Take Profit']}\033[0m"
        guadagno_colored = f"\033[92m{row['Guadagno']}\033[0m" if float(row['Guadagno'][:-1]) > 0 else f"\033[91m{row['Guadagno']}\033[0m"
        perdita_colored = f"\033[91m{row['Perdita']}\033[0m"

        print(
            f"Tipo: {type_colored}, Prezzo: {entry_price_colored}, Stop Loss: {stop_loss_colored}, "
            f"Take Profit: {take_profit_colored}, Guadagno: {guadagno_colored}, "
            f"Perdita: {perdita_colored}"
        )

    # Analisi delle performance
    total_trades = len(results)
    winning_trades = len([r for r in results if r['Tipo'] == "Buy" and float(r['Guadagno'][:-1]) > 0]) + \
                    len([r for r in results if r['Tipo'] == "Sell" and float(r['Guadagno'][:-1]) < 0])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    average_profit = np.mean([float(r['Guadagno'][:-1]) for r in results])
    average_loss = np.mean([float(r['Perdita'][:-1]) for r in results])
    average_profit_colored = f"\033[92m{average_profit:.2f}€\033[0m" if average_profit > 0 else f"\033[91m{average_profit:.2f}€\033[0m"
    average_loss_colored = f"\033[91m{average_loss:.2f}€\033[0m" if average_loss >= 0 else f"\033[92m{average_loss:.2f}€\033[0m"

    print(f"\nTotale Operazioni: {total_trades}")
    print(f"Operazioni Vincenti: {winning_trades} ({win_rate:.2f}%)")
    print(f"Operazioni Perdenti: {losing_trades}")
    print(f"Guadagno Medio: {average_profit_colored}")
    print(f"Perdita Media: {average_loss_colored}\n")

    if GENERATE_PLOT:
        plt.figure(figsize=(14, 7))
        plt.plot(df_test.index, df_test['Close'], label='Prezzo di Chiusura', color='blue')
        plt.scatter(df_test.index, df_test['Segnale'] * df_test['Close'], label='Segnali di Trading', color='red', marker='^', alpha=1, s=100)
        plt.title('Previsioni di Trading con LSTM')
        plt.xlabel('Data')
        plt.ylabel('Prezzo di Chiusura')
        plt.legend()
        plt.grid()
        plt.show()

run_trading_model()