import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop # type: ignore
import talib
import joblib
import os
from datetime import datetime, timedelta
import pytz
import logging
import argparse
import itertools
import time
from utilities.utility import str_to_bool
from utilities.telegram_sender import TelegramSender
from utilities.forwarder_MT5 import TradingAPIClient
from utilities.folder_config import setup_folders, MODELS_FOLDER, DATA_FOLDER, RESULTS_FOLDER, PLOTS_FOLDER, LOGS_FOLDER, LOG_FORECAST_FILE_PATH
from config import IP_SERVER_TRADING, PORT_SERVER_TRADING, BOT_TOKEN, CHANNEL_TELEGRAM, PARAM_GRID, FAVORITE_RATE, N_PREDICTIONS, VALIDATION_THRESHOLD, INTERVAL_MINUTES, FORECAST_VALIDITY_MINUTES
from utilities.calculator import get_profit, get_loss, get_dynamic_margin
from utilities.plots import plot_forex_candlestick, plot_model_performance

GENERATE_PLOT = False
SEND_TELEGRAM = False
SEND_SERVER_SIGNAL = False
SHOW_PLOT = False
OVERWRITE_FORECAST_CSV = False
REPEAT_TRAINING = False

SYMBOL = None
BEST_ACCURACY = None
BEST_PARAMS = None
BEST_MODEL = None
ACCURACY_LIST = []

setup_folders()

logging.basicConfig(
    filename=os.path.join(LOGS_FOLDER, LOG_FORECAST_FILE_PATH),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

urlServer = "http://" + IP_SERVER_TRADING + ":" + PORT_SERVER_TRADING
tradingClient = TradingAPIClient(urlServer)

def is_forecast_still_valid(details_notify_list, time_life_minutes=60):
    global FORECAST_RESULTS_PATH
    try:
        df = pd.read_csv(FORECAST_RESULTS_PATH)
        unique_details = list(dict.fromkeys(details_notify_list))
        now = datetime.now(pytz.utc)
        for notify in unique_details:
            if notify in df.to_string(index=False):
                if 'Data Previsione' in df.columns:
                    data_previsione = pd.to_datetime(
                        df.loc[df.apply(lambda row: notify in row.to_string(), axis=1), 'Data Previsione']
                    )
                    if not data_previsione.empty:
                        data_previsione = data_previsione.iloc[0].tz_localize(None)
                        now_naive = now.replace(tzinfo=None)
                        if now_naive - data_previsione > timedelta(minutes=time_life_minutes):
                            continue
                return False
        return True
    except FileNotFoundError:
        logging.warning(f"Il File {FORECAST_RESULTS_PATH} non esiste o potrebbe non essere ancora stato generato")
        print(f"\033[93mIl File {FORECAST_RESULTS_PATH} non esiste o potrebbe non essere ancora stato generato\n\033[0m")
        return True
    except Exception as e:
        logging.error(f"Errore durante la verifica della notifica: {e}")
        print(f"\033[91m'Errore durante la verifica della notifica: {e}'\033[0m")
        return False

def sendNotify(msg):
    if SEND_TELEGRAM is True:
        telegramSender = TelegramSender(BOT_TOKEN)
        telegramSender.sendMsg(msg, CHANNEL_TELEGRAM)
        
def exchange_currency(base, target):
    ticker = f"{base}{target}=X"
    try:
        data = yf.Ticker(ticker)
        exchange_rate = (data.history(period="1d")['Close'].iloc[-1])
        exchange_rate = round(exchange_rate, 5)
        print(f"\033[94m\nIl tasso di cambio da {base} a {target} è: \033[92m{exchange_rate}€\033[0m\n")
        logging.info(f"Il tasso di cambio da {base} a {target} è: {exchange_rate}")
        return exchange_rate
    except Exception as e:
        print(f"\033[91m'Errore nel recuperare il tasso di cambio, verrà utilizzato il suo valore di default'\033[0m")
        logging.error(f"Errore nel recuperare il tasso di cambio: {e}")
        return None

def create_model(X_train, y_train, X_test, y_test, units, dropout, epochs, batch_size, learning_rate, optimizer_name):
    global SYMBOL, BEST_ACCURACY, BEST_PARAMS, BEST_MODEL, ACCURACY_LIST    
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
        
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout),
        Dense(units=1, activation='sigmoid')
    ])
        
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Accuracy: {accuracy:.3f}%\n")
    logging.info(f"Accuracy: {accuracy:.3f}%")
    
    ACCURACY_LIST.append({'units': units, 'dropout': dropout, 'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate, 'optimizer': optimizer_name, 'accuracy': accuracy})

    if accuracy > BEST_ACCURACY:
        BEST_ACCURACY = accuracy
        BEST_PARAMS = {'units': units, 'dropout': dropout, 'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate, 'optimizer': optimizer_name}
        BEST_MODEL = model
        print(f"\033\n[92mConfigurazione migliore trovata per {SYMBOL}: {BEST_PARAMS} con accuracy={BEST_ACCURACY:.3f}%\033[0m")
        logging.info(f"Configurazione migliore trovata per {SYMBOL}: {BEST_PARAMS} con accuracy={BEST_ACCURACY:.3f}%")
    
def grid_search_optimization(X_train, y_train, X_test, y_test):
    global SYMBOL, BEST_ACCURACY, BEST_PARAMS, BEST_MODEL, ACCURACY_LIST, PLOT_MODEL_FILE_PATH, GENERATE_PLOT

    BEST_ACCURACY = 0.0
    BEST_PARAMS = {}
    ACCURACY_LIST = []
    BEST_MODEL = None

    start_time = time.time()
    for units, dropout, epochs, batch_size, learning_rate, optimizer in itertools.product(
        PARAM_GRID['units'], PARAM_GRID['dropout'], PARAM_GRID['epochs'], PARAM_GRID['batch_size'],
        PARAM_GRID['learning_rate'], PARAM_GRID['optimizer']
    ):
        print(f"\nTest Configurazione per {SYMBOL}: units={units}, dropout={dropout}, epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, optimizer={optimizer}")
        logging.info(f"Test Configurazione per {SYMBOL}: units={units}, dropout={dropout}, epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, optimizer={optimizer}")
        create_model(X_train, y_train, X_test, y_test, units, dropout, epochs, batch_size, learning_rate, optimizer)

    if GENERATE_PLOT is True:
        plot_model_performance(ACCURACY_LIST, PLOT_MODEL_FILE_PATH)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\033[93m\nTempo di elaborazione migliore configurazione {SYMBOL}: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n\033[0m")
    logging.info(f"Tempo di elaborazione migliore configurazione {SYMBOL}: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

def load_and_preprocess_data():
    global DATASET_PATH
    df = pd.read_csv(DATASET_PATH, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Media mobile, RSI, Bollinger Bands, MACD
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['MACD'], df['Signal Line'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Upper Bollinger'] = df['MA20'] + 2 * df['Volatility']
    df['Lower Bollinger'] = df['MA20'] - 2 * df['Volatility']

    # Moving Average Envelope
    df['Upper Envelope'] = df['MA20'] * 1.05
    df['Lower Envelope'] = df['MA20'] * 0.95
    
    # Parabolic SAR (Stop and Reverse)
    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    # Stochastic Oscillator
    df['Stochastic Oscillator'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()) * 100
    # Average Directional Index (ADX)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Average True Range (ATR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift())
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift())
    df['True Range'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['True Range'].rolling(window=14).mean()

    # Creazione del target per la previsione
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Rimuove eventuali valori NaN
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
        pred_ticket = row['Ticket']
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
                'Ticket': pred_ticket,
                'Data Previsione': pred_datetime,
                'Tipo': row['Tipo'],
                'Risultato': result,
                'Prezzo': predicted_entry_price,
                'Close Attuale': actual_close,
                'Take Profit': predicted_take_profit,
                'Stop Loss': predicted_stop_loss,
                'Guadagno': f"{get_profit(predicted_take_profit, predicted_entry_price, EXCHANGE_RATE):.2f}€",
                'Perdita': f"{get_loss(predicted_stop_loss, predicted_entry_price, EXCHANGE_RATE):.2f}€"
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

def run_trading_model():
    global SYMBOL, MODEL_PATH, SCALER_PATH, FORECAST_RESULTS_PATH, VALIDATION_RESULTS_PATH, LOG_FILE_PATH, PLOT_FILE_PATH, GENERATE_PLOT, REPEAT_TRAINING, INTERVAL_MINUTES
    validate_predictions()
    df = load_and_preprocess_data()
    X = df[['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'Volatility', 
        'RSI', 'MACD', 'Upper Bollinger', 'Lower Bollinger', 'ATR',
        'SAR', 'Stochastic Oscillator', 'ADX', 'Upper Envelope', 'Lower Envelope']].values
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
        grid_search_optimization(X_train, y_train, X_test, y_test)  
        model = BEST_MODEL
        print(f"\033\n[92mMigliori parametri trovati per {SYMBOL}: {BEST_PARAMS} con accuracy={BEST_ACCURACY:.3f}%\n\033[0m")
        logging.info(f"Migliori parametri trovati per {SYMBOL}: {BEST_PARAMS} con accuracy={BEST_ACCURACY:.3f}%")
        model.save(MODEL_PATH)
       
    predictions = (model.predict(X_test) > 0.5).astype(int)

    results = []
    for i, pred in enumerate(predictions.flatten()):
        entry_price = round(df['Close'].iloc[-N_PREDICTIONS + i], 3)
        order_type = "Buy" if pred == 1 else "Sell"
        order_class = "Limit" if pred == 1 else "Stop"
  
        volatility = df['Volatility'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        dynamic_tp_margin, dynamic_sl_margin = get_dynamic_margin(
            entry_price=entry_price,
            volatility=volatility,
            atr=atr,
            reward_ratio=1,
            recovery_ratio=2,
            atr_weight=0.7
        )

        if order_type == "Sell":
            take_profit = round(entry_price - dynamic_tp_margin, 5)
            stop_loss = round(entry_price + dynamic_sl_margin, 5)
        else:
            take_profit = round(entry_price + dynamic_tp_margin, 5)
            stop_loss = round(entry_price - dynamic_sl_margin, 5)

        logging.debug(f"DEBUG: Entry Price: {entry_price}")
        logging.debug(f"DEBUG: Volatility: {volatility}, ATR: {atr}")
        logging.debug(f"DEBUG: TP Margin: {dynamic_tp_margin}, SL Margin: {dynamic_sl_margin}")
        logging.debug(f"DEBUG: Take Profit: {take_profit}, Stop Loss: {stop_loss}")

        hypothetical_profit = get_profit(take_profit, entry_price, EXCHANGE_RATE)
        hypothetical_loss = get_loss(stop_loss, entry_price, EXCHANGE_RATE)

        # Allineo tutte le date a UTC come il dataset
        data_obj = datetime.now()
        italy_tz = pytz.timezone('Europe/Rome')
        utc_tz = pytz.UTC
        data_utc = data_obj.replace(tzinfo=pytz.UTC)
        data_italy = italy_tz.localize(data_obj)
        data_utc = data_italy.astimezone(utc_tz)
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
    printed_results = set()
    
    print("\nPrevisioni Generate:\n")
    row_index = 1
    details_notify_list = []
    for result in results:
        result_key = (result['Tipo'], result['Prezzo'])
        if result_key in printed_results:
            continue
        printed_results.add(result_key)
        
        if SEND_SERVER_SIGNAL:
            result['Ticket'] = tradingClient.create_order(SYMBOL,result['Tipo'], 0.01, result['Prezzo'], result['Stop Loss'], result['Take Profit'])
        else:
            result['Ticket'] = "-"    
                        
        details = (
            f"Data: {result['Data Previsione']}, Tipo: {result['Tipo']}, Prezzo: {result['Prezzo']}, "
            f"Stop Loss: {result['Stop Loss']}, Take Profit: {result['Take Profit']}, "
            f"Guadagno: {result['Guadagno']}, Perdita: {result['Perdita']}"
        )
        
        circle_emoji = "🔵" if result['Tipo'] in ["Buy", "Buy Limit", "Buy Stop"] else "🔴"
        max_width = max(len(result['Prezzo'].split('.')[0]), 3) + 9
        details_notify = (
            f"{circle_emoji} {result['Tipo']} {SYMBOL}\n"
            f"{'Prezzo:':<12}{result['Prezzo'].rjust(max_width)}\n"
            f"{'Stop Limit:':<12}{result['Stop Loss'].rjust(max_width)}\n"
            f"{'Take Profit:':<12}{result['Take Profit'].rjust(max_width)}\n"
        )
        details_notify_list.append(details_notify)
        logging.info(details)
        type_colored = f"\033[94m{result['Tipo']}\033[0m" if result['Tipo'] in ["Buy", "Buy Limit", "Buy Stop"] else f"\033[91m{result['Tipo']}\033[0m"
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
    results_df = results_df[['Ticket','Data Previsione', 'Tipo', 'Prezzo', 'Stop Loss', 'Take Profit', 'Guadagno', 'Perdita']]
    
    if is_forecast_still_valid(results_df, FORECAST_VALIDITY_MINUTES):
        sendNotify("\n".join(details_notify_list))
    else:
        logging.info(f"La previsione è ancora valida")
        print(f"\033[93mLa previsione è ancora valida\n\033[0m")
        
    if OVERWRITE_FORECAST_CSV:
        results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)
    else:
        if os.path.isfile(FORECAST_RESULTS_PATH):
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='a', index=False, header=False)
        else:
            results_df.to_csv(FORECAST_RESULTS_PATH, mode='w', index=False)
    
    if GENERATE_PLOT is True:
        plot_forex_candlestick(df, predictions, PLOT_FILE_PATH, SHOW_PLOT)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Inserire il symbol da esaminare")
    parser.add_argument("--sendSignal", type=str, required=False, help="Invia il segnale al server MT5")
    parser.add_argument("--notify", type=str, required=False, help="Invia notifica al canale telegram")
    parser.add_argument("--plot", type=str, required=False, help="Generare il grafico")
    parser.add_argument("--interval", type=str, required=False, help="Inserire l'intervallo temporale in minuti")
    parser.add_argument("--favoriteRate", type=str, required=False, help="Inserire il rate di conversione")
    parser.add_argument("--symbol", type=str, required=True, help="Inserire il symbol per avviare il bot")
    args = parser.parse_args()
    SYMBOL = (args.symbol).upper()
    
    if args.sendSignal is not None :
        SEND_SERVER_SIGNAL = str_to_bool(args.sendSignal)
        
    if args.notify is not None :
        SEND_TELEGRAM = str_to_bool(args.notify)
        
    if args.interval is not None :
        FAVORITE_RATE = args.favoriteRate
    
    if args.interval is not None :
        INTERVAL_MINUTES = args.interval
        
    if args.plot is not None:
        GENERATE_PLOT = str_to_bool(args.plot)
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    MODEL_PATH = os.path.join(MODELS_FOLDER, f"lstm_trading_model_{SYMBOL}.h5")
    SCALER_PATH = os.path.join(MODELS_FOLDER, f"scaler_{SYMBOL}.pkl")
    FORECAST_RESULTS_PATH = os.path.join(RESULTS_FOLDER, f"forecast_trading_{SYMBOL}.csv")
    VALIDATION_RESULTS_PATH = os.path.join(RESULTS_FOLDER, f"forecast_validation_{SYMBOL}.csv")
    PLOT_FILE_PATH = os.path.join(PLOTS_FOLDER, f"forecast_trading_{SYMBOL}_{timestamp}.png")
    PLOT_MODEL_FILE_PATH = os.path.join(PLOTS_FOLDER, f"forecast_trading_model_{SYMBOL}_{timestamp}.png")
    DATASET_PATH = os.path.join(DATA_FOLDER, f"DATASET_{SYMBOL}.csv")
    BASE_CURRENCY = SYMBOL[:3]
    QUOTE_CURRENCY = SYMBOL[3:]
    
    rate_exchange = exchange_currency(QUOTE_CURRENCY, FAVORITE_RATE)
    
    if rate_exchange:
        EXCHANGE_RATE = rate_exchange

    run_trading_model()