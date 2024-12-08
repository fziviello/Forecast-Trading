from datetime import datetime, time
import pytz
import logging
import yfinance as yf
from gym import Env, spaces
from stable_baselines3 import PPO
import numpy as np

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

def brokerRoule(broker_company):
    if broker_company == "Trading Point Of Financial Instruments Ltd":
        return "#"
    return ""

def forex_market_status(symbol: str) -> bool:
    # Orari di apertura e chiusura in UTC
    forex_open = time(22, 0)  # 22:00 UTC (domenica)
    forex_close = time(22, 0)  # 22:00 UTC (venerdì)
    
    now_utc = datetime.now(pytz.timezone('Europe/Rome'))
    current_time = now_utc.time()
    current_weekday = now_utc.weekday()  # 0 = lunedì, 6 = domenica
    
    current_time_utc = now_utc.astimezone(pytz.utc).time()
    
    if current_weekday == 5 or (current_weekday == 6 and current_time_utc >= forex_close):
        print(f"\033[91mIl mercato {symbol} è chiuso\033[0m")
        logging.info(f"Il mercato {symbol} è chiuso\n")
        return False
    elif current_weekday == 6 and current_time_utc < forex_open:
        print(f"\033[91mIl mercato {symbol} è chiuso\033[0m")
        logging.info(f"Il mercato {symbol} è chiuso\n")
        return False
    else:
        print(f"\033[92mIl mercato {symbol} è aperto\033[0m")
        logging.info(f"Il mercato {symbol} è aperto\n")
        return True
    
def train_rl_model(data):
    env = ForexTradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

class ForexTradingEnv(Env):
    def __init__(self, data, initial_balance=10000):
        super(ForexTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.done = False

        # Spazio degli stati (feature di mercato)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

        # Spazio delle azioni: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
    
    def _get_observation(self):
        return self.data.iloc[self.current_step].values

    def _calculate_reward(self, action, entry_price, current_price):
        # Reward per azione presa
        if action == 1:  # Buy
            return max(0, current_price - entry_price)
        elif action == 2:  # Sell
            return max(0, entry_price - current_price)
        return 0  # Hold
    
    def step(self, action):
        current_price = self.data['Close'].iloc[self.current_step]
        next_price = self.data['Close'].iloc[self.current_step + 1]

        # Aggiorna bilancio sulla base dell'azione
        reward = self._calculate_reward(action, current_price, next_price)
        self.balance += reward

        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 2

        return self._get_observation(), reward, self.done, {}
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.done = False
        return self._get_observation()