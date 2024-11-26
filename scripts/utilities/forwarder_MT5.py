import logging
import requests
import json
from datetime import datetime, timedelta, timezone
import pytz

def filter_expired_orders(symbol=None, orders=None, max_minutes=60):
    if orders is None:
        logging.warning("Lista degli ordini vuota o non fornita.")
        return []
    
    old_orders = []
    italy_timezone = pytz.timezone('Europe/Rome')
    current_time_italy = datetime.now(italy_timezone)
    
    for order in orders:
        try:
            logging.debug(f"Analizzando ordine con simbolo {order.get('symbol')} e time_setup {order.get('time_setup')}")
            
            if symbol and order.get("symbol") != symbol:
                logging.debug(f"Ordine ignorato, simbolo non corrisponde. Simbolo richiesto: {symbol}, simbolo ordine: {order.get('symbol')}")
                continue
            
            server_timezone = pytz.timezone('Etc/GMT+1')
            order_time = datetime.fromtimestamp(order["time_setup"], tz=server_timezone)
            time_difference = current_time_italy - order_time
            logging.debug(f"Tempo trascorso dall'ordine: {time_difference}")

            if time_difference < timedelta(minutes=max_minutes):
                old_orders.append(order)
                logging.debug(f"Ordine aggiunto alla lista: {order}")
        
        except KeyError as e:
            logging.warning(f"Ordine con chiave mancante: {e}")
        except Exception as e:
            logging.error(f"Errore generico durante il filtraggio degli ordini: {e}")
    
    return old_orders
    
class TradingAPIClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def create_order(self, symbol, order_type, volume, price, stop_loss=None, take_profit=None):
        url = f"{self.server_url}/order"
        headers = {"Content-Type": "application/json"}
        
        order_type = (order_type.replace(" ", "_")).lower()
        
        symbol = symbol+"#"
                
        try:
            price = round(float(price), 3) if price is not None else None
            stop_loss = round(float(stop_loss), 3) if stop_loss is not None else None
            take_profit = round(float(take_profit), 3) if take_profit is not None else None
        except ValueError as e:
            print(f"\033[91mErrore nella conversione dei parametri in float: {str(e)}\033[0m")
            logging.error(f"Errore nella conversione dei parametri in float: {str(e)}")
            return

        data = {
            "symbol": symbol,
            "type": order_type,
            "volume": volume,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        data = {k: v for k, v in data.items() if v is not None}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            data = response.json()
            if response.status_code == 200:
                order_id = data["order_id"]
                print(f"\033[92mOrdine {order_id} piazzato con successo\033[0m")
                logging.info(f"Ordine piazzato con successo: {data}")
                return order_id
                
            else:
                error_message = data["message"]
                print(f"\033[91mOrdine non piazzato: {error_message}\033[0m")
                logging.error(f"Errore nella creazione dell'ordine: {response.status_code}, {response.text}")
                error_detail = error_message.split(":", 1)[1].strip()
                return error_detail
        
        except requests.exceptions.Timeout:
            print(f"\033[91mTimeout: Il server non ha risposto entro il tempo limite.\033[0m")
            logging.error(f"Timeout: Il server non ha risposto entro il tempo limite.")     
            return "Timeout Server Error"
        except requests.exceptions.RequestException as e:
            print(f"\033[91mErrore durante la richiesta al server: {str(e)}\033[0m")
            logging.error(f"Errore durante la richiesta al server: {str(e)}")      
            return "Server Request Error"

    def update_order(self, ticket, price, stop_loss, take_profit):
        url = f"{self.server_url}/order/update"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "ticket": ticket,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

        data = {k: v for k, v in data.items() if v is not None}

        try:
            response = requests.put(url, headers=headers, data=json.dumps(data), timeout=10)
            data = response.json()
            if response.status_code == 200:
                order_id = data["order_id"]
                print(f"\033[92mOrdine {order_id} aggioranto:\033[0m")
                logging.info(f"Ordine aggiornato: {response.json()}")
                return order_id
            else:
                error_message = data["message"]
                print(f"\033[91mOrdine non aggioranto: {error_message}\033[0m")
                logging.error(f"Errore nell'aggiornamento dell'ordine: {response.status_code}, {response.text}")
                error_detail = error_message.split(":", 1)[1].strip()
                return error_detail
        
        except requests.exceptions.Timeout:
            print(f"\033[91mTimeout: Il server non ha risposto entro il tempo limite.\033[0m")
            logging.error(f"Timeout: Il server non ha risposto entro il tempo limite.")     
            return "Timeout Server Error"
        except requests.exceptions.RequestException as e:
            print(f"\033[91mErrore durante la richiesta al server: {str(e)}\033[0m")
            logging.error(f"Errore durante la richiesta al server: {str(e)}")      
            return "Server Request Error"
            
    def delete_order(self, ticket):
        url = f"{self.server_url}/order/delete"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "ticket": ticket
        }

        try:
            response = requests.delete(url, headers=headers, data=json.dumps(data), timeout=10)
            data = response.json()
            if response.status_code == 200:
                print(f"\033[92mOrdine {ticket} eliminato con successo\033[0m")
                logging.info(f"Ordine eliminato con successo: {response.json()}")
                return ticket
            else:
                error_message = data["message"]
                print(f"\033[91mOrdine {ticket} non eliminato: {error_message}\033[0m")
                logging.error(f"Errore nella cancellazione dell'ordine: {response.status_code}, {response.text}")
                error_detail = error_message.split(":", 1)[1].strip()
                return error_detail
        
        except requests.exceptions.Timeout:
            print(f"\033[91mTimeout: Il server non ha risposto entro il tempo limite.\033[0m")
            logging.error(f"Timeout: Il server non ha risposto entro il tempo limite.")     
            return "Timeout Server Error"
        except requests.exceptions.RequestException as e:
            print(f"\033[91mErrore durante la richiesta al server: {str(e)}\033[0m")
            logging.error(f"Errore durante la richiesta al server: {str(e)}")      
            return "Server Request Error"
        
    def get_orders_palaced(self, symbol=None, max_minutes=60):
        url = f"{self.server_url}/order/placed"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            if response.status_code == 200:
                orders = filter_expired_orders(symbol, data.get("orders", []), max_minutes)
                return orders
            else:
                error_message = data["message"]
                print(f"\033[91mErrore nella ricezione degli ordini attivi: {error_message}\033[0m")
                logging.error(f"Errore nella ricezione degli ordini attivi: {response.status_code}, {response.text}")
                error_detail = error_message.split(":", 1)[1].strip()
                return error_detail
        
        except requests.exceptions.Timeout:
            print(f"\033[91mTimeout: Il server non ha risposto entro il tempo limite.\033[0m")
            logging.error(f"Timeout: Il server non ha risposto entro il tempo limite.")     
            return "Timeout Server Error"
        
        except requests.exceptions.RequestException as e:
            print(f"\033[91mErrore durante la richiesta al server: {str(e)}\033[0m")
            logging.error(f"Errore durante la richiesta al server: {str(e)}")      
            return "Server Request Error"
        