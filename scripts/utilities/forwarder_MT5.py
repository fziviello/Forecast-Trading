import logging
import requests
import json

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
            print(f"Errore nella conversione dei parametri in float: {e}")
            logging.error(f"Errore nella conversione dei parametri in float: {e}")
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
            response = requests.post(url, headers=headers, data=json.dumps(data))
            data = response.json()
            if response.status_code == 200:
                print(f"Ordine creato con successo: {data}")
                logging.info(f"Ordine creato con successo: {data}")
                order_id = data["order_id"]
                return order_id
                
            else:
                print(f"Errore nella creazione dell'ordine: {response.status_code}, {response.text}")
                logging.error(f"Errore nella creazione dell'ordine: {response.status_code}, {response.text}")
                error_message = data["message"]
                error_detail = error_message.split(":", 1)[1].strip()
                return error_detail
        
        except Exception as e:
            print(f"Errore durante la richiesta al server: {str(e)}")
            logging.error(f"Errore durante la richiesta al server: {str(e)}")
        
        return "Server Request Error"

    def update_order(self, ticket, stop_loss=None, take_profit=None):
        url = f"{self.server_url}/order/update"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "ticket": ticket,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

        data = {k: v for k, v in data.items() if v is not None}

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            data = response.json()
            if response.status_code == 200:
                print(f"Ordine aggiornato con successo: {response.json()}")
                logging.info(f"Ordine aggiornato con successo: {response.json()}")
                order_id = data["order_id"]
                return order_id
            else:
                print(f"Errore nell'aggiornamento dell'ordine: {response.status_code}, {response.text}")
                logging.error(f"Errore nell'aggiornamento dell'ordine: {response.status_code}, {response.text}")
                error_message = data["message"]
                error_detail = error_message.split(":", 1)[1].strip()
                return error_detail
        
        except Exception as e:
            print(f"Errore durante la richiesta al server: {str(e)}")
            logging.error(f"Errore durante la richiesta al server: {str(e)}")
            
        return "Server Request Error"

    def delete_order(self, ticket):
        url = f"{self.server_url}/order/delete"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "ticket": ticket
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            data = response.json()
            if response.status_code == 200:
                print(f"Ordine cancellato con successo: {response.json()}")
                logging.info(f"Ordine cancellato con successo: {response.json()}")
                order_id = data["order_id"]
                return order_id
            else:
                print(f"Errore nella cancellazione dell'ordine: {response.status_code}, {response.text}")
                logging.error(f"Errore nella cancellazione dell'ordine: {response.status_code}, {response.text}")
                error_message = data["message"]
                error_detail = error_message.split(":", 1)[1].strip()
                return error_detail
        
        except Exception as e:
            print(f"Errore durante la richiesta al server: {str(e)}")
            logging.error(f"Errore durante la richiesta al server: {str(e)}")
        return "Server Request Error"