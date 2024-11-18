import logging
import requests
from datetime import datetime, timedelta, timezone

class TelegramSender:
    def __init__(self, bot_token):
        self.bot_token = bot_token

    def sendMsg(self, message, channel):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": channel,
            "text": message
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            message_data = response.json()
            message_id = message_data['result']['message_id']
            logging.debug(response.json())
            logging.info(f"Messaggio inviato al canale {channel} con successo con ID {message_id}")
            print(f"\033[92mMessaggio inviato al canale {channel} con successo con ID {message_id}\033[0m")
        except requests.exceptions.RequestException as e:
            logging.error(f"Errore nell'invio del messaggio: {e}")
            print(f"\033[91mErrore nell'invio del messaggio: {e}\033[0m")

    def getBotMsg(self, offset=None):
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {"timeout": 10}
        if offset is not None:
            params["offset"] = offset

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            updates = response.json()
            logging.debug(f"Risposta API: {response.json()}")
            if updates["ok"]:
                return updates["result"]
            else:
                logging.error(f"Errore nel recupero dei messaggi: risposta non valida")
                print(f"\033[91mErrore nel recupero dei messaggi: risposta non valida.\033[0m")
                return []
        except requests.exceptions.RequestException as e:
            logging.error(f"Errore nel recupero dei messaggi: {e}")
            print(f"\033[91mErrore nel recupero dei messaggi: {e}\033[0m")
            return []