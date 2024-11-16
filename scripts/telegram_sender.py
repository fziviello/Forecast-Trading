import requests

class TelegramSender:
    def __init__(self, bot_token):
        self.bot_token = bot_token

    def sendMsg(self,message, channel):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": channel,
            "text": message
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print(f"Messaggio inviato al canale {channel} con successo.")
        except requests.exceptions.RequestException as e:
            print(f"Errore nell'invio del messaggio: {e}")
