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
            print(f"\033[92mMessaggio inviato al canale {channel} con successo.\033[0m")
        except requests.exceptions.RequestException as e:
            print(f"\033[91mErrore nell'invio del messaggio: {e}\033[0m")
