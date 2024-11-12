import os
import subprocess
import schedule
import logging
import time
import argparse
from datetime import datetime

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG_FOLDER = BASE_PATH + '/logs'
LOG_FILE_PATH = 'training.log'
TIME_MINUTE_REPEAT = 10
N_REPEAT = 60

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
    print(f"\033[92mCartella '{LOG_FOLDER}' creata con successo.\033[0m")

logging.basicConfig(filename=os.path.join(LOG_FOLDER, LOG_FILE_PATH), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_scripts_for_symbol(symbol):
    print(f"\033[93m*** Avvio Creazione del DataSet per {symbol}\033[0m\n")
    process1 = subprocess.Popen(
        ["python", "create_dataSet.py", "--symbol", symbol],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True
    )

    if process1.stdout:
        for line in process1.stdout:
            print(line, end='', flush=True)
    if process1.stderr:
        for line in process1.stderr:
            logging.error(line.strip())

    process1.stdout.close()
    process1.stderr.close()
    process1.wait()

    print(f"\n\033[93m*** Avvio Forecast per {symbol}\033[0m\n")
    process2 = subprocess.Popen(
        ["python", "forecast_bot.py", "--symbol", symbol],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True
    )

    if process2.stdout:
        for line in process2.stdout:
            print(line, end='', flush=True)
    if process2.stderr:
        for line in process2.stderr:
            logging.error(line.strip())

    process2.stdout.close()
    process2.stderr.close()
    process2.wait()

def run_scripts(symbols):
    for symbol in symbols:
        run_scripts_for_symbol(symbol)

def schedule_scripts(symbols, N_REPEAT):
    executions = {'count': 0}

    def job():
        print(f"\033[96mESECUZIONE {executions['count'] + 1}\033[0m\n")
        if executions['count'] < N_REPEAT:
            run_scripts(symbols)
            executions['count'] += 1
        else:
            print("Esecuzioni Programmate Terminate")
            return schedule.CancelJob

    job()

    schedule.every(TIME_MINUTE_REPEAT).minutes.do(job)

    while executions['count'] < N_REPEAT:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scheduler per esecuzione script multipli.")
    parser.add_argument("--symbols", type=str, required=True, help="Lista di simboli separati da virgola (es: EURUSD,AUDJPY,GBPUSD)")
    args = parser.parse_args()

    symbols = (args.symbols.upper()).split(',')
    now = datetime.now()
    print(f'\033[96mTrainer Avviato il {now.strftime("%Y-%m-%d %H:%M:%S")} per {N_REPEAT} esecuzioni\033[0m\n')
    
    schedule_scripts(symbols, N_REPEAT)
