import os
import subprocess
import schedule
import logging
import time
from datetime import datetime

LOG_FOLDER = 'LOGS'
LOG_FILE_PATH = 'scheduler.log'
SCRIPT_CREATE_DATASET = "create_dataSet.py"
SCRIPT_FORECAST = "forecast_bot.py --symbol AUDJPY"
TIME_MINUTE_REPEAT = 40
N_REPEAT = 5

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
    print(f"\033[92mCartella '{LOG_FOLDER}' creata con successo.\033[0m")
    
logging.basicConfig(filename=os.path.join(LOG_FOLDER, LOG_FILE_PATH), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
   
def run_scripts(SCRIPT_CREATE_DATASET, SCRIPT_FORECAST):
    print(f"\033[93m*** Avvio Creazione del DataSet\033[0m\n")
    process1 = subprocess.Popen(
        ["python", SCRIPT_CREATE_DATASET],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True
    )

    for line in process1.stdout:
        print(line, end='', flush=True)
    for line in process1.stderr:
        logging.error(line.strip())

    process1.stdout.close()
    process1.stderr.close()
    process1.wait()

    print(f"\n\033[93m*** Avvio Forecast\033[0m\n")
    process2 = subprocess.Popen(
        ["python", SCRIPT_FORECAST],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True
    )

    for line in process2.stdout:
        print(line, end='', flush=True)
    for line in process2.stderr:
        logging.error(line.strip())

    process2.stdout.close()
    process2.stderr.close()
    process2.wait()

def schedule_scripts(SCRIPT_CREATE_DATASET, SCRIPT_FORECAST, N_REPEAT):
    executions = {'count': 0}

    def job():
        print(f"\033[96mESECUZIONE {executions['count']+1}\033[0m\n")
        if executions['count'] < N_REPEAT:
            run_scripts(SCRIPT_CREATE_DATASET, SCRIPT_FORECAST)
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
    now = datetime.now()
    print(f'\033[96mTrainer Avviato il {now.strftime("%Y-%m-%d %H:%M:%S")} per {N_REPEAT} esecuzioni\033[0m\n')
    schedule_scripts(SCRIPT_CREATE_DATASET, SCRIPT_FORECAST, N_REPEAT)
