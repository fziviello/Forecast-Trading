import argparse
import platform
import os

def colorize_amount(amount):
    if amount.startswith('-'):
        return f"\033[91m{amount}\033[0m"
    else:
        return f"\033[92m{amount}\033[0m"
    
def str_to_bool(value):
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Valore booleano non valido: {value}")
    
def get_system_type():
    system = platform.system()
    if system == "Windows":
        return "Windows"
    elif system == "Darwin":
        return "macOS"
    elif system == "Linux":
        return "Linux"
    else:
        return "Sistema operativo non riconosciuto"

def get_all_exist_models(path, extension=".pkl"):
    if not os.path.isdir(path):
        print(f"\033[91mIl percorso specificato '{path}' non è una directory valida\033[0m")
        raise ValueError(f"Il percorso specificato '{path}' non è una directory valida")
    
    files = [
        f[len("scaler_"):-len(extension)]
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.startswith("scaler_") and f.endswith(extension)
    ]
    
    if not files:
        print(f"\033[93mNessun modello con estensione '{extension}' trovato in '{path}'\033[0m")
    
    return files