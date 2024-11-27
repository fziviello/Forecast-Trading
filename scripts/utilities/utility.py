import argparse
import platform

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