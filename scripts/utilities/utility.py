import argparse
import platform

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