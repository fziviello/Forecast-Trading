import argparse

def str_to_bool(value):
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Valore booleano non valido: {value}")