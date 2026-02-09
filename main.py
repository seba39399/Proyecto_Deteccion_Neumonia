import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencio total de logs informativos
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Desactiva el mensaje de oneDNN

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from src_directory.gui import App

def main():
    """
    Punto de entrada principal. Configura el sistema e inicia la GUI.
    """
    print("Iniciando Sistema de Diagnóstico...")
    try:
        App()
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")

if __name__ == "__main__":
    main()