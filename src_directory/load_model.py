import os
import tensorflow as tf
import glob

def load_any_model(directory="model"):
    """
    Busca automáticamente el primer modelo disponible (.h5) en el directorio especificado y lo carga.

    Args:
        directory (str): Ruta del directorio donde se buscan los modelos. 
            Por defecto es "model".
    Returns:
        tf.keras.Model: El modelo cargado listo para usar.

    """
    # Buscamos archivos que terminen en .h5
    formatos = ['*.h5']
    archivos_modelo = []
    
    for formato in formatos:
        archivos_modelo.extend(glob.glob(os.path.join(directory, formato)))

    # Si no hay archivos, lanzamos error
    if not archivos_modelo:
        raise FileNotFoundError(f"No se encontró ningún modelo (.h5 o .keras) en: {os.path.abspath(directory)}")

    # Tomamos el primero que encuentre 
    model_path = archivos_modelo[0]
    
    print(f"--- Cargando modelo detectado: {model_path} ---")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Error técnico al cargar {model_path}: {e}")

if __name__ == "__main__":
    model = load_any_model()