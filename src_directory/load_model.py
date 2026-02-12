import os
import tensorflow as tf

def model_fun():
    """
    Carga un modelo entrenado desde un archivo .h5.


    Parameters
    ----------
    path_model : str
        Ruta al archivo del modelo entrenado.


    Returns
    -------
    tf.keras.Model
        Modelo cargado sin compilar.


    Raises
    ------
    FileNotFoundError
        Si la ruta del modelo no existe.
    """


    # Variable local
    model_path = 'conv_MLP_84.h5'


    # Validar existencia del archivo
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo no existe: {model_path}")


    # Cargar modelo
    model = tf.keras.models.load_model(model_path, compile=False)


    return model