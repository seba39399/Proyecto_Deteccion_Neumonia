import tensorflow as tf

def model_fun():
    """
    Carga un modelo entrenado de TensorFlow/Keras desde un archivo .h5.

    Args:
        path_model (str):
            Ruta del archivo del modelo en formato .h5.

    Returns:
        tf.keras.Model:
            Modelo de Keras cargado y listo para inferencia.

    Example:
        model = model_fun("conv_MLP_84.h5")
    """
    
    model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
    return model

