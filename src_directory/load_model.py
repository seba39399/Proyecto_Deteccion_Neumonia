import tensorflow as tf

def model_fun():
    """Carga el modelo una sola vez."""
    model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
    return model