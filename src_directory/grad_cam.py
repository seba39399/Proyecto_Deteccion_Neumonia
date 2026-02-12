import tensorflow as tf
import numpy as np
import cv2
from src_directory.preprocess_img import preprocess

def grad_cam(array, model):
    """Recibe la imagen y la pasa por el modelo, toma la última capa convolucional y a partir de esta
    calcula el gradiente de la clase predicha, luego se usan esos gradientes como pesos para generar un mapa de calor y superponerlo a la imagen original
    
    Args:
        array (numpy.ndarray): Array original de la imagen (puede ser escala de grises o RGB).
        model (tf.keras.Model): Modelo de inteligencia artificial previamente cargado.
                               Debe contener una capa llamada 'conv10_thisone'.

    Returns:
        numpy.ndarray: Imagen final en formato RGB (512, 512, 3) con el mapa de calor 
                       superpuesto mediante transparencia (0.6 original / 0.4 heatmap).
    
    """
    img = preprocess(array)

    #Se obtiene la última capa convolucional
    last_conv_layer = model.get_layer("conv10_thisone") 

    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    #Se calculan gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        if isinstance(predictions, list): predictions = predictions[0]
        argmax = np.argmax(predictions[0])
        loss = predictions[:, argmax]

    grads = tape.gradient(loss, conv_outputs)

    #Se promedia  los gradientes para obtener la importancia de cada filtro
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    #Se multiplican mapas de activación por su importancia
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Corrección del error .numpy()
    if hasattr(heatmap, 'numpy'):
        heatmap = heatmap.numpy()
        
    #Se normaliza, se realiza un resize y se suporpone la imagen para obtener la imagen con el mapa de calor
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img2 = cv2.resize(array, (512, 512))
    if len(img2.shape) == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    superimposed_img = cv2.addWeighted(img2, 0.6, heatmap, 0.4, 0)
    return superimposed_img[:, :, ::-1] # BGR a RGB