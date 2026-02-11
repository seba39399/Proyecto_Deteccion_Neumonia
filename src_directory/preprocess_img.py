import cv2
import numpy as np

def preprocess(array):

    """
    Este módulo se encarga de preprocesar una imagen de entrada para el modelo de detección de neumonía. Realiza redimensionamiento a 512x512, conversión a escala de grises, mejora de contraste mediante CLAHE, normalización en el rango [0,1] y adaptación de la forma del arreglo a un tensor compatible con el modelo.

    Args:
        array (np.ndarray): Imagen de entrada en formato numpy.

    Returns:
        np.ndarray: Tensor preprocesado con forma (1, 512, 512, 1) y tipo float32.
    """

    array = cv2.resize(array, (512, 512))
    if len(array.shape) == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array.astype('float32')
