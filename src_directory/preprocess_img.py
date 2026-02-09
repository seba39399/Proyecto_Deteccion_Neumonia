import cv2
import numpy as np

def preprocess(array):

    """
    Este módulo se encarga de preprocesar una imagen de entrada para el modelo de detección de neumonía. Realiza redimensionamiento a 512x512, conversión a escala de grises, mejora de contraste mediante CLAHE, normalización en el rango [0,1] y adaptación de la forma del arreglo a un tensor compatible con el modelo.

    Entradas (argumentos):
        array (np.ndarray): Imagen de entrada en formato numpy.

    Returns (Salidas):
        np.ndarray: Tensor preprocesado con forma (1, 512, 512, 1) y tipo float32.
    """

   #1. Redimensionamiento de la imagen a 512x512 píxeles para cumplir con el tamaño de entrada del modelo
    array = cv2.resize(array, (512, 512))

    #2. Si la imagen tiene tres canales (RGB/BGR), se convierte a escala de grises
    if len(array.shape) == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    #3. Creación de un objeto CLAHE para mejorar el contraste local de la imagen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # Aplicación de CLAHE a la imagen en escala de grises
    array = clahe.apply(array)

    # Normalización de los valores de los píxeles al rango [0, 1]
    array = array / 255

    # Agrega el canal (depth) para cumplir con el formato (alto, ancho, canales)
    array = np.expand_dims(array, axis=-1)

    # Agrega la dimensión de batch para el modelo (batch_size, alto, ancho, canales)
    array = np.expand_dims(array, axis=0)

    # Conversión del arreglo a tipo float32, requerido por TensorFlow/Keras
    return array.astype('float32')
