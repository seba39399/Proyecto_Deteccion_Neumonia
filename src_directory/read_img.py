import pydicom as dicom
import cv2
import numpy as np
from PIL import Image

def read_dicom_file(path):
    """
    Lee un archivo en formato DICOM y lo prepara para visualización y procesamiento.

    Esta función extrae los datos de píxeles del archivo médico, normaliza los valores 
    para que se encuentren en el rango de 0-255 y convierte la imagen de escala 
    de grises a RGB.

    Args:
        path (str): Ruta local del archivo .dcm a procesar.

    Returns:
        tuple: Una tupla que contiene:
            - img_RGB (numpy.ndarray): Arreglo de la imagen en formato RGB (NumPy array).
            - img2show (PIL.Image.Image): Objeto de imagen listo para mostrar en la interfaz.
    """
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    
    # Normalización de la imagen
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show

def read_jpg_file(path):
    """
    Lee un archivo de imagen estándar (JPG, PNG, JPEG) y lo normaliza.

    Convierte la imagen cargada en un arreglo de NumPy y genera una versión 
    normalizada para asegurar la consistencia en el despliegue visual.

    Args:
        path (str): Ruta local de la imagen (.jpg, .png, etc.).

    Returns:
        tuple: Una tupla que contiene:
            - img2 (numpy.ndarray): Arreglo normalizado de la imagen.
            - img2show (PIL.Image.Image): Objeto de imagen para uso en la UI.
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    
    # Normalización para consistencia visual
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    
    return img2, img2show