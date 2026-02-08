import pydicom as dicom
import cv2
import numpy as np
from PIL import Image

class DataManager:
    """
    Clase encargada de la gestión de archivos y conversión de formatos
    """

    def read_file(path):
        """
        Detecta el tipo de archivo y lo convierte en un array de NumPy 
        y un objeto de imagen para la interfaz
        """
        path_lower = path.lower()
        
        if path_lower.endswith('.dcm'):
            return DataManager._read_dicom(path)
        else:
            return DataManager._read_standard(path)

    def _read_dicom(path):
        """Procesa archivos médicos DICOM, normalizando y asegurando formato RGB"""
        ds = dicom.dcmread(path)
        img_array = ds.pixel_array
        
        # Normalización, los DICOM pueden tener valores > 4000 (12-16 bits)
        # Convertimos al rango 0-255 para visualización estándar
        img_norm = (np.maximum(img_array, 0) / img_array.max()) * 255.0
        img_uint8 = np.uint8(img_norm)
        
        # Aseguramos que sea RGB para la interfaz
        if len(img_uint8.shape) == 2:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_uint8
            
        return img_rgb, Image.fromarray(img_rgb)

    def _read_standard(path):
        """Procesa archivos estándar (JPG, PNG, JPEG)."""
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"No se pudo leer la imagen en la ruta: {path}")
            
        # Convertir BGR (OpenCV) a RGB (PIL/Tkinter)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb, Image.fromarray(img_rgb)

    def save_to_csv(filepath, data):
        """Guarda los resultados en el historial."""
        import csv
        import os
        
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter="-")
            # Si el archivo es nuevo, podrías escribir cabeceras aquí
            writer.writerow(data)