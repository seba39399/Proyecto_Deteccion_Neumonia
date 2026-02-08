import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
import numpy as np
import cv2

class AIService:
    def __init__(self, model_path):
        """Inicializa el modelo y configura los parámetros base"""
        # Cargamos el modelo sin compilar para máxima compatibilidad
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.target_size = (512, 512)
        self.labels = {0: "Bacteriana", 1: "Normal", 2: "Viral"}

    def preprocess(self, img_array):
        """Prepara la imagen y limpia discrepancias de tipo de dato."""
        # Redimensionar imágenes
        img = cv2.resize(img_array, self.target_size)
        
        # Escala de grises
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Mejora de contraste (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)
        
        # Normalización y ajuste de dimensiones
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)  # (512, 512, 1)
        img = np.expand_dims(img, axis=0)   # (1, 512, 512, 1)
        
        # Convertimos a float32 para evitar el warning de estructura de Keras
        return img.astype('float32')

    def get_gradcam(self, original_array, preprocessed_img):
        """Calcula el mapa de calor (Grad-CAM)"""
        last_conv_layer = self.model.get_layer("conv10_thisone")
        
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [last_conv_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_img)
            if isinstance(predictions, list): 
                predictions = predictions[0]
            
            class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalización del Heatmap
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
        heatmap = cv2.resize(heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap, self.target_size)
        heatmap = np.uint8(255 * heatmap)
        
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Preparar fondo
        background = cv2.resize(original_array, self.target_size)
        if len(background.shape) == 2 or background.shape[2] == 1:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
            
        superimposed_img = cv2.addWeighted(background, 0.6, heatmap_color, 0.4, 0)
        return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    def predict(self, original_array):
        """Realiza la predicción silenciando warnings de consola."""
        img_ready = self.preprocess(original_array)
        
        # Realizamos la predicción. Verbose=0 apaga la barra de carga.
        res = self.model.predict(img_ready, verbose=0)
        
        # Si el modelo retorna una lista (común en Keras 3 functional API)
        if isinstance(res, list): 
            res = res[0]
        
        idx = np.argmax(res[0])
        proba = np.max(res[0]) * 100
        label = self.labels.get(idx, "Desconocido")
        
        heatmap_img = self.get_gradcam(original_array, img_ready)
        
        return label, proba, heatmap_img