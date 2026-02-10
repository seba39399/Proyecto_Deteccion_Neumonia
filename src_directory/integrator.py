import numpy as np
from src_directory.preprocess_img import preprocess
from src_directory.grad_cam import grad_cam


def predict(array, model):
    """
    Realiza una predicción sobre una imagen utilizando un modelo entrenado
    y genera un mapa de activación Grad-CAM.

    Args:
        array:
            Imagen de entrada en formato NumPy (sin preprocesar).

        model:
            Modelo entrenado para clasificación de imágenes.

    Returns:
        tuple:
            Una tupla con tres elementos:
                - label (str): Clase predicha ("bacteriana", "normal", "viral").
                - proba (float): Probabilidad asociada a la predicción (en porcentaje).
                - heatmap (np.ndarray): Mapa de calor generado con Grad-CAM.

    Example:
        label, proba, heatmap = predict(img_array, model)
    """

    batch_array_img = preprocess(array)

    res = model.predict(batch_array_img, verbose=0)

    if isinstance(res, list):
        res = res[0]

    prediction = np.argmax(res[0])
    proba = np.max(res[0]) * 100

    label_dict = {
        0: "bacteriana",
        1: "normal",
        2: "viral"
    }

    label = label_dict.get(prediction, "Desconocido")

    heatmap = grad_cam(array, model)

    return label, proba, heatmap
