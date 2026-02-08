import numpy as np
from src_directory.preprocess_img import preprocess
from src_directory.grad_cam import grad_cam

def predict(array, model):
    batch_array_img = preprocess(array)
    res = model.predict(batch_array_img, verbose=0)
    
    if isinstance(res, list): res = res[0]
    prediction = np.argmax(res[0])
    proba = np.max(res[0]) * 100
    
    label_dict = {0: "bacteriana", 1: "normal", 2: "viral"}
    label = label_dict.get(prediction, "Desconocido")
    
    heatmap = grad_cam(array, model)
    return (label, proba, heatmap)