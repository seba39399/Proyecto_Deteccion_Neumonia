import numpy as np
from src_directory.preprocess_img import preprocess

def test_preprocess_dtype():
    """En esta función se realiza una validación del tipo de dato que retorna preprocess, el cual debe ser float 32"""
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    result = preprocess(img)

    assert result.dtype == np.float32