import numpy as np
from src_directory.preprocess_img import preprocess

def test_preprocess_shape():
    """Se realiza una verificación del tamaño del tensor que retorna la función preprocess, el cual debe ser(1, 512, 512, 1)"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    result = preprocess(img)

    assert result.shape == (1, 512, 512, 1)