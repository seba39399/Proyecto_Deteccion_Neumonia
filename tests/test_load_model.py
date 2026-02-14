import pytest
import os

# Usamos 'as' para mapear el nombre real de tu función al nombre que usa el test
from src_directory.load_model import load_any_model as load_trained_model

# -----------------------------
# Test 1: Cuando el modelo existe en la carpeta
# -----------------------------
def test_load_trained_model_success(monkeypatch):
    # Definimos una ruta de carpeta falsa
    fake_dir = "/fake/directory/model"
    # El archivo que glob "encontrará"
    fake_file_path = os.path.join(fake_dir, "modelo_test.h5")

    # 1. Simulamos que glob.glob encuentra un archivo .h5
    monkeypatch.setattr(
        "src_directory.load_model.glob.glob",
        lambda pattern: [fake_file_path]
    )

    # 2. Creamos un objeto de modelo falso
    fake_model_obj = "Soy un modelo de Keras"

    # 3. Simulamos la carga real de TensorFlow
    monkeypatch.setattr(
        "src_directory.load_model.tf.keras.models.load_model",
        lambda path, compile=False: fake_model_obj
    )

    # Ejecutamos la función (que ahora es load_any_model gracias al alias)
    result = load_trained_model(fake_dir)

    # Verificamos que el resultado es el que simulamos
    assert result == fake_model_obj


# -----------------------------
# Test 2: Cuando NO hay archivos .h5 o .keras
# -----------------------------
def test_load_trained_model_not_found(monkeypatch):
    fake_dir = "/fake/directory/empty"

    # Simulamos que glob.glob devuelve una lista vacía
    monkeypatch.setattr(
        "src_directory.load_model.glob.glob",
        lambda pattern: []
    )

    # Verificamos que se lanza la excepción FileNotFoundError
    with pytest.raises(FileNotFoundError) as excinfo:
        load_trained_model(fake_dir)
    
    assert "No se encontró ningún modelo" in str(excinfo.value)