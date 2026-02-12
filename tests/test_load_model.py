import pytest

from src_directory.load_model import model_fun


# -----------------------------
# Test 1: cuando el archivo existe
# -----------------------------
def test_model_fun_when_file_exists(monkeypatch):

    # Simula que el archivo SÍ existe
    monkeypatch.setattr(
        "src_directory.load_model.os.path.exists",
        lambda path: True
    )

    # Modelo falso
    fake_model = object()

    # Simula load_model
    monkeypatch.setattr(
        "src_directory.load_model.tf.keras.models.load_model",
        lambda path, compile=False: fake_model
    )

    model = model_fun()

    assert model is fake_model


# -----------------------------
# Test 2: cuando el archivo NO existe
# -----------------------------
def test_model_fun_when_file_not_exists(monkeypatch):

    # Simula que el archivo NO existe
    monkeypatch.setattr(
        "src_directory.load_model.os.path.exists",
        lambda path: False
    )

    with pytest.raises(FileNotFoundError):
        model_fun()
