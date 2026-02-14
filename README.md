# Práctica: Detección rápida de Neumonía con Deep Learning + Grad‑CAM

**Estudiantes:**  
- Miguel Angel Franco Restrepo (22506163)  
- Saulo Quiñones Góngora (22506635)  
- Adrian Felipe Vargas Rojas (22505561)
- Juan Sebastián Peña Valderrama (22502483)  

**Curso:** Desarrollo de Proyectos de IA  
**Institución:** Universidad Autónoma de Occidente  
**Periodo:** 2026‑1  
**Repositorio:** Proyecto_Deteccion_Neumonia

---

## 1. Resumen del proyecto

Este proyecto implementa una herramienta académica para apoyar la detección rápida de neumonía en radiografías de tórax (DICOM o imágenes estándar), mediante un modelo de Deep Learning (CNN en Keras/TensorFlow). Adicionalmente, incorpora **Grad‑CAM** para generar un mapa de calor que resalta las regiones de la imagen que más influyeron en la clasificación.

**Clases de salida:**
- Neumonía Bacteriana  
- Neumonía Viral  
- Sin Neumonía (Normal)

---

## 2. Objetivo de la práctica

El objetivo principal de la práctica es evidenciar un diseño modular con alta cohesión y bajo acoplamiento, donde:

- Cada módulo cumple una responsabilidad única (leer, preprocesar, cargar modelo, explicar, integrar).  
- La interfaz (GUI) no implementa la lógica de IA: solo la consume mediante un servicio/función de predicción.  
- El flujo se puede validar con pruebas simples utilizando frameworks como `Pytest`.

---

## 3. Estructura del proyecto

```
Proyecto_Deteccion_Neumonia/
│
├── src_directory/                         # Núcleo del sistema (alta cohesión)
│   ├── __init__.py
│   ├── read_img.py                        # Lectura de DICOM / JPG → np.ndarray
│   ├── preprocess_img.py                  # Preprocesamiento → tensor (1,512,512,1)
│   ├── load_model.py                      # Carga del modelo conv_MLP_84.h5
│   ├── grad_cam.py                        # Generación de heatmap (Grad-CAM)
│   ├── integrator.py                      # Orquestación del flujo completo
│   └── gui.py                             # Interfaz gráfica Tkinter
│
├── tests/
│   ├── __pycache__/
│   ├── test_shape_preprocess.py           # Prueba del preprocesamiento (shape del tensor)
│   └── test_load_model.py                 # Prueba de carga del modelo (.h5)
│
│   .gitignore                             # Archivo donde se indican que elementos no se deben
│                                            cargar al repositorio de Guithub
├── docker.dockerignore                    # Archivo donde se indican que elementos se deben
│                                            incluir en la imagen del proyecto
├── Dockerfile                             # Archivo de configuración para la creación de la
│                                            imagen del proyecto
├── favicon.ico                            # Ícono de la aplicación
├── main.py                                # Punto de entrada principal (GUI)
│
├── pyproject.toml                         # Dependencias del proyecto (UV)
├── uv.lock                                # Lockfile reproducible (UV)
├── pytest.ini                             # Configuración de pruebas
├── .python-version                        # Versión de Python usada
└── README.md

```
---

## 4. Requerimientos y entorno (UV)

### 4.1 ¿Por qué se usa UV y no `requirements.txt`?

Este proyecto usa UV como gestor moderno de entornos y dependencias. En lugar de mantener un `requirements.txt` manual, UV trabaja con:

- **`pyproject.toml`**: declara las dependencias del proyecto (fuente de verdad).
- **`uv.lock`**: bloquea versiones exactas para garantizar reproducibilidad.

Al utilizar este gestor se fomentan prácticas adecuadas de desarrollo de software, teniendo en cuenta que:

- Previene inconsistencias entre entornos de desarrollo.
- Permite la recreación determinística del entorno.
- Disminuye errores derivados de diferencias locales de configuración.

Considerando lo anterior, no es necesario un `requirements.txt`, teniendo en cuenta que **`pyproject.toml` + `uv.lock`** cubren la instalación completa.

### 4.2 Creación del entorno e instalación de las dependencias

Clone el repositorio:

```bash
git clone https://github.com/seba39399/Proyecto_Deteccion_Neumonia.git  
cd Proyecto_Deteccion_Neumonia  
```

Cree el entorno virtual e instale dependencias con **UV**:

Desde la carpeta raíz del proyecto (donde está `pyproject.toml`):

```bash
uv venv
uv sync
```

**Activar el entorno (opcional):**

- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:
```bash
source .venv/bin/activate
```
Se recomienda ejecutar `uv sync` para instalar exactamente las versiones registradas en `uv.lock`.

**Nota sobre el entorno de desarrollo:**

El proyecto fue desarrollado y probado utilizando **Python 3.10.19**. Aunque en `pyproject.toml` se especifica `requires-python = ">=3.10"`, la versión empleada durante la implementación fue 3.10.19, garantizando compatibilidad con TensorFlow y el resto de dependencias.

Dependencias principales (versiones mínimas declaradas):

```
- customtkinter >= 5.2.2  
- flask >= 3.1.2  
- fpdf >= 1.7.2  
- img2pdf >= 0.6.3  
- matplotlib >= 3.10.8  
- opencv-python >= 4.13.0.92  
- pandas >= 2.3.3  
- pillow >= 12.1.0  
- pyautogui >= 0.9.54  
- pydicom >= 3.0.1  
- pytest >= 9.0.2  
- python-xlib >= 0.33  
- tensorflow >= 2.20.0  
- tkcap >= 0.0.4  
```

## 5. Modelo (.h5)

El proyecto requiere un modelo entrenado en formato Keras `.h5`. Para dicho fin, coloque el archivo del modelo en la carpeta raíz del proyecto (junto a `main.py`).

El modelo debe ubicarse en la carpeta raíz:

`conv_MLP_84.h5`  

Si cambia el nombre, modifique la ruta en:

`src_directory/load_model.py`

### Arquitectura del modelo

CNN inspirada en la propuesta de Pasa et al., para análisis de radiografías:

- 5 bloques convolucionales con 16, 32, 48, 64 y 80 filtros (3×3).  
- Conexiones tipo skip para evitar desvanecimiento del gradiente.  
- MaxPooling por bloque y Average Pooling final.  
- 3 capas Dense (1024, 1024 y 3 neuronas).  
- Dropout del 20% para regularización.

## 6. Ejecución de la aplicación

Ejecute desde la raíz:

```bash
uv run python main.py
```

## 7. Uso de la interfaz (GUI)

1. Ingresar la cédula del paciente (Opcional).  
2. Presionar **Cargar Imagen** y seleccionar un archivo `.dcm` o `.jpg/.png`.  
3. Presionar **Predecir** para obtener:
   - clase (bacteriana / viral / normal)
   - probabilidad (%)
   - heatmap Grad‑CAM
4. Presionar **Guardar / PDF** para registrar resultados.
5. Presionar **Borrar** para cargar una nueva imagen.

---

## 8. Módulos clave (breve)

- **`read_img.py`**: lee DICOM o imágenes estándar y entrega un `np.ndarray` para el pipeline y un objeto para visualización.  
- **`preprocess_img.py`**: aplica resize, grises, CLAHE y normalización, y forma el tensor 4D **(1, 512, 512, 1)**.  
- **`load_model.py`**: carga el modelo `.h5` (idealmente una sola vez).  
- **`grad_cam.py`**: calcula el heatmap Grad‑CAM y lo superpone a la imagen original.  
- **`integrator.py`**: coordina el flujo completo y retorna lo necesario para la interfaz: **clase, probabilidad y heatmap**.

---

## 9. Grad‑CAM: capa de interés

El Grad‑CAM suele requerir una capa convolucional específica, por ejemplo:

```python
model.get_layer("conv10_thisone")
```

Si el modelo `.h5` no contiene esa capa con ese nombre, Grad‑CAM puede fallar. En ese caso, se debe:
- ajustar el nombre de la capa en el código, o
- modificar Grad‑CAM para detectar automáticamente la última capa `Conv2D`.

---

## 10. Pruebas

El proyecto incluye pruebas utilizando el framework de Pytest, para ejecutarlas, se puede utilizar el siguiente comando:

```bash
uv run pytest
```

---

## 10. Diagrama UML

A continuación, se presenta un diagrama UML, el cual presenta la arquitectura modular del sistema de detección de neumonía, mostrando la organización de los módulos, sus responsabilidades y las dependencias entre componentes. Se ilustra el flujo principal desde la interfaz gráfica hasta el procesamiento, la predicción y la generación de la explicación mediante Grad-CAM.

<img width="1459" height="1007" alt="image" src="https://github.com/user-attachments/assets/a204f3bf-7609-42a1-b060-99589605a309" />

---
## 11. Uso académico

Este proyecto es de uso educativo y no reemplaza un diagnóstico médico profesional.

## 12. Licencia.

Este proyecto está licenciado bajo la licencia MIT: consulte el archivo de LICENCIA para obtener más detalles.
