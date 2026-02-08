import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
from PIL import ImageTk, Image
from src_directory_othermodules.data_manager import DataManager

class App:
    def __init__(self, root, ai_service):
        self.root = root
        self.ai = ai_service  # Servicio de IA inyectado desde main.py
        
        # Variables de estado
        self.current_array = None
        self.report_id = 0
        self.last_label = ""
        self.last_proba = 0.0

        # Configuración de la ventana
        self.root.title("Software de Apoyo Médico - Detección de Neumonía")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        self._setup_ui()

    def _setup_ui(self):
        """Define la estructura visual de la aplicación."""
        bold_font = font.Font(weight="bold")

        # Título principal
        ttk.Label(self.root, text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", 
                  font=bold_font).place(x=122, y=25)

        # Etiquetas de imágenes
        ttk.Label(self.root, text="Imagen Radiográfica", font=bold_font).place(x=110, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=bold_font).place(x=545, y=65)

        # Contenedores de imágenes (Text widgets usados como marcos)
        self.canvas_orig = tk.Text(self.root, width=31, height=15)
        self.canvas_heat = tk.Text(self.root, width=31, height=15)
        self.canvas_orig.place(x=65, y=90)
        self.canvas_heat.place(x=500, y=90)

        # Panel de Datos del Paciente
        ttk.Label(self.root, text="Cédula Paciente:", font=bold_font).place(x=65, y=350)
        self.entry_id = ttk.Entry(self.root, width=15)
        self.entry_id.place(x=200, y=350)
        self.entry_id.focus_set()

        # Panel de Resultados
        ttk.Label(self.root, text="Resultado:", font=bold_font).place(x=500, y=350)
        ttk.Label(self.root, text="Probabilidad:", font=bold_font).place(x=500, y=400)
        
        self.display_res = tk.Text(self.root, width=12, height=1)
        self.display_prob = tk.Text(self.root, width=12, height=1)
        self.display_res.place(x=610, y=350)
        self.display_prob.place(x=610, y=400)

        # Botonera
        self.btn_load = ttk.Button(self.root, text="Cargar Imagen", command=self.handle_load)
        self.btn_predict = ttk.Button(self.root, text="Predecir", state="disabled", command=self.handle_predict)
        self.btn_save = ttk.Button(self.root, text="Guardar", command=self.handle_save)
        self.btn_clear = ttk.Button(self.root, text="Borrar", command=self.handle_clear)
        
        self.btn_load.place(x=70, y=460)
        self.btn_predict.place(x=220, y=460)
        self.btn_save.place(x=370, y=460)
        self.btn_clear.place(x=670, y=460)

    # --- MANEJADORES DE EVENTOS (HANDLERS) ---

    def handle_load(self):
        """Gestiona la carga de la imagen a través del DataManager."""
        path = filedialog.askopenfilename(
            title="Seleccionar Radiografía",
            filetypes=[("Archivos Médicos", "*.dcm *.jpg *.jpeg *.png")]
        )
        if path:
            # Obtenemos el array para la IA y la imagen para mostrar
            self.current_array, img_obj = DataManager.read_file(path)
            
            # Ajustar para mostrar en la interfaz
            img_resized = img_obj.resize((250, 250), Image.Resampling.LANCZOS)
            self.tk_orig = ImageTk.PhotoImage(img_resized)
            
            self.canvas_orig.delete(1.0, tk.END)
            self.canvas_orig.image_create(tk.END, image=self.tk_orig)
            self.btn_predict["state"] = "normal"

    def handle_predict(self):
        """Llama al servicio de IA y actualiza la vista con el resultado."""
        if self.current_array is None: return

        # Limpiar resultados anteriores
        self.canvas_heat.delete(1.0, tk.END)
        self.display_res.delete(1.0, tk.END)
        self.display_prob.delete(1.0, tk.END)

        # Ejecutar IA
        label, proba, heatmap_rgb = self.ai.predict(self.current_array)
        
        # Guardar en estado para persistencia
        self.last_label = label
        self.last_proba = proba

        # Mostrar Heatmap
        heat_img = Image.fromarray(heatmap_rgb).resize((250, 250), Image.Resampling.LANCZOS)
        self.tk_heat = ImageTk.PhotoImage(heat_img)
        
        self.canvas_heat.image_create(tk.END, image=self.tk_heat)
        self.display_res.insert(tk.END, label)
        self.display_prob.insert(tk.END, f"{proba:.2f}%")
        
        print("Predicción exitosa: OK")

    def handle_save(self):
        """Guarda la información en el historial CSV."""
        cedula = self.entry_id.get()
        if not cedula or not self.last_label:
            messagebox.showwarning("Advertencia", "Debe ingresar cédula y realizar predicción.")
            return

        data = [cedula, self.last_label, f"{self.last_proba:.2f}%"]
        DataManager.save_to_csv("historial.csv", data)
        messagebox.showinfo("Éxito", "Los datos del paciente se han guardado correctamente.")

    def handle_clear(self):
        """Limpia todos los campos de la interfaz."""
        if messagebox.askokcancel("Confirmación", "¿Desea borrar todos los datos actuales?"):
            self.entry_id.delete(0, tk.END)
            self.display_res.delete(1.0, tk.END)
            self.display_prob.delete(1.0, tk.END)
            self.canvas_orig.delete(1.0, tk.END)
            self.canvas_heat.delete(1.0, tk.END)
            self.btn_predict["state"] = "disabled"
            self.current_array = None