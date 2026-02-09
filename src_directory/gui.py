import csv
import os
from datetime import datetime
from tkinter import *
from tkinter import ttk, font, filedialog, messagebox
from PIL import ImageTk, Image
from fpdf import FPDF 

# Importación de módulos lógicos
from src_directory.load_model import model_fun
from src_directory.read_img import read_dicom_file, read_jpg_file
from src_directory.integrator import predict

class App:
    """
    Clase de interfaz gráfica. 
    Diseñada para ser instanciada desde un módulo externo (main.py).
    """
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)
        
        self.ID = StringVar()
        self.array = None
        self.label = "" 
        self.proba = 0.0
        
        fonti = font.Font(weight="bold")
        self._setup_widgets(fonti)
        
        # Carga diferida del modelo para no bloquear el dibujado de la ventana
        self.root.after(100, self._load_model_async)
        self.root.mainloop()

    def _setup_widgets(self, fonti):
        """Define y posiciona todos los elementos visuales."""
        # Etiquetas
        ttk.Label(self.root, text="Imagen Radiográfica", font=fonti).place(x=110, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=fonti).place(x=545, y=65)
        ttk.Label(self.root, text="Resultado:", font=fonti).place(x=500, y=350)
        ttk.Label(self.root, text="Cédula Paciente:", font=fonti).place(x=65, y=350)
        ttk.Label(self.root, text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO", font=fonti).place(x=122, y=25)
        ttk.Label(self.root, text="Probabilidad:", font=fonti).place(x=500, y=400)

        # Entradas y Visores de imagen
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        self.text1.place(x=200, y=350)
        
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img1.place(x=65, y=90)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text_img2.place(x=500, y=90)
        
        # Cajas de resultado (Solo Lectura)
        self.text2 = Text(self.root, state=DISABLED)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3 = Text(self.root, state=DISABLED)
        self.text3.place(x=610, y=400, width=90, height=30)

        # Botones de acción
        self.button1 = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button1.place(x=220, y=460)
        
        ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file).place(x=70, y=460)
        ttk.Button(self.root, text="Borrar", command=self.delete).place(x=670, y=460)
        ttk.Button(self.root, text="Guardar y PDF", command=self.save_results_full).place(x=370, y=460)
        
        self.text1.focus_set()

    def _load_model_async(self):
        """Carga el motor de IA."""
        try:
            self.model = model_fun()
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al cargar modelo: {e}")

    def load_img_file(self):
        """Explorador de archivos para imágenes médicas."""
        filepath = filedialog.askopenfilename(filetypes=(("DICOM", "*.dcm"), ("Images", "*.jpg *.jpeg *.png")))
        if filepath:
            if filepath.lower().endswith('.dcm'):
                self.array, img2show = read_dicom_file(filepath)
            else:
                self.array, img2show = read_jpg_file(filepath)
            
            self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.delete(1.0, END)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        """Controla el flujo de predicción y actualización de la UI."""
        self.text2.config(state=NORMAL); self.text3.config(state=NORMAL)
        self.text2.delete(1.0, END); self.text3.delete(1.0, END)
        self.text_img2.delete(1.0, END)
        
        self.label, self.proba, heatmap_array = predict(self.array, self.model)
        
        self.img2 = Image.fromarray(heatmap_array).resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}%".format(self.proba))
        
        self.text2.config(state=DISABLED); self.text3.config(state=DISABLED)

    def save_results_full(self):
        """Exportación de datos a CSV y PDF."""
        cedula = self.text1.get()
        if not cedula or not self.label:
            messagebox.showwarning("Atención", "Se requiere cédula y predicción previa.")
            return

        # CSV
        try:
            with open("historial.csv", "a", newline='', encoding='utf-8') as f:
                csv.writer(f, delimiter="-").writerow([cedula, self.label, f"{self.proba:.2f}%", datetime.now()])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo escribir en CSV: {e}")

        # PDF
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "INFORME MEDICO DE APOYO AL DIAGNOSTICO", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 10, f"Paciente ID: {cedula}", ln=True)
            pdf.cell(0, 10, f"Fecha de emision: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, f"HALLAZGO: {self.label.upper()}", ln=True)
            pdf.cell(0, 10, f"CONFIANZA DEL MODELO: {self.proba:.2f}%", ln=True)
            
            pdf.output(f"Reporte_{cedula}.pdf")
            messagebox.showinfo("Proceso Exitoso", f"Reporte PDF y CSV generados para ID: {cedula}")
        except Exception as e:
            messagebox.showerror("Error PDF", str(e))

    def delete(self):
        """Resetea la interfaz."""
        if messagebox.askokcancel("Confirmar", "¿Desea limpiar todos los campos?"):
            self.text2.config(state=NORMAL); self.text3.config(state=NORMAL)
            self.text1.delete(0, END); self.text2.delete(1.0, END); self.text3.delete(1.0, END)
            self.text_img1.delete(1.0, END); self.text_img2.delete(1.0, END)
            self.text2.config(state=DISABLED); self.text3.config(state=DISABLED)
            self.array = None