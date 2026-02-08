import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tkinter import *
from tkinter import ttk, font, filedialog, messagebox
from PIL import ImageTk, Image
import csv
import numpy as np
import tensorflow as tf

# Importación de tus nuevos módulos
from src_directory.load_model import model_fun
from src_directory.read_img import read_dicom_file, read_jpg_file
from src_directory.integrator import predict

tf.config.run_functions_eagerly(True)

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        
        # Carga del modelo al iniciar
        self.model = model_fun()

        fonti = font.Font(weight="bold")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # INTERFAZ ORIGINAL INTACTA
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(self.root, text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", font=fonti)
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.ID = StringVar()
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        self.button1 = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button2 = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button6 = ttk.Button(self.root, text="Guardar", command=self.save_results_csv)

        # POSICIONES ORIGINALES
        self.lab1.place(x=110, y=65); self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350); self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25); self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460); self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460); self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350); self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90); self.text_img2.place(x=500, y=90)

        self.text1.focus_set()
        self.array = None
        self.root.mainloop()

    def load_img_file(self):
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
        self.text2.delete(1.0, END); self.text3.delete(1.0, END); self.text_img2.delete(1.0, END)
        self.label, self.proba, heatmap_array = predict(self.array, self.model)
        
        self.img2 = Image.fromarray(heatmap_array).resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}%".format(self.proba))
        print("Predicción exitosa: OK")

    def save_results_csv(self):
        with open("historial.csv", "a", newline='') as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow([self.text1.get(), self.label, "{:.2f}%".format(self.proba)])
        messagebox.showinfo("Guardar", "Los datos se guardaron con éxito.")

    def delete(self):
        if messagebox.askokcancel("Confirmación", "Se borrarán todos los datos."):
            self.text1.delete(0, END); self.text2.delete(1.0, END); self.text3.delete(1.0, END)
            self.text_img1.delete(1.0, END); self.text_img2.delete(1.0, END)
            self.array = None

if __name__ == "__main__":
    App()