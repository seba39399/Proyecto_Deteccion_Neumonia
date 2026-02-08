#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo muestra errores graves
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Apaga el mensaje de optimización

from tkinter import *
from tkinter import ttk, font, filedialog, Entry
from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time

import pydicom as dicom
import cv2

import tensorflow as tf
from keras import backend as K

# Forzamos que TensorFlow trabaje en modo real (Eager) para evitar el error de SymbolicTensor
tf.config.run_functions_eagerly(True)

#funcion creada que no existia
def model_fun():
    model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
    return model


def grad_cam(array):
    img = preprocess(array)
    model = model_fun()
    
    # 1. Definimos la capa convolucional de interés
    last_conv_layer = model.get_layer("conv10_thisone") 
    
    # 2. Creamos un modelo intermedio que nos dé la salida de esa capa y la predicción final
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    # 3. Calculamos los gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        # Si predictions es una lista, tomamos el primer elemento
        if isinstance(predictions, list): predictions = predictions[0]
        
        argmax = np.argmax(predictions[0])
        loss = predictions[:, argmax]

    # Gradientes de la pérdida con respecto a la salida de la capa conv
    grads = tape.gradient(loss, conv_outputs)
    
    # Pesos (Promedio global de los gradientes)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 4. Multiplicamos la salida de la capa por los pesos
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Post-procesamiento del Heatmap
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10) # ReLU y Normalización
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 6. Superponer el mapa de calor en la imagen original
    img2 = cv2.resize(array, (512, 512))
    superimposed_img = cv2.addWeighted(img2, 0.6, heatmap, 0.4, 0)
    
    return superimposed_img[:, :, ::-1] # Convertimos BGR a RGB para Tkinter


def predict(array):
    # 1. Pre-procesar imagen
    batch_array_img = preprocess(array)
    
    # 2. Cargar modelo y predecir
    model = model_fun()
    res = model.predict(batch_array_img, verbose=0)
    
    # Si res es una lista, tomamos el primer elemento
    if isinstance(res, list): res = res[0]
    
    prediction = np.argmax(res[0])
    proba = np.max(res[0]) * 100
    
    label_dict = {0: "bacteriana", 1: "normal", 2: "viral"}
    label = label_dict.get(prediction, "Desconocido")
        
    # 3. Generar Grad-CAM
    heatmap = grad_cam(array)
    
    return (label, proba, heatmap)


def read_dicom_file(path):
    img = dicom.dcmread(path) #correccion codigo, se cambio dicom.read_file por dicom.dcmread
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            self.array, img2show = read_dicom_file(filepath)
            self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS) #correccion codigo , se cambio Image.ANTIALIAS por Image.Resampling.LANCZOS
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        # 1. Limpiamos los cuadros de texto para que no se amontonen los resultados previos
        self.text2.delete(1.0, END)
        self.text3.delete(1.0, END)
        self.text_img2.delete(1.0, END)

        # 2. Ejecutamos la predicción
        self.label, self.proba, self.heatmap = predict(self.array)

        # 3. Procesamos el Heatmap (Imagen con el mapa de calor)
        self.img2 = Image.fromarray(self.heatmap)
        
        # CORRECCIÓN: Usamos Resampling.LANCZOS en lugar de ANTIALIAS
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS)
        
        self.img2 = ImageTk.PhotoImage(self.img2)
        
        # 4. Mostramos los resultados en la interfaz
        print("Predicción exitosa: OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(self.img1, "end")
            self.text_img2.delete(self.img2, "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return my_app


if __name__ == "__main__":
    main()
