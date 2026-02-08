from flask import Flask, render_template, request, send_file
from src_directory_othermodules.ai_logic import AIService
from src_directory_othermodules.data_manager import DataManager
from fpdf import FPDF
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ai = AIService('conv_MLP_84.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        cedula = request.form.get('cedula')
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesamiento
            img_array, _ = DataManager.read_file(filepath)
            label, proba, heatmap_rgb = ai.predict(img_array)
            
            # Guardar Heatmap
            heat_name = "heat_" + filename
            heat_path = os.path.join(app.config['UPLOAD_FOLDER'], heat_name)
            cv2.imwrite(heat_path, cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
            
            return render_template('index.html', 
                                   original=filename, heatmap=heat_name, 
                                   label=label, proba=f"{proba:.2f}%", cedula=cedula)
            
    return render_template('index.html')

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    # Recuperar datos del formulario oculto
    cedula = request.form.get('cedula')
    label = request.form.get('label')
    proba = request.form.get('proba')
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "INFORME DE DIAGNÓSTICO MÉDICO POR IA", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Cédula del Paciente: {cedula}", ln=True)
    pdf.cell(200, 10, f"Resultado del Análisis: {label}", ln=True)
    pdf.cell(200, 10, f"Confidencia: {proba}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, "Firma del Software: UAO Health AI System", ln=True)
    
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"reporte_{cedula}.pdf")
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)