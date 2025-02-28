from flask import Flask, render_template, request, jsonify
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 Model (pretrained for now)
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Perform Detection
    results = model(filepath)
    detected_classes = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            detected_classes.append(label)

    detected = list(set(detected_classes))
    return render_template('result.html', image=file.filename, detections=detected)

if __name__ == '__main__':
    app.run(debug=True)
