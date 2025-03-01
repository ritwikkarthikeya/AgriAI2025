from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import torch
from ultralytics import YOLO
import time

app = Flask(__name__)

# ---------------- Leaf Detection Model ----------------
leaf_model = tf.keras.models.load_model("model/Vgg.h5")
leaf_classes = ['Algal Leaf', 'Anthracnose', 'Bird Eye Spot', 'Healthy', 'Red Leaf Spot']

# ---------------- YOLOv8 Animal Detection Model ----------------
yolo_model = YOLO('yolov8n.pt')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['last_detection'] = []

# Check File Extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess Image for Leaf Model
def preprocess_leaf_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File format not supported! Use JPG, JPEG, or PNG"}), 400

    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ### -------- Step 1: YOLO Detection -------- ###
    results = yolo_model(filepath)
    detected_classes = set()

    # Store all detected animals
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = yolo_model.names.get(cls, f"Unknown_{cls}")
            detected_classes.add(label)

    # If YOLO detects any object -> Animal Detected
    if detected_classes:
        app.config['last_detection'] = list(detected_classes)
        return jsonify({
            "Detected Object": "Animal",
            "Detections": list(detected_classes)
        })

    ### -------- Step 2: If No Animal Detected, Use Leaf Model -------- ###
    image = preprocess_leaf_image(filepath)
    prediction = leaf_model.predict(image)

    if prediction.size == 0:
        return jsonify({"error": "No Prediction Made, Please Check Image Quality!"}), 400

    predicted_class = leaf_classes[np.argmax(prediction[0])]
    confidence = np.max(prediction)

    return jsonify({
        "Detected Object": "Leaf",
        "Prediction": predicted_class,
        "Confidence": str(round(confidence * 100, 2)) + "%"
    })


@app.route('/status', methods=['GET'])
def status():
    last_detection = app.config['last_detection']
    return jsonify({"buzzer": "ON" if last_detection else "OFF", "detections": last_detection})


if __name__ == '__main__':
    app.run(debug=True)
