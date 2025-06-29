import os
import numpy as np
from PIL import Image
import base64
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import pymongo

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MongoDB configuration
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["hospital_database"]
collection = db["mri_scans"]

# Set the absolute path for the model
model_dir = os.path.abspath("C:/Users/rohit/OneDrive/Documents/3rd year/Data Modelling NOSQL/hospital management system/hospital/braintumor")
model_path = os.path.join(model_dir, 'final_model.h5')

# Load the trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} was not found. Please ensure it is in the correct directory.")
model = load_model(model_path)

def load_accuracy():
    accuracy_file = os.path.join(model_dir, 'accuracy.txt')
    with open(accuracy_file, 'r', encoding='utf-8') as f:
        accuracy = float(f.read().strip())
    return accuracy

accuracy = load_accuracy()

def predict(image_path):
    image = Image.open(image_path).resize((128, 128))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

# Function to convert image to Base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        username = request.form.get('username', 'Unknown User')
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            scan_record = {
                "username": username,
                "filename": filename,
                "result": result
            }
            collection.insert_one(scan_record)
            image_base64 = convert_image_to_base64(file_path)  # Convert to base64
            return render_template("braintumor_result.html", result=result, accuracy=accuracy, image_base64=image_base64)
    return render_template("braintumor_upload.html")
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ensure UTF-8 encoding for all responses
@app.after_request
def set_response_headers(response):
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response

if __name__ == "__main__":
    import sys
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach()) # Ensure UTF-8 encoding for console output
    app.run(debug=True)
