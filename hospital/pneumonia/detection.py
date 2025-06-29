import os
import numpy as np
from PIL import Image
import base64
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import pymongo
import sys

# Ensure UTF-8 encoding for Python output
sys.stdout.reconfigure(encoding='utf-8')

# Create Flask app for Pneumonia Detection
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MongoDB configuration
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["hospital_database"]
collection = db["pneumonia_scans"]

# Set the absolute path for the model
model_dir = os.path.abspath("C:/Users/rohit/OneDrive/Documents/3rd year/Data Modelling NOSQL/hospital management system/hospital/pneumonia")
model_path = os.path.join(model_dir, 'pneumonia_final_model.h5')

# Load the trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} was not found.")
model = load_model(model_path)

def load_accuracy():
    accuracy_file = os.path.join(model_dir, 'pneumonia_accuracy.txt')
    with open(accuracy_file, 'r', encoding='utf-8') as f:
        accuracy = float(f.read().strip())
    return accuracy

accuracy = load_accuracy()

def predict(image_path):
    image = Image.open(image_path).resize((224, 224))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia Detected"

# Convert image to Base64
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
            image_base64 = convert_image_to_base64(file_path)
            return render_template("pneumonia_result.html", result=result, accuracy=accuracy, image_base64=image_base64)
    return render_template("pneumonia_upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ensure UTF-8 encoding for all responses
@app.after_request
def set_response_headers(response):
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response

# Configure Flask to handle JSON responses in UTF-8
app.config['JSON_AS_ASCII'] = False

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Specify a different port for pneumonia detection service
import os
import numpy as np
from PIL import Image
import io
import uuid
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify, send_from_directory
import sys

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['JSON_AS_ASCII'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Alternative path from your original code:
MODEL_PATH = "C:/Users/rohit/OneDrive/Documents/3rd year/Data Modelling NOSQL/hospital management system/hospital/pneumonia/pneumonia_final_model.h5"

# Load the trained model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"The model file {MODEL_PATH} was not found.")
    
    model = load_model(MODEL_PATH)
    print(f"✅ Pneumonia model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading pneumonia model: {str(e)}")
    model = None

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_chest_xray(img_array):
    """
    Validates if an image is a chest X-ray using multiple techniques.
    Returns True if the image appears to be a chest X-ray, False otherwise.
    """
    try:
        # Work with the first image if batched
        single_img = img_array[0]
        
        # 1. Grayscale analysis - X-rays are essentially grayscale
        if single_img.shape[-1] == 3:  # If RGB
            # Calculate mean of each channel
            r_mean, g_mean, b_mean = np.mean(single_img[:,:,0]), np.mean(single_img[:,:,1]), np.mean(single_img[:,:,2])
            
            # Calculate standard deviation between channel means
            channel_means = [r_mean, g_mean, b_mean]
            channel_std = np.std(channel_means)
            
            # Calculate maximum difference between any two channels
            max_diff = max([abs(channel_means[i] - channel_means[j]) 
                           for i in range(3) for j in range(i+1, 3)])
            
            print(f"📊 Channel analysis: R={r_mean:.4f}, G={g_mean:.4f}, B={b_mean:.4f}")
            print(f"📊 Channel std: {channel_std:.4f}, Max diff: {max_diff:.4f}")
            
            # True X-rays converted to RGB usually have very similar channel values
            if channel_std > 0.05 or max_diff > 0.1:
                print("❌ Failed grayscale check: High color variation between channels")
                return False
        
        # 2. Convert to grayscale for further analysis
        if single_img.shape[-1] == 3:
            gray_img = np.mean(single_img, axis=-1)
        else:
            gray_img = single_img.squeeze()
        
        # 3. Brightness and contrast analysis
        img_mean = np.mean(gray_img)
        img_std = np.std(gray_img)
        
        print(f"📊 Image brightness (mean): {img_mean:.4f}")
        print(f"📊 Image contrast (std): {img_std:.4f}")
        
        # X-rays typically have moderate brightness and good contrast
        if img_mean < 0.2 or img_mean > 0.8:
            print("❌ Failed brightness check: Image too dark or too bright")
            return False
            
        if img_std < 0.1:
            print("❌ Failed contrast check: Image has insufficient contrast")
            return False
        
        # 4. Edge detection to look for anatomical structures
        gx = np.abs(np.gradient(gray_img, axis=0))
        gy = np.abs(np.gradient(gray_img, axis=1))
        edges = np.sqrt(gx**2 + gy**2)
        
        # Analyze edge statistics
        edge_threshold = 0.05
        edge_percentage = np.mean(edges > edge_threshold)
        
        print(f"📊 Edge percentage: {edge_percentage:.4f}")
        
        # X-rays have a moderate amount of edges from anatomical structures
        if edge_percentage < 0.01 or edge_percentage > 0.3:
            print("❌ Failed edge check: Unusual edge distribution")
            return False
        
        # 5. Spatial consistency check
        h, w = gray_img.shape
        center_region = gray_img[h//4:3*h//4, w//4:3*w//4]
        edge_regions = np.concatenate([
            gray_img[:h//4, :].flatten(),  # top
            gray_img[3*h//4:, :].flatten(),  # bottom
            gray_img[h//4:3*h//4, :w//4].flatten(),  # left
            gray_img[h//4:3*h//4, 3*w//4:].flatten()  # right
        ])
        
        center_mean = np.mean(center_region)
        edge_mean = np.mean(edge_regions)
        
        print(f"📊 Center region mean: {center_mean:.4f}, Edge region mean: {edge_mean:.4f}")
        
        # In most X-rays, the difference shouldn't be extreme
        if abs(center_mean - edge_mean) > 0.5:
            print("❌ Failed spatial consistency check: Unusual center-edge difference")
            return False
        
        # If all checks pass, it's likely a chest X-ray
        print("✅ Image passed all X-ray validation checks")
        return True
        
    except Exception as e:
        print(f"⚠️ Error in X-ray validation: {str(e)}")
        # Fall back to a basic grayscale check
        try:
            if single_img.shape[-1] == 3:
                r, g, b = single_img[:,:,0], single_img[:,:,1], single_img[:,:,2]
                diff_rg = np.mean(np.abs(r - g))
                diff_rb = np.mean(np.abs(r - b))
                diff_gb = np.mean(np.abs(g - b))
                
                avg_diff = (diff_rg + diff_rb + diff_gb) / 3
                print(f"📊 Average channel difference (fallback): {avg_diff:.4f}")
                
                return avg_diff < 0.05
            return True
        except Exception as e:
            print(f"⚠️ Error in fallback validation: {str(e)}")
            return True

def preprocess_image(img):
    """Preprocess image for model input"""
    try:
        # Resize to the input size expected by your model
        target_size = (224, 224)  # Adjust based on your model
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        print(f"📊 Preprocessed image shape: {img_array.shape}")
        print(f"📊 Value range: {np.min(img_array):.4f} to {np.max(img_array):.4f}")
        
        return img_array
    except Exception as e:
        print(f"❌ Error in image preprocessing: {str(e)}")
        raise

def predict_pneumonia(img_array):
    """Make pneumonia prediction using the loaded model"""
    try:
        if model is None:
            raise Exception("Model not loaded. Please check the model path.")
        
        prediction = model.predict(img_array, verbose=0)
        print(f"📊 Raw prediction: {prediction}")
        
        # Extract probability (adjust based on your model's output format)
        probability = float(prediction[0][0])
        is_pneumonia = probability > 0.5
        
        print(f"📊 Probability: {probability:.4f}")
        print(f"📊 Prediction: {'Pneumonia Detected' if is_pneumonia else 'Normal'}")
        
        return is_pneumonia, probability
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        raise

@app.route("/")
def index():
    """Serve the main index page"""
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the frontend"""
    try:
        print("\n🔍 Starting prediction process...")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Please upload PNG, JPG, or JPEG images.'})
        
        print(f"📁 Processing file: {file.filename}")
        
        # Create unique filename for temporary storage
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Read and process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Save the file temporarily (optional, for debugging)
        img.save(file_path)
        print(f"💾 Image saved to: {file_path}")
        
        # Preprocess image
        img_array = preprocess_image(img)
        
        # Validate if it's a chest X-ray
        print("🔍 Validating if image is a chest X-ray...")
        is_valid_xray = is_chest_xray(img_array)
        
        if not is_valid_xray:
            # Clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'error': 'The uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.'
            })
        
        # Make prediction
        print("🧠 Making pneumonia prediction...")
        is_pneumonia, probability = predict_pneumonia(img_array)
        
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Return results in the format expected by your frontend
        result = {
            'is_pneumonia': is_pneumonia,
            'probability': probability,
            'confidence': probability if is_pneumonia else (1 - probability),
            'message': 'Pneumonia detected' if is_pneumonia else 'Normal chest X-ray'
        }
        
        print(f"✅ Prediction completed: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Processing error: {str(e)}"})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files (if needed)"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

# Ensure UTF-8 encoding for all responses
@app.after_request
def set_response_headers(response):
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response

if __name__ == "__main__":
    print("🚀 Starting Simple Pneumonia Detection System...")
    print(f"📊 Model Status: {'✅ Loaded' if model is not None else '❌ Not Loaded'}")
    print(f"📁 Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🌐 Server starting on http://localhost:5000")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)