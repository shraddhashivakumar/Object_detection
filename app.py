# app.py
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained model (e.g., a MobileNet or VGG16 model)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def process_image(image_path):
    # Load the image
    img = Image.open(image_path)
    img_resized = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict using the model
    predictions = model.predict(img_array)

    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    # Get the object with the highest confidence score
    best_prediction = max(decoded_predictions, key=lambda x: x[2])  # x[2] is the confidence score
    
    # Open the original image using OpenCV for bounding box drawing
    img_cv2 = cv2.imread(image_path)

    # Draw bounding box and label for the best prediction
    start_point = (50, 50)
    end_point = (200, 200)  # Placeholder coordinates; replace with actual model output if needed
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(img_cv2, start_point, end_point, color, thickness)
    
    label = f"{best_prediction[1]}: {best_prediction[2] * 100:.2f}%"
    cv2.putText(img_cv2, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(image_path))
    cv2.imwrite(output_image_path, img_cv2)

    return best_prediction, output_image_path


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image and get the best prediction and image with bounding box
        best_prediction, output_image_path = process_image(filepath)

        # Pass the best prediction and image to the template
        return render_template('results.html', best_prediction=best_prediction, image_name=os.path.basename(output_image_path))
    
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
