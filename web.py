import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model_path = 'Classification_Model.h5'
loaded_model = tf.keras.models.load_model(model_path)

# Define the categories
CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (130, 130), interpolation=cv2.INTER_LINEAR)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Function to predict brain tumor type from the uploaded image
def predict_tumor_type(image):
    preprocessed_image = preprocess_image(image)
    predictions = loaded_model.predict(preprocessed_image)
    label = CATEGORIES[np.argmax(predictions)]
    return label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Check if the file exists and has an allowed extension
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read and preprocess the uploaded image
            image = Image.open(file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            label = predict_tumor_type(image)

            # Render the result template with the prediction
            return render_template('index.html', prediction=label, filename=file.filename)

    # Render the index template
    return render_template('index.html')

if __name__ == '__main__':
    app.static_folder = 'static'
    app.run(debug=True)
