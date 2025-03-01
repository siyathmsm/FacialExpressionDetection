from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('FER_model.h5')

# Define image size and class names
image_size = (48, 48)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the mapping for interested and not-interest
interested = ['happy', 'surprise', 'neutral']
not_interest = ['angry', 'sad', 'fear', 'disgust']

def process_image(image_data):
    # Decode the base64 image data
    img_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size)
    img = img.reshape((1, 48, 48, 1)) / 255.0
    
    # Predict the facial expression
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    class_name = class_names[class_idx]
    
    return class_name

@app.route('/student_interface')
def student_interface():
    return render_template('student_interface.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    data = request.json
    image_data = data['image']
    expression = process_image(image_data)
    
    # Save the image to the appropriate directory
    category = 'interested' if expression in interested else 'not-interest'
    directory = os.path.join('sorted_images', category, expression)
    os.makedirs(directory, exist_ok=True)
    img_count = len(os.listdir(directory))
    img_path = os.path.join(directory, f'image_{img_count + 1}.png')
    
    img_data = base64.b64decode(image_data.split(',')[1])
    with open(img_path, 'wb') as f:
        f.write(img_data)
    
    return jsonify({'status': 'success', 'expression': expression})

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/conductSession')
def conductSession():
    return render_template('conduct_session.html')

@app.route('/analytics_data')
def analytics_data():
    interested_count = sum([len(files) for r, d, files in os.walk('sorted_images/interested')])
    not_interest_count = sum([len(files) for r, d, files in os.walk('sorted_images/not-interest')])
    return jsonify({'interested_count': interested_count, 'not_interest_count': not_interest_count})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
