from flask import Flask, request, render_template, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = load_model('FER_model.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define image size and class names
image_size = (48, 48)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load in color
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    # If no face is detected, return None
    if len(faces) == 0:
        return None

    # Take the first detected face (you can modify this to handle multiple faces)
    (x, y, w, h) = faces[0]
    
    # Crop the image to the face area
    face_img = gray_img[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, image_size)  # Resize to model input size
    face_img = face_img.reshape((1, 48, 48, 1)) / 255.0  # Normalize and reshape

    # Predict the facial expression
    predictions = model.predict(face_img)
    class_idx = np.argmax(predictions)
    class_name = class_names[class_idx]

    return class_name

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('upload.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error='No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        expression = process_image(file_path)
        
        if expression is None:
            # Remove the file after processing
            # os.remove(file_path)
            # Pass the image filename along with the error message
            return render_template('result1.html', expression="No face detected", image_filename=filename, error_message="No face detected. Please try another image.")
        
        # If face is detected, render result with expression and image
        return render_template('result1.html', expression=expression, image_filename=filename)
    else:
        return render_template('upload.html', error='Invalid file type')
        
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
