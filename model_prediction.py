# model_prediction.py
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('facial_expression_model.h5')

def predict_image(model, image):
    predictions = model.predict(image)
    return predictions
