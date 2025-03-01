# image_capture.py
import cv2
import numpy as np
import time

def capture_image():
    cap = None
    for index in range(5):  # Try up to 5 different camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            break
    else:
        print("No camera found.")
        return None

    # Set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Adjust camera settings (if supported)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Example value
    cap.set(cv2.CAP_PROP_CONTRAST, 50)     # Example value
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)     # Example value for lower exposure

    # Allow the camera to adjust
    time.sleep(0.5)

    ret, frame = cap.read()
    if ret:
        # Optional: Apply some pre-processing
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # Save the captured image (if needed)
        cv2.imwrite("captured_image2.jpg", frame)
        print("Image captured successfully!")
        cap.release()
        return frame
    else:
        print("Error capturing image.")
        cap.release()
        return None

def preprocess_image(image, target_size=(224, 224)):
    # Resize image to the target size
    image_resized = cv2.resize(image, target_size)
    # Normalize the image if needed
    image_normalized = image_resized / 255.0
    # Expand dimensions to fit the model input shape
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded
