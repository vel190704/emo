import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('path_to_your_model.h5')  # Replace with your trained model's path

# Define image size expected by the model
IMAGE_SIZE = 224  # Replace if your model uses a different input size

def preprocess_frame(frame):
    """
    Preprocess the frame to match the model's input requirements.
    """
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to model's input size
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam")
    exit()

print("Press 'q' to exit")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Predict using the model
    predictions = model.predict(processed_frame)
    
    # Display predictions (modify as per your model's output format)
    predicted_class = np.argmax(predictions)  # For classification
    confidence = np.max(predictions)  # Confidence score

    # Show results on the frame
    cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam - Model Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
