import cv2
import numpy as np
from keras.models import load_model
import time
import os

class MicroExpressionDetector:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'data', 'models', 'emotion_recognition_model.keras')
        
        print(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model(model_path)
        self.frame_buffer = []
        self.buffer_size = 15
        self.threshold = 0.3
        
    def preprocess_frame(self, frame):
        # Detect faces in the original frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        # Process the largest face
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Extract face ROI from the original BGR frame
        face_roi = frame[y:y+h, x:x+w]
        
        # Resize to match model input (keeping all 3 color channels)
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize pixel values
        face_roi = face_roi / 255.0
        
        return face_roi, (x, y, w, h)
        
    def detect_micro_expression(self, frame):
        processed_data = self.preprocess_frame(frame)
        if processed_data is None:
            return None, None
            
        face_roi, face_coords = processed_data
        
        # Add to frame buffer
        self.frame_buffer.append(face_roi)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
            
        # Need minimum frames for analysis
        if len(self.frame_buffer) < 3:
            return None, face_coords
            
        # Analyze rapid changes
        predictions = []
        for face in self.frame_buffer[-3:]:
            # Reshape to match model input shape (batch_size, height, width, channels)
            face_input = np.expand_dims(face, axis=0)
            pred = self.emotion_model.predict(face_input, verbose=0)
            predictions.append(pred[0])
            
        # Calculate emotion changes
        emotion_changes = np.abs(np.diff(predictions, axis=0))
        max_change = np.max(emotion_changes)
        
        if max_change > self.threshold:
            changing_emotions = np.where(emotion_changes[-1] > self.threshold)[0]
            return changing_emotions, face_coords
            
        return None, face_coords

def start_webcam_detection():
    try:
        detector = MicroExpressionDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            micro_expressions, face_coords = detector.detect_micro_expression(frame)
            
            if face_coords is not None:
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if micro_expressions is not None:
                    detected_emotions = [emotion_labels[i] for i in micro_expressions]
                    text = "Micro-expression: " + ", ".join(detected_emotions)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
            
            cv2.imshow('Micro-Expression Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam_detection()