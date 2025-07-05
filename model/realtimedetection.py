import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque
import time

class EmotionDetector:
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    IMG_SIZE = 48
    TEMPORAL_WINDOW = 3
    
    def __init__(self, model_path=None):
        if model_path is None:
            self.model_path = os.path.join("models", "best_model.keras")
        else:
            self.model_path = model_path
        
        self.load_models()
        self.prediction_history = deque(maxlen=self.TEMPORAL_WINDOW)
        
        # More distinctive colors
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Pure Red
            'disgust': (0, 255, 0),    # Pure Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 140, 0)   # Orange
        }
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def load_models(self):
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def preprocess_face(self, face):
        try:
            # Resize
            face = cv2.resize(face, (self.IMG_SIZE, self.IMG_SIZE))
            
            # Strong contrast enhancement
            face = cv2.convertScaleAbs(face, alpha=1.5, beta=30)
            
            # Convert to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Strong CLAHE enhancement
            lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            lab_planes = cv2.split(lab)
            lab_planes = [clahe.apply(plane) for plane in lab_planes]
            lab = cv2.merge(lab_planes)
            face = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Normalize to [-1, 1]
            face = (face.astype(np.float32) - 127.5) / 127.5
            face = np.expand_dims(face, axis=0)
            return face
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def adjust_predictions(self, predictions):
        # Strongly reduce neutral bias and boost other emotions
        adjusted_weights = np.array([1.5, 1.5, 1.5, 1.8, 0.5, 1.5, 1.8])  # Boost happy and surprise more
        adjusted_preds = predictions * adjusted_weights
        
        # Suppress very low probabilities
        adjusted_preds[adjusted_preds < 0.1] = 0
        
        # Renormalize
        if np.sum(adjusted_preds) > 0:
            adjusted_preds = adjusted_preds / np.sum(adjusted_preds)
        
        return adjusted_preds
    
    def smooth_predictions(self, current_prediction):
        adjusted_pred = self.adjust_predictions(current_prediction)
        self.prediction_history.append(adjusted_pred)
        
        if len(self.prediction_history) < self.TEMPORAL_WINDOW:
            return adjusted_pred
        
        return np.mean(self.prediction_history, axis=0)
    
    def draw_emotion_info(self, frame, predictions, x, y, w, h):
        # Always get top emotion
        emotion_idx = np.argmax(predictions)
        emotion = self.EMOTIONS[emotion_idx]
        prob = predictions[emotion_idx] * 100
        color = self.emotion_colors[emotion]
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw text with background
        text = f"{emotion.upper()}"
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        
        # Background rectangle
        cv2.rectangle(frame,
                    (x, y - text_size[1] - 10),
                    (x + text_size[0], y),
                    color, -1)
        
        # Text
        cv2.putText(frame, text,
                   (x, y - 5),
                   cv2.FONT_HERSHEY_DUPLEX,
                   font_scale, (255, 255, 255), thickness)
    
    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            processed_face = self.preprocess_face(face_roi)
            
            if processed_face is not None:
                predictions = self.model.predict(processed_face, verbose=0)[0]
                smoothed_predictions = self.smooth_predictions(predictions)
                self.draw_emotion_info(frame, smoothed_predictions, x, y, w, h)
        
        return frame

def start_webcam_detection():
    try:
        detector = EmotionDetector()
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        print("Starting emotion detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            processed_frame = detector.process_frame(frame)
            cv2.imshow('Emotion Detection', processed_frame)
            
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