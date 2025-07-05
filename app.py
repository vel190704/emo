from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Load the saved model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'emotion_recognition_model.keras')
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_face = preprocess_face(image)
        
        predictions = model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        emotion = EMOTIONS[emotion_idx]
        confidence = float(predictions[0][emotion_idx] * 100)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Emotion Detection API is running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)