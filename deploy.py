import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48

# Path Configuration
PROJECT_ROOT = 'E:/Project/Predicting Human Thoughts'
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_model.keras')
VALIDATION_DIR = os.path.join(PROJECT_ROOT, 'data', 'validation')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'evaluation_results')

def preprocess_image(image_path, face_cascade):
    """Preprocess single image"""
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image: {image_path}")

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Process the largest face if found
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        face = image[y:y+h, x:x+w]
    else:
        face = image
        
    # Resize and preprocess
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Enhance image
    face = enhance_image(face)
    
    # Normalize
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    
    return face

def enhance_image(image):
    """Apply image enhancement"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def evaluate_model():
    """Evaluate model performance"""
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_predictions = []
    all_true_labels = []
    results = []
    
    print("\nStarting evaluation...")
    
    # Process each emotion category
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(VALIDATION_DIR, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found for {emotion}")
            continue
            
        print(f"\nProcessing {emotion}...")
        
        for img_file in os.listdir(emotion_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            try:
                # Process image
                image_path = os.path.join(emotion_dir, img_file)
                face = preprocess_image(image_path, face_cascade)
                
                # Get predictions
                predictions = model.predict(face, verbose=0)[0]
                
                # Get top 3 predictions
                top_indices = np.argsort(predictions)[-3:][::-1]
                top_emotions = [(EMOTIONS[i], predictions[i] * 100) for i in top_indices]
                
                predicted_emotion = EMOTIONS[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                # Store results
                all_predictions.append(EMOTIONS.index(predicted_emotion))
                all_true_labels.append(EMOTIONS.index(emotion))
                
                results.append({
                    'file': img_file,
                    'true': emotion,
                    'predicted': predicted_emotion,
                    'confidence': confidence,
                    'top_3': top_emotions
                })
                
                print(f"{img_file}: {predicted_emotion} ({confidence:.1f}%)")
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
    
    # Generate report
    generate_report(all_true_labels, all_predictions, results)

def generate_report(true_labels, predictions, results):
    """Generate evaluation report"""
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    
    # Classification report
    report = classification_report(true_labels, predictions, target_names=EMOTIONS)
    
    # Save detailed results
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write("Emotion Recognition Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")
        
        for result in results:
            f.write(f"\nFile: {result['file']}\n")
            f.write(f"True Emotion: {result['true']}\n")
            f.write(f"Predicted: {result['predicted']} ({result['confidence']:.1f}%)\n")
            f.write("Top 3 Predictions:\n")
            for emotion, conf in result['top_3']:
                f.write(f"  {emotion}: {conf:.1f}%\n")
    
    print("\nEvaluation Report:")
    print(report)
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    try:
        evaluate_model()
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise e