import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import GaussianNoise

# Constants
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 64  # Increased from 48
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 0.001

class EmotionDataGenerator(Sequence):
    def __init__(self, dir_path, batch_size=32, is_training=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.is_training = is_training
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Get image paths and labels
        self.image_paths = []
        self.labels = []
        print(f"\nLoading data from {dir_path}")
        
        for i, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(dir_path, emotion)
            if os.path.exists(emotion_dir):
                image_files = [f for f in os.listdir(emotion_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"{emotion}: {len(image_files)} images")
                
                self.image_paths.extend([os.path.join(emotion_dir, f) for f in image_files])
                self.labels.extend([i] * len(image_files))
        
        self.indexes = np.arange(len(self.image_paths))
        if is_training:
            np.random.shuffle(self.indexes)
        
        print(f"Total images: {len(self.image_paths)}")
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        batch_size = len(batch_indexes)
        X = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        y = np.zeros((batch_size, len(EMOTIONS)), dtype=np.float32)
        
        for i, idx in enumerate(batch_indexes):
            img_path = self.image_paths[idx]
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    processed_img = self.preprocess_image(img)
                    if processed_img is not None:
                        if self.is_training:
                            processed_img = self.augment_image(processed_img)
                        X[i] = processed_img
                        y[i, self.labels[idx]] = 1.0
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        return X, y
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)
    
    def augment_image(self, img):
        """Apply various augmentations"""
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)  # Horizontal flip
        
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            height, width = img.shape[:2]
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            img = cv2.warpAffine(img, M, (width, height))
        
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
        return img
    
    def preprocess_image(self, img):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Get the face region
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face = img[y:y+h, x:x+w]
        else:
            face = img
        
        # Resize to model input size
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        # Convert to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE to each channel
        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_planes = cv2.split(lab)
        lab_planes = [clahe.apply(plane) for plane in lab_planes]
        lab = cv2.merge(lab_planes)
        face = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Normalize to [-1, 1]
        face = (face.astype(np.float32) - 127.5) / 127.5
        
        return face

def create_model():
    # Create base model
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Create model
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add noise for regularization
    x = GaussianNoise(0.1)(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom top layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # First dense block
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Second dense block
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output layer with weighted activations
    outputs = tf.keras.layers.Dense(
        len(EMOTIONS), 
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with optimized settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR,
        clipnorm=1.0,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model():
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    model_dir = os.path.join(os.path.dirname(base_dir), 'models')
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Create data generators
    train_generator = EmotionDataGenerator(
        train_dir, 
        batch_size=BATCH_SIZE, 
        is_training=True
    )
    
    validation_generator = EmotionDataGenerator(
        validation_dir, 
        batch_size=BATCH_SIZE, 
        is_training=False
    )
    
    # Create model
    model, base_model = create_model()
    model.summary()
    
    # Calculate class weights with balanced heuristic
    total_samples = len(train_generator.labels)
    class_counts = np.bincount(train_generator.labels)
    class_weights = {}
    
    max_count = np.max(class_counts)
    for i in range(len(EMOTIONS)):
        weight = (total_samples / (len(EMOTIONS) * class_counts[i])) * (class_counts[i] / max_count)
        class_weights[i] = np.clip(weight, 0.1, 10.0)  # Clip weights to prevent instability
    
    print("\nClass weights:", class_weights)
    
    # Training callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.1,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # First training phase
    print("\nPhase 1: Training with frozen base model...")
    history1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Fine-tuning phase
    print("\nPhase 2: Fine-tuning...")
    base_model.trainable = True
    
    # Freeze batch normalization layers
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR/10),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Save final model
    model.save(os.path.join(model_dir, 'final_model.keras'))
    
    return history1, history2

if __name__ == '__main__':
    print("CUDA Available:", tf.config.list_physical_devices('GPU'))
    try:
        history = train_model()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise e