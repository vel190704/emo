import os
import numpy as np
from data.train import EmotionDataGenerator, EMOTIONS

# Load training data using the existing data generator
train_dir = 'data/train'
train_generator = EmotionDataGenerator(train_dir, batch_size=32, is_training=False)

# Print first 5 image paths
print("First 5 training image paths:")
for i in range(min(5, len(train_generator.image_paths))):
    print(f"{i+1}: {train_generator.image_paths[i]}")

print(f"\nTotal training images: {len(train_generator.image_paths)}")
print(f"Emotion classes: {EMOTIONS}")

# Show distribution of emotions
emotion_counts = {}
for label in train_generator.labels:
    emotion_name = EMOTIONS[label]
    emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1

print(f"\nEmotion distribution in training data:")
for emotion, count in emotion_counts.items():
    print(f"  {emotion}: {count} images")