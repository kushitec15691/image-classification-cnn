import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(" Running from:", sys.executable)
print("TensorFlow version:", tf.__version__)

# Load and normalize CIFAR-10 data
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([]); plt.yticks([])
    plt.imshow(training_images[i])
    plt.xlabel(class_names[training_labels[i].item()])
plt.tight_layout()
plt.savefig("samples.png")  # Save the figure to the project folder
plt.show()


# Reduce training set size
training_images = training_images[:40000]
training_labels = training_labels[:40000]

# Data Augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(
    data_augmentation.flow(training_images, training_labels, batch_size=64),
    epochs=5,  # use 5 while testing
    validation_data=(testing_images, testing_labels),
    verbose=1
)

# Evaluation
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f" Test Loss: {loss:.4f}")
print(f" Test Accuracy: {accuracy:.4f}")
