"""
Authors: Łukasz Kos, Emilian Murawski

CIFAR-10 Image Classification using TensorFlow

This program uses TensorFlow and Keras to build a Convolutional Neural Network (CNN)
for image classification on the CIFAR-10 dataset.

CIFAR-10 Dataset:
- The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes.
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

Model Architecture:
- The CNN model consists of three convolutional layers with max-pooling,
  followed by a flattening layer and two dense layers.
- Rectified Linear Unit (ReLU) is used as the activation function for convolutional layers,
  and softmax for the output layer.
- Adam optimizer is used with Sparse Categorical Crossentropy as the loss function.

Usage:
1. Load the CIFAR-10 dataset and normalize pixel values to be between 0 and 1.
2. Define the CNN model architecture using the Sequential API of Keras.
3. Compile the model specifying the optimizer, loss function, and evaluation metrics.
4. Train the model using the training data and validate it on the test data.
5. Evaluate the trained model on the test set and print the test accuracy.
6. Plot the training history to visualize accuracy over epochs.
7. Generate and print the confusion matrix.

Parameters:
- train_images (numpy.ndarray): Training images.
- train_labels (numpy.ndarray): Training labels.
- test_images (numpy.ndarray): Test images.
- test_labels (numpy.ndarray): Test labels.

Returns:
- None

Example:
    python cifar10_classification.py
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models  # pylint: disable=import-error, no-name-in-module
import matplotlib.pyplot as plt


# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=12,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

predictions = model.predict(test_images)
predicted_labels = tf.argmax(predictions, axis=1)

# Confusion Matrix using TensorFlow
cm = tf.math.confusion_matrix(test_labels, predicted_labels, num_classes=10)

print("\nConfusion matrix:")
print(cm)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
