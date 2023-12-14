"""
Authors: Łukasz Kos, Emilian Murawski

Fashion-MNIST Image Classification using TensorFlow

This program uses TensorFlow and Keras to build a Convolutional Neural Network (CNN)
for image classification on the Fashion-MNIST dataset.
It creates two models with different network sizes just for comparing purposes.

Fashion-MNIST Dataset:
- The Fashion-MNIST dataset consists of 60,000 28x28 grayscale images in 10 different classes.
- Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

Model Architecture:
- The CNN model consists of three convolutional layers with max-pooling,
  followed by a flattening layer and two dense layers.
- Rectified Linear Unit (ReLU) is used as the activation function for convolutional layers,
  and no activation for the output layer.
- Adam optimizer is used with Sparse Categorical Crossentropy as the loss function.

Usage:
1. Load the Fashion-MNIST dataset and normalize pixel values to be between 0 and 1.
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
    python fashion_mnist_classification.py
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models  # pylint: disable=import-error, no-name-in-module
import matplotlib.pyplot as plt


def plot_model_performance(history, model_name):
    """Function to plot model performance"""
    plt.plot(history.history['accuracy'], label='Train Accuracy (' + model_name + ')')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy (' + model_name + ')')
    plt.plot(history.history['loss'], label='Train Loss (' + model_name + ')', linestyle="--")
    plt.plot(history.history['val_loss'], label='Val Loss (' + model_name + ')', linestyle="--")


# Load fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the first bigger model
model1 = models.Sequential()
model1.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(128, activation='relu'))
model1.add(layers.Dense(10))

# Compile the model
model1.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

# Train the model
history1 = model1.fit(train_images, train_labels, epochs=12,
                      validation_data=(test_images, test_labels))

# Evaluate the model
test_loss1, test_acc1 = model1.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy1: {test_acc1}")

# Smaller network
model2 = models.Sequential()
model2.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(32, activation='relu'))
model2.add(layers.Dense(10))

# Compile the model
model2.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

# Train the model
history2 = model2.fit(train_images, train_labels, epochs=12,
                      validation_data=(test_images, test_labels))

# Evaluate the model
test_loss2, test_acc2 = model1.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy1: {test_acc2}")

# Predictions
predictions1 = model1.predict(test_images)
predicted_labels1 = tf.argmax(predictions1, axis=1)

# Confusion Matrix using TensorFlow
cm1 = tf.math.confusion_matrix(test_labels, predicted_labels1, num_classes=10)

print("\nConfusion matrix 1:")
print(cm1)

# Predictions
predictions2 = model2.predict(test_images)
predicted_labels2 = tf.argmax(predictions2, axis=1)

# Confusion Matrix using TensorFlow
cm2 = tf.math.confusion_matrix(test_labels, predicted_labels2, num_classes=10)

print("\nConfusion matrix 2:")
print(cm2)

plot_model_performance(history1, 'Bigger network')
plot_model_performance(history2, 'Smaller network')

plt.title('Model Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()
