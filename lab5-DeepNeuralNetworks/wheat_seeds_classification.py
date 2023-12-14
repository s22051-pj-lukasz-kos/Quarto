"""
Authors: Łukasz Kos, Emilian Murawski

Wheat Seeds classification from lab4 using TensorFlow

Dataset classifies seeds on class 1, 2 and 3. It takes into account perimeter of seed, compactness,
length of kernel, width of kernel, asymmetry coefficient and length of kernel groove.

Wheat Seeds Dataset:
- Wheat Seeds datasets consist of table with 210 records with 6 features and 1 label column
- Classes: 1, 2 or 3

Model Architecture:
- Sequential model is the simplest type of model in Keras and Tensorflow. It is suitable for
  a plain stack of layers where each layer has exactly one input tensor and one output tensor.
- This example consist of three dense layers

Usage:
- Split the data into features and target labels.
- Normalize the data and create one-hot encoding format.
- Building Sequential model with tree Dense layers
- First two layers use rectified linear unit activation function and output layer uses
  probability distribution (values are in range (0, 1) and sum to 1.
- Compiling the model with the Adam optimizer and categorical crossentropy loss function.
- Training the model on the training data.
- Evaluating the model on the test data.
- Generate and print the confusion matrix

Parameters:
- X_train (numpy.ndarray): Training features
- y_train (numpy.ndarray): Training labels
- X_test (numpy.ndarray): Test features
- y_test (numpy.ndarray); Test labels

Returns:
- None

Example:
- wheat_seeds_classification.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tensorflow.keras.models import Sequential  # pylint: disable=import-error, no-name-in-module
from tensorflow.keras.layers import Dense  # pylint: disable=import-error, no-name-in-module
from tensorflow.keras.utils import to_categorical  # pylint: disable=import-error, no-name-in-module

# Load the dataset
column_names = ['Area', 'Perimeter', 'Compactness', 'Length of kernel',
                'Width of kernel', 'Asymmetry coefficient',
                'Length of kernel groove', 'Class']
dataset = pd.read_csv('seeds_dataset.txt', sep='\t', names=column_names)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Normalize data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

'''
Creating one-hot encoded format, which means that each unique class is represented by 
separate column. Each column is binary (it contains 0s and 1s)
'''
y_one_hot = to_categorical(y - 1)  # subtracting 1 to make classes start from 0

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = (train_test_split
                                    (X_normalized, y_one_hot, test_size=0.2, random_state=42))

# Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))
