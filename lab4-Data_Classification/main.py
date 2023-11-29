import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
file_path = 'seeds_dataset.txt'
column_names = ['Area', 'Perimeter', 'Compactness', 'Length of kernel', 'Width of kernel',
                'Asymmetry coefficient', 'Length of kernel groove', 'Class']
df = pd.read_csv(file_path, sep='\t', names=column_names)

# Separating features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Standardization scales the data such that it has a mean of 0 and a standard deviation of 1. 
This is particularly useful for algorithms that are sensitive to the scale of the input features.
Standardization is generally the preferred method for algorithms like SVMs and k-NN.
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
For SVM (Support Vector Machine), the most critical parameters are:
- C: Regularization parameter. The strength of the regularization is inversely proportional to C.
  Must be strictly positive. Type float, default is 1.
- Kernel Type: Determines the type of hyperplane used to separate the data.
  This includes linear, poly, rbf, and sigmoid.
- Gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Values are 'scale' (default), 'auto' or
  float (non-negative).
"""
svm_classifier = SVC(C=1, kernel='sigmoid', gamma='scale')

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Predict on the test set
svm_predictions = svm_classifier.predict(X_test_scaled)

# Evaluation
# Accuracy is a ratio of correctly predicted observation to the total number of observations
accuracy = accuracy_score(y_test, svm_predictions)
print("Accuracy:", accuracy)

'''
Confusion matrix (error matrix) is a useful tool for multiclass classification problems
Rows represents actual classes, columns represents predicted classes.
In first row (actual class one): 
- first column show number of instances correctly classified as class one (True Positives)
- second column show number of instances incorrectly classified as class two (False Positives)
- third column show number of instances incorrectly classified as class three (False Positives)
In second row (actual class two):
- first column show number of instances incorrectly classified as class one (False Positives)
- second column show number of instances correctly classified as class two (True Positives)
- third column show number of instances incorrectly classified as class three (False Positives)
etc. 
'''
conf_matrix = confusion_matrix(y_test, svm_predictions)
print("Confusion Matrix:\n", conf_matrix)

'''
Classification Report with Precision, Recall and F1-Score

Precision (positive predictive value, PPV) is the ratio of correctly predicted true positives 
to the total number of predicted positives for specific class. 
High precision relates to the low false positive rate.

Recall (sensitivity or true positive rate, TPR) is the ratio of correctly predicted true positives 
to the all observations in actual class.

F1 Score is the average of Precision and Recall.
'''
class_report = classification_report(y_test, svm_predictions)
print("Classification Report:\n", class_report)
