import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    """
       Authors: Łukasz Kos,Emilian Murawski
       Trains a classifier using the provided training data and evaluates its performance on the test data.

       Parameters:
       - classifier (object): An instance of a scikit-learn classifier (e.g., DecisionTreeClassifier, SVC).
       - X_train (DataFrame or array-like): Features of the training dataset.
       - y_train (Series or array-like): Target variable of the training dataset.
       - X_test (DataFrame or array-like): Features of the test dataset.
       - y_test (Series or array-like): Target variable of the test dataset.

       Returns:
       None: Prints accuracy, confusion matrix, and classification report of the classifier on the test data.
    """
    classifier.fit(X_train, y_train)

    # Predict on the test set
    predictions = classifier.predict(X_test)

    # Evaluation
    # Accuracy is a ratio of correctly predicted observation to the total number of observations
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

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
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")

    '''
    Classification Report with Precision, Recall and F1-Score

    Precision (positive predictive value, PPV) is the ratio of correctly predicted true positives 
    to the total number of predicted positives for specific class. 
    High precision relates to the low false positive rate.

    Recall (sensitivity or true positive rate, TPR) is the ratio of correctly predicted true positives 
    to the all observations in actual class.

    F1 Score is the average of Precision and Recall.
    '''
    class_report = classification_report(y_test, predictions)
    print(f"Classification Report:\n{class_report}")


# Load the wheat dataset
file_path_wheat = 'seeds_dataset.txt'
column_names_wheat = ['Area', 'Perimeter', 'Compactness', 'Length of kernel', 'Width of kernel',
                      'Asymmetry coefficient', 'Length of kernel groove', 'Class']
df_wheat = pd.read_csv(file_path_wheat, names=column_names_wheat, sep="\t")

# Separating features and target variable for wheat dataset
X_wheat = df_wheat.iloc[:, :-1]
y_wheat = df_wheat.iloc[:, -1]

# Split the wheat dataset into training (80%) and testing (20%) sets
X_train_wheat, X_test_wheat, y_train_wheat, y_test_wheat = train_test_split(X_wheat, y_wheat, test_size=0.2,
                                                                            random_state=42)

# Load the heart dataset
# Used data https://www.kaggle.com/datasets/ineubytes/heart-disease-dataset/data
file_path_heart = 'heart.csv'

df_heart = pd.read_csv(file_path_heart, header=0, sep=',')

# Separating features and target variable for heart dataset
X_heart = df_heart.iloc[:, :-1]
y_heart = df_heart.iloc[:, -1]

# Split the heart dataset into training (80%) and testing (20%) sets
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2,
                                                                            random_state=42)

'''
Standardization scales the data such that it has a mean of 0 and a standard deviation of 1. 
This is particularly useful for algorithms that are sensitive to the scale of the input features.
Standardization is generally the preferred method for algorithms like SVMs and k-NN.
'''
scaler = StandardScaler()
X_train_heart_scaled = scaler.fit_transform(X_train_heart)
X_test_heart_scaled = scaler.transform(X_test_heart)
X_train_wheat_scaled = scaler.fit_transform(X_train_wheat)
X_test_wheat_scaled = scaler.transform(X_test_wheat)

"""
For Decision Trees, the most critical parameters are:
- max_depth: The maximum depth of the tree. It controls the maximum number of levels in the tree.
A higher value can lead to overfitting.
- min_samples_split: The minimum number of samples required to split an internal node. 
Higher values prevent the tree from making overly specific and detailed splits.
- min_samples_leaf: The minimum number of samples required to be in a leaf node.
It controls the granularity of the leaves. Higher values prevent the tree from creating tiny leaves.
- criterion: The function to measure the quality of a split. 
Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.

For SVM (Support Vector Machine), the most critical parameters are:
- C: Regularization parameter. The strength of the regularization is inversely proportional to C.
  Must be strictly positive. Type float, default is 1.
- Kernel Type: Determines the type of hyperplane used to separate the data.
  This includes linear, poly, rbf, and sigmoid.
- Gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Values are 'scale' (default), 'auto' or
  float (non-negative).
"""

# Decision Tree parameters
dt_params = {
    'max_depth': 5,
    'random_state': 42,
    'criterion': 'gini'
}

# SVM parameters
svm_params = {
    'C': 1,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# Create and evaluate Decision Tree classifier for the wheat dataset
dt_classifier_wheat = DecisionTreeClassifier(**dt_params)
print("Results for Decision Tree Classifier on Wheat Dataset:")
train_and_evaluate_classifier(dt_classifier_wheat, X_train_wheat, y_train_wheat, X_test_wheat, y_test_wheat)

# Create and evaluate Decision Tree classifier for the heart dataset
dt_classifier_heart = DecisionTreeClassifier(**dt_params)
print("\nResults for Decision Tree Classifier on Heart Disease Dataset:")
train_and_evaluate_classifier(dt_classifier_heart, X_train_heart, y_train_heart, X_test_heart, y_test_heart)

# Create and evaluate SVM classifier for the wheat dataset
svm_classifier_wheat = SVC(**svm_params)
print("\nResults for SVM Classifier on Wheat Dataset:")
train_and_evaluate_classifier(svm_classifier_wheat, X_train_wheat_scaled, y_train_wheat, X_test_wheat_scaled,
                              y_test_wheat)

# Create and evaluate SVM classifier for the heart dataset
svm_classifier_heart = SVC(**svm_params)
print("\nResults for SVM Classifier on Heart Disease Dataset:")
train_and_evaluate_classifier(svm_classifier_heart, X_train_heart_scaled, y_train_heart, X_test_heart_scaled,
                              y_test_heart)
