# Tumor Classification using Image Processing and Machine Learning

## Project Overview

This project focuses on tumor classification using image processing and machine learning techniques. The dataset consists of medical images categorized into different tumor types. The approach involves image preprocessing, feature engineering, and model training for classification.

Packages Used

The following packages are utilized in this project:

os, PIL (Pillow), numpy, skimage, scikit-learn, matplotlib

imblearn.over_sampling.SMOTE - Used for handling class imbalance

## Dataset and Preprocessing Approach

Dataset is split into train, test, and validation sets.

Images are converted to grayscale, resized to 256x256, and normalized.

SMOTE is applied to balance the training dataset.

## Approaches Used

### Feature Engineering:

Images are resized and normalized to standardize input.

Extracted features using transformation techniques.

Flattened images and applied MinMax Scaling.

SMOTE was used to balance classes.

### Machine Learning Models:

Support Vector Machine (SVM) with RBF Kernel

Random Forest Classifier

K-Nearest Neighbors (k-NN)

Logistic Regression

Decision Tree Classifier

### Hyperparameter Tuning:

Applied RandomizedSearchCV for optimizing model performance.

Used cross-validation to improve generalization.

Model Evaluation:

Accuracy, precision, recall, and F1-score were computed.

Confusion matrices and classification reports were generated.

Compared model performance based on test set results.

## Future Improvements

Further fine-tuning of hyperparameters.

Testing deep learning approaches.

Expanding dataset and feature engineering techniques.

