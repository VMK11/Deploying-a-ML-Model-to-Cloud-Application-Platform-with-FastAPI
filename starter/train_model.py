"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Standard library imports
import logging
import joblib

# Third-party imports
from sklearn.model_selection import train_test_split

# Application-specific imports
from ml.data import load_data, process_data, slice_performance
from ml.model import train_model, compute_model_metrics, inference


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Load the data
data = load_data(path='./starter/data/census.csv')

# Use train-test split (consider K-fold cross-validation for enhancement)
train, test = train_test_split(data, test_size=0.20)

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and evaluate the model
print("Starting training")
model = train_model(X_train, y_train)

print("Inference")
y_pred = inference(model, X_test)

print("Evaluation")
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f"Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, FBeta: {round(fbeta, 2)}")

# Evaluate performance on data slices
print("Calculating performance of the model on slices of the data")
slice_performance(test, model, encoder, lb, compute_model_metrics, cat_features=cat_features)

# Save the model and preprocessors
print("Saving model")
joblib.dump(model, './starter/model/model.pkl')
joblib.dump(encoder, './starter/model/encoder.pkl')
joblib.dump(lb, './starter/model/lb.pkl')
