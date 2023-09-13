import numpy as np
import pandas as pd

def preprocess_data(X, encoder):
    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"]
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    
    return X