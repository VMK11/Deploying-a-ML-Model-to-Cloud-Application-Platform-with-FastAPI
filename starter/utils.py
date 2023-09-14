"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Standard library imports
import numpy as np


def preprocess_data(x, encoder):
    """
    Preprocesses the data by encoding the categorical features and combining them with the continuous features.

    Parameters
    ----------
    x : pd.DataFrame
        Input data containing both categorical and continuous features.
    encoder : Encoder object (e.g., OneHotEncoder, LabelEncoder)
        Pre-trained encoder object to transform the categorical features.

    Returns
    -------
    np.ndarray
        Preprocessed data array where categorical features are encoded and combined with the continuous features.
    """
    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]
    x_categorical = x[categorical_features].values
    x_continuous = x.drop(*[categorical_features], axis=1)
    x_categorical = encoder.transform(x_categorical)
    x = np.concatenate([x_continuous, x_categorical], axis=1)

    return x
