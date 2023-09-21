"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Third-party imports
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
# Application-specific imports
from starter.ml.data import process_data, load_data, cat_features
from starter.ml.model import inference, train_model, compute_model_metrics

@pytest.fixture(scope="module")
def data():
    """
    Description:
        Pytest fixture to load the census data for the tests.
    
    Parameters: None
    
    Returns:
        pandas.DataFrame: Loaded data from the census CSV file.
    """
    datapath = "./starter/data/census.csv"
    return load_data(datapath)


def test_load_data(data):
    """
    Description:
        Test the `load_data` function to ensure it properly loads the data.
    
    Parameters:
        data (pandas.DataFrame): Loaded census data.
    
    Returns:
        None. Asserts are used to validate the loaded data.
    """
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_process_data(data):
    """
    Description:
        Test the `process_data` function to ensure it splits and processes the data correctly.
    
    Parameters:
        data (pandas.DataFrame): Loaded census data.
    
    Returns:
        None. Asserts are used to validate the processed data.
    """
    train, test = train_test_split(data, test_size=0.3, random_state=0)
    # Process data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(data)


def test_model_output(data):
    """
        Test the model inference after training
    """
    model = joblib.load("./starter/model/model.pkl")
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    y_pred = inference(model, X_test)
    assert len(y_test) == len(y_pred)
    
