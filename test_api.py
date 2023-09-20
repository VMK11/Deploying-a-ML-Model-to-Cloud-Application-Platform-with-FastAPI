"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Third-party imports
from fastapi.testclient import TestClient



# Application-specific imports
from main import app  # Import our FastAPI app from main.py.

# Instantiate a test client with our FastAPI app.
client = TestClient(app)

def test_post_inference_false_query():
    """
        test_model inference with false query
    """
    sample = {
                "age": 35,
                "workclass": "Private",
                "fnlgt": 77516,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Divorced",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "Black",
                "sex": "Male"
            }
    r = client.post("/predict", json=sample)
    assert 'capital_gain' not in r.json()
    assert 'capital_loss' not in r.json()
    assert 'hours_per_week' not in r.json()
    assert 'native_country' not in r.json()

def test_post_inference():
    """
        test_model inference
    """
    sample = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    response = client.post("/predict", json=sample)
    print(response)
    assert response.status_code == 200
    assert response.json() == "<=50K" 

def test_say_welcome():
    """
    Description:
        Test the root endpoint to ensure it returns a 200 status code 
        and the expected greeting message.
    
    Parameters: None
    
    Returns:
        None. Asserts are used to validate responses.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello world!"}

