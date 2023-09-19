"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Third-party imports
from fastapi.testclient import TestClient

# Application-specific imports
# from main import app  # Import our FastAPI app from main.py.
import sys
sys.path.append('../')
from main import app  # Import our FastAPI app from main.py.

# Instantiate a test client with our FastAPI app.
client = TestClient(app)


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


def test_post_inference():
    """
    Description:
        Test the model inference endpoint with a sample data payload.
        Expect a 200 status code and an income prediction of "<=50K".
    
    Parameters: None
    
    Returns:
        None. Asserts are used to validate responses.
    """
    sample = {
        # Sample user data for model inference.
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json() == "<=50K"


def test_post_inference_false_query():
    """
    Description:
        Test the model inference endpoint with an incomplete data payload.
        Verify that the response does not contain certain keys that are 
        missing in the request payload.
    
    Parameters: None
    
    Returns:
        None. Asserts are used to validate responses.
    """
    sample = {
        # Sample incomplete user data for model inference.
    }
    response = client.post("/predict", json=sample)
    assert 'capital_gain' not in response.json()
    assert 'capital_loss' not in response.json()
    assert 'hours_per_week' not in response.json()
    assert 'native_country' not in response.json()


def test_inference_greater_than_50k_case():
    """
    Description:
        Test the registration endpoint with a data payload that should result
        in a predicted income greater than 50K. Expect a 200 status code and 
        a salary prediction of 1 (">50K").
    
    Parameters: None
    
    Returns:
        None. Asserts are used to validate responses.
    """
    response = client.post("/registers/", json={
        # Sample user data that should result in a ">50K" income prediction.
    })
    assert response.status_code == 200
    assert response.json() == {"salary": 1}


def test_inference_less_than_or_equal_to_50k_case():
    """
    Description:
        Test the registration endpoint with a data payload that should result
        in a predicted income less than or equal to 50K. Expect a 200 status 
        code and a salary prediction of 0 ("<=50K").
    
    Parameters: None
    
    Returns:
        None. Asserts are used to validate responses.
    """
    response = client.post("/registers/", json={
        # Sample user data that should result in a "<=50K" income prediction.
    })
    assert response.status_code == 200
    assert response.json() == {"salary": 0}
