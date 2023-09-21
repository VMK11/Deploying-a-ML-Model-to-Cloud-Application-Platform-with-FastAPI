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

def test_predict_error_case():
    """
    Test the model's prediction endpoint for specific error cases.

    This test case sends a request with a predefined set of data that
    lacks certain key attributes. It then checks the response for:
    1. A non-200 status code, indicating an error or non-standard response.
    2. Absence of specific attributes ("age", "education_num", "race", 
       "sex", and "native_country") in the JSON response, which are expected 
       to be missing or not returned by the server.

    Attributes:
        data_test (dict): The test data used for the prediction request.
        respone (Response): The server's response to the POST request.

    Assertions:
        1. The response status code is not 200.
        2. Certain keys are missing in the returned JSON response.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    data_test = {
        "workclass": "Private",
        "fnlgt": 59496,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "capital_gain": 2407,
        "capital_loss": 0,
        "hours_per_week": 40,
    }
    respone = client.post('/predict', json=data_test)
    assert respone.status_code != 200
    assert "age" not in respone.json()
    assert "education_num" not in respone.json()
    assert "race" not in respone.json()
    assert "sex" not in respone.json()
    assert "native_country" not in respone.json()

def test_predict_above_50k():
    """
    Test the model's prediction endpoint for a specific >50k scanrio.
    
    This test sends a request with a predefined dataset representing an individual 
    with attributes that, based on the underlying model, are expected to yield a 
    prediction result of ">50K". 
    
    Attributes:
        data_test (dict): The test data used for the prediction request. Represents 
                          an individual with a certain set of socio-economic attributes.
        respone (Response): The server's response to the POST request.

    Assertions:
        1. The response status code is 200, indicating a successful request.
        2. The returned prediction from the model is ">50K".

    Raises:
        AssertionError: If any of the assertions fail.
    """
    data_test = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 29933,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    respone = client.post('/predict', json=data_test)
    # Check response code
    assert respone.status_code == 200
    # Check response predict results of the model
    assert respone.json() == ">50K"

def test_predict_below_equal_50k():
    """
    Test the model's prediction endpoint for a specific <=50k scanrio.
    
    This test sends a request with a predefined dataset representing an individual 
    with attributes that, based on the underlying model, are expected to yield a 
    prediction result of "<=50K". 
    
    Attributes:
        data_test (dict): The test data used for the prediction request. Represents 
                          an individual with a certain set of socio-economic attributes.
        respone (Response): The server's response to the POST request.

    Assertions:
        1. The response status code is 200, indicating a successful request.
        2. The returned prediction from the model is "<=50K".

    Raises:
        AssertionError: If any of the assertions fail.
    """
    data_test = {
                "age": 30,
                "workclass": "Private",
                "fnlgt": 59496,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Sales",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2407,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
    respone = client.post('/predict', json=data_test)
    # Check response code
    assert respone.status_code == 200
    # Check response predict results of the model
    assert respone.json() == "<=50K" 