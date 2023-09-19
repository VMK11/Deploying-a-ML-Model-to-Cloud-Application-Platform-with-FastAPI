"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Application-specific imports
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

data = {
    "age": 25,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Adm-clerical",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
    "salary": "<=50K"}


def test_get():
    """
    Test the GET request for the root URL ("/").
    
    This function:
    - Sends a GET request to the root URL ("/")
    - Checks if the status code is 200
    - Checks if the JSON response matches the expected output
    
    Example:
        >>> test_get()
        Should print the JSON output and pass the assertions.
    
    Notes:
        - Uses a test client object named `client`.
        - Prints the JSON response to stdout.
    """
    r = client.get("/")
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {"fetch": "Welcome!"}


def test_post_query():
    """
    Test the POST request for the URL "/data/".

    This function:
    - Sends a POST request to the URL "/data/" with JSON payload `data`
    - Checks if the status code is 200

    Example:
        >>> test_post_query()
        Should print the JSON output and pass the assertions.

    Notes:
        - Uses a test client object named `client`.
        - The variable `data` should contain the JSON payload to be sent in the POST request.
        - Prints the JSON response to stdout.
    """
    r = client.post("/data/", json=data)
    print(r.json())
    assert r.status_code == 200


test_get()
test_post_query()