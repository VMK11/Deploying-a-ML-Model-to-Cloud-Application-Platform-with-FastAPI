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
    r = client.get("/")
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {"fetch": "Welcome!"}


def test_post_query():
    r = client.post("/data/", json=data)
    print(r.json())
    assert r.status_code == 200

test_get()
test_post_query()ß