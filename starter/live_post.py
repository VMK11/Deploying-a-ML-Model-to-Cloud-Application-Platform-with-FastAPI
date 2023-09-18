"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Standard library imports
import requests

print("Starting requests POSTing on live API")
URL = "https://model-serving-rer4.onrender.com/predict"
data = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
        }

print("Data requested: ", data)
respone = requests.post(URL, json=data)
print("Status code: ", respone.status_code)
print("Predict Result: ", respone.json())