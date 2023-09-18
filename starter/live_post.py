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
print("Data requested: ", data)
respone = requests.post(URL, json=data)
print("Status code: ", respone.status_code)
print("Predict Result: ", respone.json())