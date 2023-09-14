"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from utils import preprocess_data

# Instantiate the FastAPI app.
app = FastAPI()


@app.get("/")
async def say_welcome():
    """
    Returns a greeting as a JSON response.
    
    Returns:
    --------
    dict
        A dictionary containing the greeting.
    """
    return {"greeting": "Welcome!"}


class Register(BaseModel):
    """
    Class to define the data object with its components and their types.
    """
    age: int = 22
    workclass: str = "Private"
    fnlgt: int = 31387
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Married-civ-spouse"
    occupation: str = "Adm-clerical"
    relationship: str = "Own-child"
    race: str = "Amer-Indian-Eskimo"
    sex: str = "Female"
    capital_gain: int = 2885
    capital_loss: int = 0
    hours_per_week: int = 25
    native_country: str = "United-States"


@app.post("/registers/")
async def create_register(register: Register):
    """
    Loads a pre-trained model and encoder, preprocesses the incoming data
    and predicts the outcome based on the model.

    Parameters:
    -----------
    register : Register
        An object of type Register that contains all the necessary features for the prediction.

    Returns:
    --------
    dict : A dictionary of predictions
    """
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')

    # Convert input to DataFrame
    X = pd.DataFrame(register.dict(), index=[0])

    # Preprocess the data
    X = preprocess_data(X, encoder)

    # Make predictions
    preds = model.predict(X)

    return {"salary": int(preds)}