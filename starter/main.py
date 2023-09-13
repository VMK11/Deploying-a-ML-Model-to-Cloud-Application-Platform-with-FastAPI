# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from utils import preprocess_data

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_welcome():
    return {"greeting": "Welcome!"}

# Declare the data object with its components and their type.
class Register(BaseModel):
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
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    X = pd.DataFrame(register.dict(), index=[0])
    X = preprocess_data(X, encoder)
    preds = model.predict(X)
    return {"salary": int(preds)}