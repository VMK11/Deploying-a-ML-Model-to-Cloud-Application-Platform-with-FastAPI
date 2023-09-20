"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 13-09-2023
"""

# Standard library imports
import logging
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

# Third-party imports
from pydantic import BaseModel
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference

logging.basicConfig(
    filename="server.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()


class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
        }

app = FastAPI()

model = joblib.load("starter/model/model.pkl")
encoder = joblib.load("starter/model/encoder.pkl")
label_binarizer = joblib.load("starter/model/lb.pkl")

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

@app.get("/")
async def get_welcome():
    return {"greeting": "Hello world!"}

@app.post("/data/")
async def inference_data(data: Input):
    if data.age < 0:
        raise HTTPException(status_code=400, detail="Age needs to be above 0.")

    try:
        test = pd.DataFrame(jsonable_encoder(data), index=[0])
        
    #     # Proces the test data with the process_data function.
        x_test, _, _, _ = process_data(
            test, categorical_features=cat_features, label=None, training=False,
            encoder = encoder, lb=label_binarizer
        )

        preds = inference(model, x_test)
        salary = label_binarizer.inverse_transform(preds)[0]

        logger.info("Inference successful: {}".format(salary))

        return salary

    except Exception as e:
        logger.error("An error occurred: {}".format(e))
        raise