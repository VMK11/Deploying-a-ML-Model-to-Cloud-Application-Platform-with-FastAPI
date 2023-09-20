# """
# Description: Helper functions to assist with the customer churn classification
# Author: V.Manousakis-Kokorakis
# Date: 13-09-2023
# """

# # Standard library imports
# import logging
# import joblib
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from fastapi.encoders import jsonable_encoder

# # Third-party imports
# from pydantic import BaseModel
# import pandas as pd
# from starter.ml.data import process_data
# from starter.ml.model import inference

# logging.basicConfig(
#     filename="server.log",
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# logger = logging.getLogger()

# column = [
#         "age",
#         "workclass",
#         "fnlwgt",
#         "education",
#         "education_num",
#         "marital-status",
#         "occupation",
#         "relationship",
#         "race",
#         "sex",
#         "capital_gain",
#         "capital_loss",
#         "hours-per-week",
#         "native-country"]


# class InputData(BaseModel):
#     age: int
#     workclass: str
#     fnlgt: int
#     education: str
#     education_num: int
#     marital_status: str
#     occupation: str
#     relationship: str
#     race: str
#     sex: str
#     capital_gain: int
#     capital_loss: int
#     hours_per_week: int
#     native_country: str

#     class Config:
#         schema_extra = {
#             "example": {
#                 "age": 39,
#                 "workclass": "State-gov",
#                 "fnlgt": 77516,
#                 "education": "Bachelors",
#                 "education_num": 13,
#                 "marital_status": "Never-married",
#                 "occupation": "Adm-clerical",
#                 "relationship": "Not-in-family",
#                 "race": "White",
#                 "sex": "Male",
#                 "capital_gain": 2174,
#                 "capital_loss": 0,
#                 "hours_per_week": 40,
#                 "native_country": "United-States"
#             }
#         }

# app = FastAPI()

# model = joblib.load("starter/model/model.pkl")
# encoder = joblib.load("starter/model/encoder.pkl")
# label_binarizer = joblib.load("starter/model/lb.pkl")

# cat_features = [
#     "workclass",
#     "education",
#     "marital_status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native_country",
# ]

# @app.get("/")
# async def get_welcome():
#     return {"greeting": "Hello world!"}

# #POST that does model inference.
# @app.post("/predict")
# async def predict(input_data: InputData):
#     input = np.array([[
#                         input_data.age,
#                         input_data.workclass,
#                         input_data.fnlgt,
#                         input_data.education,
#                         input_data.education_num,
#                         input_data.marital_status,
#                         input_data.occupation,
#                         input_data.relationship,
#                         input_data.race,
#                         input_data.sex,
#                         input_data.capital_gain,
#                         input_data.capital_loss,
#                         input_data.hours_per_week,
#                         input_data.native_country
#                     ]])
#     data = pd.DataFrame(data=input, columns=column)
#     #Process the data
#     X, _, _, _ = process_data(
#                     data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
#                 )
#     #Inference 
#     y = inference(model=model, X=X)
#     output = label_binarizer.inverse_transform(y)[0]
#     return output


# @app.post("/data/")
# async def inference_data(data: InputData):
#     if data.age < 0:
#         raise HTTPException(status_code=400, detail="Age needs to be above 0.")

#     try:
#         test = pd.DataFrame(jsonable_encoder(data), index=[0])
        
#     #     # Proces the test data with the process_data function.
#         x_test, _, _, _ = process_data(
#             test, categorical_features=cat_features, label=None, training=False,
#             encoder = encoder, lb=label_binarizer
#         )

#         preds = inference(model, x_test)
#         salary = label_binarizer.inverse_transform(preds)[0]

#         logger.info("Inference successful: {}".format(salary))

#         return salary

#     except Exception as e:
#         logger.error("An error occurred: {}".format(e))
#         raise

from fastapi import FastAPI
import pandas as pd    
import numpy as np
from pydantic import BaseModel
import joblib
from starter.ml.data import process_data, cat_features
from starter.ml.model import inference

column = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

class InputData(BaseModel):
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
        }

# create app
app = FastAPI()

# load models
model = joblib.load("starter/model/model.pkl")
encoder = joblib.load("starter/model/encoder.pkl")
label_binarizer = joblib.load("starter/model/lb.pkl")

#GET on the root giving a welcome message.
@app.get("/")
async def get_welcome():
    return {"greeting": "Hello world!"}

#POST that does model inference.
@app.post("/predict")
async def predict(input_data: InputData):
    input = np.array([[
                        input_data.age,
                        input_data.workclass,
                        input_data.fnlgt,
                        input_data.education,
                        input_data.education_num,
                        input_data.marital_status,
                        input_data.occupation,
                        input_data.relationship,
                        input_data.race,
                        input_data.sex,
                        input_data.capital_gain,
                        input_data.capital_loss,
                        input_data.hours_per_week,
                        input_data.native_country
                    ]])
    data = pd.DataFrame(data=input, columns=column)
    #Process the data
    X, _, _, _ = process_data(
                    data, categorical_features=cat_features, training=False, encoder=encoder, lb=label_binarizer
                )
    #Inference 
    y = inference(model=model, X=X)
    output = label_binarizer.inverse_transform(y)[0]
    return output