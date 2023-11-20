# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load model
with open("D:/Project_Data/project/Project Pribadi/Deployment_visualization/best_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

# Categorical features

class DataInput(BaseModel):
    area: int
    Bedrooms_1: int
    Bedrooms_2: int
    Bedrooms_3: int
    Bedrooms_4: int
    Bedrooms_5: int
    Bathrooms_1: int
    Bathrooms_2: int
    Bathrooms_3: int
    Stories_1: int
    Stories_2: int
    Stories_3: int
    Stories_4: int
    Mainroad_Yes: int
    Mainroad_No: int
    GuestRoom_Yes: int
    GuestRooms_No: int
    Basement_Yes: int
    Basement_No: int
    HotWaterHeating_Yes: int
    HotWaterHeating_No: int
    Airconditioning_Yes: int
    Airconditioning_No: int
    Parking_2: int
    Parking_3: int
    Parking_0: int
    Parking_1: int
    Prefarea_Yes: int
    Prefarea_No: int
    furnishingstatus_furnished: int
    furnishingstatus_semi_furnished: int
    furnishingstatus_unfurnished: int

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API'}

origins = [
    "http://localhost",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/prediction')
async def get_data(data: DataInput):
    try:
        # Convert data to a list for model prediction
        input_list = [getattr(data, feature) for feature in data.__annotations__]

        # Reshape input data to match the shape used during training
        input_data = [input_list]

        # Debugging statements
        print("Input Data:", input_data)

        result_prediction = model.predict(input_data)[0]

        # Debugging statement
        print("Result Prediction:", result_prediction)

        return {"prediction": result_prediction}
    except Exception as e:
        # Debugging statement
        print("Error:", str(e))
        return {"error": f"An error occurred: {str(e)}"}
