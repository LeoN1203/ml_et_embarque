from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class HouseFeatures(BaseModel):
    size: int
    nb_rooms: int
    garden: int

@app.get("/predict")
def predict_get():
    return {"y_pred": 2}

@app.post("/predict")
def predict_post(data: HouseFeatures):
    model = joblib.load("regression.joblib")
    input_data = [[data.size, data.nb_rooms, data.garden]]
    prediction = model.predict(input_data)

    return {"y_pred": float(prediction[0])}
