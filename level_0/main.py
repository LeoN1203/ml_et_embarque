from fastapi import FastAPI
import joblib

app = FastAPI()

@app.get("/predict")
def predict_get():
    return {"y_pred": 2}

@app.post("/predict")
def predict_post(data: dict):
    model = joblib.load("regression.joblib")
    input_data = [[data["size"], data["nb_rooms"], data["garden"]]]
    prediction = model.predict(input_data)

    return {"y_pred": float(prediction[0])}
