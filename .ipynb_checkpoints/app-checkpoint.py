from fastapi import FastAPI
import joblib
import pandas as pd

artifact = joblib.load("house_price_model_final.pkl")

model = artifact["model"]
selected_features = artifact["selected_features"]
scaler = artifact["scaler"]

app = FastAPI()


@app.get("/")
def home():
    return {"message": "House Price Prediction API"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    df = df.reindex(columns=selected_features, fill_value=0)

    if scaler is not None:
        df = scaler.transform(df)

    prediction = model.predict(df)

    return {
        "predicted_price": float(prediction[0])
    }