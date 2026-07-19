import os
import pandas as pd
import pickle 
from fastapi import FastAPI
from pydantic import BaseModel
from model_training import models_dir

app = FastAPI(title="E-commerce API")
models_dir = 'models'
print("loading models into memory...")

# rb = read binary

with open(os.path.join(models_dir, 'model_a.pkl'), 'rb') as f:
    model_a = pickle.load(f)

with open(os.path.join(models_dir, 'model_b.pkl'), 'rb') as f:
    model_b = pickle.load(f)

# data structure
class SessionFeatures(BaseModel):
    view_count: int
    cart_count: int
    unique_products: int
    avg_price: float
    session_duration: float
    
# prediction endpoints
@app.post("/predict/model_a")
def predict_modela(data: SessionFeatures):
    # JSON to dataframe
    # data.model_dump() converts pydantic to dict
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model_a.predict(input_df)[0]
    probability = model_a.predict_proba(input_df)[0][1]
    
    return {
    "model_used": "Logistic_Regression_A",
    "will_purchase": int(prediction),
    "purchase_probability": float(probability)
    }
    # JSON (Web) ➔ Dictionary (Python) ➔ DataFrame (Pandas) ➔ Prediction (ML Model) ➔ JSON (Response)
    
@app.post("/predict/model_b")
def predict_modelb(data: SessionFeatures):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model_b.predict(input_df)[0]
    probability = model_b.predict_proba(input_df)[0][1]
    return {
        "model_used": "Random_forest_B",
        "will_purchase": int(prediction),
        "purchase_probability": float([probability])
    }
    