from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from typing import Optional
import joblib  # assuming you're using joblib to save your model
import numpy as np
import os

from models.model import HouseFeatures

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="House Price Prediction API", description="Predict house prices using machine learning")

# Create directories if they don't exist
# os.makedirs("templates", exist_ok=True)
# os.makedirs("static/css", exist_ok=True)
# os.makedirs("static/js", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


# Load your trained model (replace with your actual model loading)
# model = joblib.load('your_model.pkl')

def predict_price(features: HouseFeatures) -> float:
    """
    Replace this function with your actual model prediction logic
    This is just a placeholder that returns a mock prediction
    """
    # Mock prediction for demonstration
    # Replace with: prediction = model.predict([[features.bedrooms, features.bathrooms, ...]])
    
    # Simple mock calculation for demo purposes
    base_price = 200000
    price = (base_price + 
             features.bedrooms * 15000 + 
             features.bathrooms * 10000 + 
             features.living_room_sqft * 100 +
             features.lot_sqft * 5 +
             features.floors * 5000 +
             features.waterfront * 100000 +
             features.view * 10000 +
             features.condition * 8000 +
             features.sqft_above * 80 +
             features.sqft_basement * 40 +
             max(0, 2024 - features.year_built) * -500 +
             (features.year_renovated > 0) * 20000)
    
    return max(price, 50000)  # Minimum price of $50,000

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    bedrooms: int = Form(...),
    bathrooms: float = Form(...),
    living_room_sqft: float = Form(...),
    lot_sqft: float = Form(...),
    floors: float = Form(...),
    waterfront: int = Form(...),
    view: int = Form(...),
    condition: int = Form(...),
    sqft_above: float = Form(...),
    sqft_basement: float = Form(...),
    year_built: int = Form(...),
    year_renovated: int = Form(0)
):
    
    features = HouseFeatures(
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        living_room_sqft=living_room_sqft,
        lot_sqft=lot_sqft,
        floors=floors,
        waterfront=waterfront,
        view=view,
        condition=condition,
        sqft_above=sqft_above,
        sqft_basement=sqft_basement,
        year_built=year_built,
        year_renovated=year_renovated
    )
    
    predicted_price = predict_price(features)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": predicted_price,
        "form_data": features.model_dump()
    })

@app.get("/api/predict")
async def api_predict(features: HouseFeatures):
    """API endpoint for programmatic access"""
    predicted_price = predict_price(features)
    return {"predicted_price": predicted_price}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)