from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn, numpy as np, os, sys

from models.model import HouseFeatures

from src.pipeline.prediction_pipeline import PredictionPipeline, HousePricePredictorDataset
from exception.custom_exception import CustomException
from logger.custom_logger import CustomLogger

app = FastAPI(title="House Price Prediction API", description="Predict house prices using machine learning")

# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files BEFORE setting up templates
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates with custom functions
templates = Jinja2Templates(directory="templates")

# Add url_for function to Jinja2 environment
def url_for(request: Request, name: str, **path_params):
    if name == "static":
        # Handle static files
        filename = path_params.get("filename", "")
        return f"/static/{filename}"
    else:
        # Handle other routes
        return request.url_for(name, **path_params)

# Add the url_for function to Jinja2 globals
templates.env.globals["url_for"] = url_for

predictor = PredictionPipeline()

logging = CustomLogger().get_logger(__file__)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    logging.info("Loading home page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    bedRoom: int = Form(...),
    bathroom: int = Form(...),
    area: float = Form(...),
    price_per_sqft: float = Form(...),
    floorNum: int = Form(...),
):
    try:
        logging.info("Received request from UI for prediction")
        dataset = HousePricePredictorDataset(
            price_per_sqft=price_per_sqft,
            area=area,
            bedRoom=bedRoom,
            bathroom=bathroom,
            floorNum=floorNum
        )
        
        predicted_price = predictor.predict(dataset.get_dataframework())

        logging.info("Send predicted result to UI")
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": predicted_price,
            "form_data": dataset.model_dump()
        })
    
    except Exception as e:
        # Handle prediction errors gracefully
        app_exc = CustomException(e, sys)
        logging.error(str(app_exc))
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Prediction failed: {str(app_exc)}",
            "form_data": {
                "bedRoom": bedRoom,
                "bathroom": bathroom,
                "area": area,
                "price_per_sqft": price_per_sqft,
                "floorNum": floorNum
            }
        })

@app.get("/api/predict")
async def api_predict(features: HouseFeatures):
    """API endpoint for programmatic access"""
    try:
        logging.info("Received request from API for prediction")
        dataset = HousePricePredictorDataset(
            **features.model_dump()
        )
        predicted_price = predictor.predict(dataset.get_dataframework())
        logging.info("API Prediction completed")
        return {"predicted_price": predicted_price, "status": "success"}
    except Exception as e:
        app_exc = CustomException(e, sys)
        logging.error(str(app_exc))
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)