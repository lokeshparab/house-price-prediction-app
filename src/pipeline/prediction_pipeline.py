from src.ml_components import MODEL_TRAINING_CONFIG, PREPROCESSOR_CONFIG
from src.utils import load_object

from exception.custom_exception import CustomException
from logger.custom_logger import CustomLogger
import sys, pandas as pd


logging = CustomLogger().get_logger(__file__)

class PredictionPipeline:
    def __init__(self):
        self.model = load_object(MODEL_TRAINING_CONFIG["path"])
        self.preprocessor = load_object(PREPROCESSOR_CONFIG["path"])

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Prediction the results")
            transformed_features = self.preprocessor.transform(features)
            logging.info("Prediction completed")
            return self.model.predict(transformed_features)
            
        except Exception as e:
            app_exc = CustomException(e, sys)
            logging.error(str(app_exc))
            raise app_exc
        
class HousePricePredictorDataset:
    def __init__(
        self,
        bedrooms: int, bathrooms: float, living_room_sqft: float, lot_sqft: float, floors: float, waterfront: int, view: int, condition: int, sqft_above: float, sqft_basement: float, year_built: int, year_renovated: int
    ):
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.living_room_sqft = living_room_sqft
        self.lot_sqft = lot_sqft
        self.floors = floors
        self.waterfront = waterfront
        self.view = view
        self.condition = condition
        self.sqft_above = sqft_above
        self.sqft_basement = sqft_basement
        self.year_built = year_built
        self.year_renovated = year_renovated
    
    def get_dataframework(self)->pd.DataFrame:
        return pd.DataFrame({
            "bedrooms": [self.bedrooms],
            "bathrooms": [self.bathrooms],
            "sqft_living": [self.living_room_sqft],
            "sqft_lot": [self.lot_sqft],
            "floors": [self.floors],
            "waterfront": [self.waterfront],
            "view": [self.view],
            "condition": [self.condition],
            "sqft_above": [self.sqft_above],
            "sqft_basement": [self.sqft_basement],
            "year_built": [self.year_built],
            "year_renovated": [self.year_renovated]
        })
        