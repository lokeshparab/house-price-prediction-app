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
            price = self.model.predict(transformed_features)[0]
            return round(price, 2)
            
        except Exception as e:
            app_exc = CustomException(e, sys)
            logging.error(str(app_exc))
            raise app_exc
        
class HousePricePredictorDataset:
    def __init__(
        self,
        price_per_sqft:float, area:float,
        bedRoom: int, bathroom: int,
        floorNum: int
    ):
        self.price_per_sqft = price_per_sqft
        self.area = area
        self.bedRoom = bedRoom
        self.bathroom = bathroom
        self.floorNum = floorNum
    
    def get_dataframework(self)->pd.DataFrame:
        return pd.DataFrame({
            "price_per_sqft": [self.price_per_sqft],
            "area": [self.area],
            "bedRoom": [self.bedRoom],
            "bathroom": [self.bathroom],
            "floorNum": [self.floorNum],
        })
    
    def model_dump(self):
        return {
            "price_per_sqft": self.price_per_sqft,
            "area": self.area,
            "bedRoom": self.bedRoom,
            "bathroom": self.bathroom,
            "floorNum": self.floorNum,
        }
    
if __name__ == "__main__":
    predictor = PredictionPipeline()
    dataset = HousePricePredictorDataset(
        price_per_sqft=1000,
        area=1000,
        bedRoom=2,
        bathroom=2,
        floorNum=2
    )
    print(predictor.predict(dataset.get_dataframework()))
        