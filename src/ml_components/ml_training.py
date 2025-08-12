from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from dataclasses import dataclass

import sys, numpy as np, pandas as pd

from src.ml_components import MODEL_TRAINING_CONFIG
from src.utils import  X_y_split, save_object

logging = CustomLogger().get_logger(__name__)


@dataclass
class ModelTrainingConfig:
    trained_model_file_path:str = MODEL_TRAINING_CONFIG["path"]

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
    
    def initiate_model_training(self, train_arr, test_arr, target_arr):
        try:
            logging.info("Model Training Started")

            
