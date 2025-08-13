from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from omegaconf import OmegaConf
from hydra.utils import instantiate

from datetime import datetime

from dataclasses import dataclass
from tqdm import tqdm

import sys, numpy as np, pandas as pd, warnings

from src.ml_components import MODEL_TRAINING_CONFIG, MODEL_CONFIGS
from src.utils import  add_timestamp_to_filename, save_object, create_directory_through_file_path

logging = CustomLogger().get_logger(__file__)

warnings.filterwarnings("ignore")

# model_config = OmegaConf.load("config/model_config.yaml")

# MODEL_CONFIGS = {}
# for k, v in model_config.items():
#     MODEL_CONFIGS[k] = {
#         'model': instantiate(model_config[k]),
#         'params':v['params'] if v['params'] is not None else {}
#     }

@dataclass
class ModelTrainingConfig:
    trained_model_file_path:str = MODEL_TRAINING_CONFIG["path"]
    model_report_file_path:str = MODEL_TRAINING_CONFIG["report_path"]  

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
        self.model_configs = MODEL_CONFIGS
    
    def  initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Model Training Started")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1], 
                test_arr[:, :-1], test_arr[:, -1]
            )

            model_report = {
                'model_name': [],
                'best_model_parmas':[],
                'model':[],
                'r2_score':[],
                'mean_squared_error':[],
                'mean_absolute_percentage_error':[],
            }

            logging.info(f"Started Tuning  for {len(self.model_configs.keys())} models")
            for model_name, model_config in tqdm(self.model_configs.items()):
                
                # Instantiate the model using Hydra's instantiate function
                model = model_config['model']
                params = model_config['params']
                logging.info(f"{model}|{params}")
                logging.info(f"{type(model)}|{type(params)}")
                logging.info(f"Started Tuning for {model_name}-{type(model)} for parmas {params} ")

                grid = RandomizedSearchCV(
                    estimator=model,  # Pass the instantiated model object
                    param_distributions=params,  # Pass the parameters directly
                    cv=5, scoring='accuracy', refit=True, verbose=False,
                )
                grid.fit(X_train, y_train)

                y_pred = grid.predict(X_test)

                model_report['model_name'        ].append(model_name)
                model_report['best_model_parmas' ].append(grid.best_params_)    
                model_report['model'             ].append(grid.best_estimator_)
                model_report['mean_absolute_percentage_error'    ].append(mean_absolute_percentage_error(y_test, y_pred))
                model_report['r2_score'          ].append(r2_score(y_test, y_pred))
                model_report['mean_squared_error'].append(mean_squared_error(y_test, y_pred))
            
            logging.info("Model Tuning Completed")

            model_report = pd.DataFrame(model_report)
            model_report.sort_values('r2_score', ascending= False , inplace= True)
            create_directory_through_file_path(self.model_training_config.model_report_file_path)
            model_report.to_csv(
                add_timestamp_to_filename(self.model_training_config.model_report_file_path),
                index=False,
            )

            logging.info("Model Report Saved")

            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=grid.best_estimator_
            )

            logging.info("Model Saved Sucessfully")

            logging.info("Model Training Completed")

            if model_report.iloc[0]['r2_score'] < 0.6:
                raise CustomException(
                    "Model Training accuracy is very low below 0.6"
                    f"\nPlease check the report {self.model_training_config.model_report_file_path}",
                    sys
                )
        
            return (
                self.model_training_config.trained_model_file_path,
                self.model_training_config.model_report_file_path
            )

        except Exception as e:
            app_exc = CustomException(e, sys)
            logging.error(str(app_exc))
            raise app_exc