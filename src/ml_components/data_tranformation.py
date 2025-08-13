from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataclasses import dataclass

import sys, numpy as np, pandas as pd

from src.ml_components import PREPROCESSOR_CONFIG
from src.utils import  X_y_split, save_object


logging = CustomLogger().get_logger(__file__)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = PREPROCESSOR_CONFIG["path"]
    target_column = PREPROCESSOR_CONFIG["target_column"]

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self, numerical_columns, categorical_columns)->ColumnTransformer:
        try:
            logging.info("Creating preprocessor object ...")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Created pipeline for Numerical columns")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                ]
            )

            logging.info(f"Created pipeline for Categorical columns")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            logging.info(f"Created preprocessor object")

            return preprocessor

        except Exception as e:
            app_exc = CustomException(e, sys)
            logging.error(str(app_exc))
            raise app_exc
    
    def initiate_data_transformation(self, train_path, test_path)->tuple[np.ndarray, np.ndarray, str]:
        try:
            logging.info("Data transformation started")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded Train and Test dataset")

            input_feature_train_df, target_feature_train_df = X_y_split(train_df, self.data_transformation_config.target_column)
            input_feature_test_df, target_feature_test_df = X_y_split(test_df, self.data_transformation_config.target_column)


            preprocessor = self.get_data_transformer_object(
                numerical_columns=input_feature_train_df.select_dtypes(exclude=["object"]).columns,
                categorical_columns=input_feature_train_df.select_dtypes(include=["object"]).columns
            )

            logging.info("Created preprocessor object")

            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Data Transformed suceesfully")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            return(
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            app_exc = CustomException(e, sys)
            logging.error(str(app_exc))
            raise app_exc

if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(
        train_path="notebooks/data/train.csv",
        test_path="notebooks/data/test.csv"
    )

    print(train_arr.shape)
    print(test_arr.shape)
    print(preprocessor_obj_file_path)


