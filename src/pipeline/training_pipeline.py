
from src.ml_components.data_ingestion import DataIngestion
from src.ml_components.data_tranformation import DataTransformation
from src.ml_components.ml_training import ModelTraining

from exception.custom_exception import CustomException
from logger.custom_logger import CustomLogger
import sys

logging = CustomLogger().get_logger(__file__)

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_training = ModelTraining()

    def initiate_training_pipeline(self):
        try:
            logging.info("Training pipeline started")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            train_arr, test_arr, preprocessor_obj_file_path = self.data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            trained_model_file_path , model_report_file_path = self.model_training.initiate_model_training(train_arr, test_arr)

            logging.info("Training pipeline completed")

            return trained_model_file_path, preprocessor_obj_file_path, model_report_file_path

        except Exception as e:
            app_exc = CustomException(e, sys)
            logging.error(str(app_exc))
            raise app_exc

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    trained_model_file_path, model_report_file_path = training_pipeline.initiate_training_pipeline()