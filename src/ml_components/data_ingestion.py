from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

import os, sys, pandas as pd

from src.ml_components import DATA_INGESTION_CONFIG
from src.utils import create_directory_through_file_path

logging = CustomLogger().get_logger(__name__)

@dataclass
class DataIngestionConfig:
    train_data_path: str = DATA_INGESTION_CONFIG["dataset"]["train"]
    test_data_path: str = DATA_INGESTION_CONFIG["dataset"]["test"]
    raw_data_path: str = DATA_INGESTION_CONFIG["dataset"]["raw"]

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            df.drop(['date','country'], axis=1, inplace=True)
            logging.info("Read the dataset as dataframe")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train test split completed")

            create_directory_through_file_path(self.ingestion_config.test_data_path)
            create_directory_through_file_path(self.ingestion_config.train_data_path)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            app_exc = CustomException(e, sys)
            logging.error(str(app_exc))
            raise app_exc

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
