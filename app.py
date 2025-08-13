from logger.custom_logger import CustomLogger

logging = CustomLogger().get_logger(__name__)

from src.pipeline.training_pipeline import TrainingPipeline

def main():
    print("Hello from house-price-prediction-app!")




if __name__ == "__main__":
    main()

    training_pipeline = TrainingPipeline()
    trained_model_file_path, model_report_file_path = training_pipeline.initiate_training_pipeline()
