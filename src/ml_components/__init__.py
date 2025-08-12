import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["data_ingestion"]
    f.close()

DATA_INGESTION_CONFIG = config['dataset']
PREPROCESSOR_CONFIG = config['preprocessor']
MODEL_TRAINING_CONFIG = config['model']