# import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate


config = OmegaConf.load("config/config.yaml")
model_config = OmegaConf.load("config/model_config.yaml")

DATA_INGESTION_CONFIG = config['dataset']
PREPROCESSOR_CONFIG = config['preprocessor']
MODEL_TRAINING_CONFIG = config['model']

MODEL_CONFIGS = {}
for k, v in model_config.items():
    MODEL_CONFIGS[k] = dict(instantiate(model_config[k]))

    if MODEL_CONFIGS[k]['params']:
        MODEL_CONFIGS[k]['params'] = {
            str(k): list(v)
            for k, v in model_config[k]['params'].items()
        }
    else:
        MODEL_CONFIGS[k]['params'] = {}