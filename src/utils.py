import os, pickle
from pandas import DataFrame
from datetime import datetime

def add_timestamp_to_filename(file_path: str) -> str:
    base, ext = os.path.splitext(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"

def create_directory_through_file_path(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.makedirs(
            os.path.dirname(path_to_dir), exist_ok=True
        )

def X_y_split(df: DataFrame, target_column: str) -> tuple[DataFrame, DataFrame]:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y

def save_object(file_path, obj):
    try:
        create_directory_through_file_path(file_path)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise e

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise e