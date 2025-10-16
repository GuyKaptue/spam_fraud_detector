# spam_fraud_detector/core/data_loader.py
# type: ignore
import os
import pandas as pd 
import kagglehub 
from kagglehub import KaggleDatasetAdapter 
from spam_fraud_detector.core.utils import get_dataset_config, ensure_dir, save_dataframe

class KaggleDataLoader:
    def __init__(self, task: str, adapter=KaggleDatasetAdapter.PANDAS, save_dir: str = "datasets"):
        """
        Initialize the loader with a task name ('spam' or 'fraud'), adapter, and save directory.
        """
        config = get_dataset_config(task)
        self.task = task
        self.dataset_name = config["dataset_name"]
        self.file_path = config["file_path"]
        self.adapter = adapter
        self.save_dir = os.path.join(save_dir, task)
        ensure_dir(self.save_dir)

    def download_dataset(self) -> str:
        """
        Downloads the dataset from Kaggle.
        """
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            print(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            print(f"Download failed: {e}")
            return ""

    def load_dataset(self) -> pd.DataFrame | None:
        """
        Loads a specific file from the dataset.
        """
        try:
            data = kagglehub.dataset_load(self.adapter, self.dataset_name, path=self.file_path)
            print(f"Loaded file: {self.file_path}")
            return data
        except Exception as e:
            print(f"Loading failed: {e}")
            return None

    def save_data(self, data: pd.DataFrame, filename: str, format: str = "csv") -> str:
        """
        Saves the loaded data to the specified format (csv or xlsx).
        """
        save_path = os.path.join(self.save_dir, filename)
        return save_dataframe(data, save_path, format=format)

    def download_load_and_save(self, save_as: str = "csv") -> pd.DataFrame | None:
        """
        Downloads, loads, and saves the dataset file in the desired format.
        """
        if not self.download_dataset():
            return None

        data = self.load_dataset()
        if data is not None:
            filename = os.path.splitext(self.file_path)[0] + f".{save_as}"
            self.save_data(data, filename, format=save_as)
        return data
