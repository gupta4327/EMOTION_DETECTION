import os
from abc import ABC
import pandas as pd

class DataLoader(ABC):

    def load_data(self) -> pd.DataFrame:
        pass

class DataLinkLoader(DataLoader):

    def __init__(self, data_link:str)->None:

        """Sets up the link for loading the data."""
    

        if data_link == None:
            raise ValueError("Data link cannot be None")
        
        if not isinstance(data_link,str):
            raise ValueError("Data link must be String")
        
        self.data_link = data_link


    def load_data(self)->pd.DataFrame:

        """Returns the pandas dataframe of the data loaded from the provided url"""
        try:
            data = pd.read_csv(self.data_link)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load data with error {e}")
        
class LocalStorageLoader(DataLoader):

    def __init__(self, load_path)->pd.DataFrame:

        if not isinstance(load_path, str):
            raise ValueError("Load path must be a string path of data")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Specified file : {load_path} does not exists")
        
        self.load_path = load_path
      
    def load_data(self):

        """Returns the pandas dataframe of the data loaded from the provided local path"""
        try:
            data = pd.read_csv(self.load_path)
            return data
        
        except Exception as e:
            raise RuntimeError(f"Failed to load data with error {e}")
