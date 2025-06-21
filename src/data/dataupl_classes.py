import os
from abc import ABC

class DataUploader(ABC):

    def upload_data(self, df, name):
        pass

class LocalStorageUploader(DataUploader):

    def __init__(self, upload_path):

        if isinstance(upload_path, str):
            self.upload_path = upload_path
        else:
            raise ValueError("Path for file must be a valid string")

    def upload_data(self, df, name):


        try:
            os.makedirs(self.upload_path, exist_ok=True)
            path = os.path.join(self.upload_path, name)
            print(path)
            df.to_csv(path, index=False)
        
        except Exception as e:

            raise RuntimeError(f"Failed in uploading the file {name} to path : {self.upload_path} with error {e}")

    