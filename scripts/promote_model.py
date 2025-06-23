import os 
import mlflow
import mlflow.tracking
from src.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

try:
    logger = get_logger()
except Exception as e:
    raise RuntimeError(f"Unable to initialize logger with error : {e}")

try:
    aws_tracking_uri = os.getenv("AWS_TRACKING_URI")
    if aws_tracking_uri is None:
        raise EnvironmentError("AWS tracking URI cannot be none")
except Exception as e:
    raise RuntimeError(f"Failed in initializing aws tracking url with error : {e}")

try: 
    mlflow.set_tracking_uri(aws_tracking_uri)
except Exception as e: 
    raise RuntimeError("Failed in setting up mlflow with aws tracking URI")

def promote_model():

       model_name = "emotion_detector_model"
       latest_version_stg = get_latest_version(model_name, "staging")
       latest_version_prod = get_latest_version(model_name, "production")
       
       if latest_version_prod == None or latest_version_stg == None:
           raise RuntimeError("Not able to fetch latest model from mlflow")
       
       client = mlflow.tracking.MlflowClient()

       client.set_registered_model_alias(
            name=model_name,
            alias="archived",  # replaces "Staging" stage
            version=latest_version_prod
        )
       
       client.set_registered_model_alias(
            name=model_name,
            alias="production",  # replaces "Staging" stage
            version=latest_version_stg
        )
    
def get_latest_version(model_name, alias):
    client_mlflow = mlflow.tracking.MlflowClient()
    latest_version = client_mlflow.get_model_version_by_alias(name=model_name, alias=alias)
    if latest_version:
        return latest_version.version
    else: 
        return None
    
if __name__ == "__main__":
    promote_model()