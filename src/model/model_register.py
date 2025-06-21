import mlflow.tracking
from src.utils.logger import get_logger
from src.utils.load_params import load_params
import mlflow 
import os 
from dotenv import load_dotenv
import json

load_dotenv()

try:
    logger = get_logger()
except Exception as e:
    raise RuntimeError("Failed in initializing the logs file")

try:
    aws_tracking_uri = os.getenv("AWS_TRACKING_URI")
    reports_dir = os.getenv("REPORTS_DIR")
    model_info_path = os.path.join(reports_dir, "experiment_info.json")

except Exception as e:
    logger.error(f"Failed in initializing the parameter with error : {e}")
    raise RuntimeError(f"Failed in initializing the parameter with error : {e}")

try:
    mlflow.set_tracking_uri(aws_tracking_uri)

except Exception as e:
    raise EnvironmentError(f"Failed in setting up the aws environment for Mlflow with error : {e}")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name:str, model_info:dict):

    """rRegistering the model in model registry in the mlflow and transition it to stage"""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version = mlflow.register_model(model_uri=model_uri,name=model_name)

        client = mlflow.tracking.MlflowClient()

        client.set_registered_model_alias(
            name=model_name,
            alias="staging",  # replaces "Staging" stage
            version=model_version.version
        )

        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

try:
    model_info = load_model_info(model_info_path)
    model_name = "emotion_detector_model"
    register_model(model_name=model_name, model_info= model_info)
except Exception as e:
    logger.error('Error in model registery file: %s', e)
    raise




    

