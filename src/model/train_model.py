from src.data.dataload_classes import LocalStorageLoader
from src.model.model_classes import ModelBuilding
from dotenv import load_dotenv
import os
from src.utils.logger import get_logger
from src.utils.load_params import load_params
load_dotenv()


try:
    logger = get_logger()
except Exception as e: 
    raise RuntimeError("Failed in initializing the logs file")

try:
    processed_dir = os.getenv("PROCESSED_DIR")
    models_dir = os.getenv("MODELS_DIR")
    model_save_path = os.getenv(models_dir, "emotion_detector_model.pkl")
    processed_train_path = os.path.join(processed_dir, "processed_train.csv")
    params = load_params("params.yaml")
    model_name = params["train_model"]["model"]
    hyperparameters = dict(params["train_model"])
    hyperparameters.pop("model")
except Exception as e:
    logger.error(f"Failed in initializing the parameters in {__name__} with error : {e}")
    raise RuntimeError(f"Failed in initializing the parameters in {__name__} with error : {e}")

try:
    processed_train_loader = LocalStorageLoader(processed_train_path)
    model = ModelBuilding(model_name,hyperparameters=hyperparameters)
    model.fit(processed_train_loader, "sentiment")
    model.save_model(model_save_path)

except Exception as e:
    logger.error(f"Failed in model training with error : {e}")
    raise e


