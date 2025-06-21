from src.data.dataload_classes import LocalStorageLoader
from src.model.model_classes import ModelEvaluation
from dotenv import load_dotenv
import os 
import mlflow
from src.utils.logger import get_logger
import json
from utils.load_params import load_params
 
load_dotenv()

try:
    logger = get_logger()
except Exception as e:
    raise RuntimeError("Failed in initializing the logs file")


try:
    
    processed_dir = os.getenv("PROCESSED_DIR")
    models_dir = os.getenv("MODELS_DIR")
    model_save_path = os.getenv(models_dir, "emotion_detector_model.pkl")
    processed_test_path = os.path.join(processed_dir, "processed_test.csv")
    processed_train_path = os.path.join(processed_dir, "processed_train.csv")
    aws_tracking_url = os.getenv("AWS_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME")
    reports_dir = os.getenv("REPORTS_DIR")
    metrics_file_path = os.path.join(reports_dir, "metrics.json")
    exp_info_json_path  = os.path.join(reports_dir, "experiment_info.json") 

except Exception as e:

    logger.error(f"Failed in initializing path or variables with error : {e}")
    raise RuntimeError(f"Failed in initializing path or variables with error : {e}")


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


try:
    
    mlflow.set_tracking_uri(aws_tracking_url)
    mlflow.set_experiment(experiment_name="emotion_detection_pipeline")

except Exception as e:

    raise EnvironmentError(f"Failed in setting up the aws environment for Mlflow with error : {e}")


try:

    processed_test_loader = LocalStorageLoader(processed_test_path)
    processed_train_loader = LocalStorageLoader(processed_train_path)
    evaluator = ModelEvaluation(model_save_path)
    evaluator.load_model()

except Exception as e:

    logger.error(f"failed in data loading and model loading with error : {e}")
    raise e

try:

    with mlflow.start_run() as run:

        evaluation_train_dict = evaluator.evaluate(processed_train_loader, "sentiment")
        evaluation_test_dict = evaluator.evaluate(processed_test_loader, "sentiment")

        evaluation_dict = {"train":evaluation_train_dict, "test": evaluation_test_dict}

        evaluator.save_metrics(evaluation_dict, metrics_file_path)

        for outer_key, inner_dict in evaluation_dict.items():
            for inner_key, val in inner_dict.items():
                mlflow.log_metric(f"{outer_key}_{inner_key}", val)

        fit_model = evaluator.get_model()

        save_model_info(run_id=run.info.run_id, model_path="model", file_path=exp_info_json_path)

                    # Log model parameters to MLflow
        if hasattr(fit_model, 'get_params'):
            params = fit_model.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

        # Log model to MLflow
        mlflow.sklearn.log_model(fit_model, "model")
        
        # Log the metrics file to MLflow
        mlflow.log_artifact(metrics_file_path)

        # Log the model info file to MLflow
        mlflow.log_artifact(exp_info_json_path)

except Exception as e:
    logger.error(f"Failed in model evaluation step with error : {e}")
    raise e


    

    

