from src.data.dataload_classes import LocalStorageLoader
from src.data.dataupl_classes import LocalStorageUploader
from src.features.feat_eng_classes import TextFeatureEngineering
from dotenv import load_dotenv
import os
import pickle
from src.utils.logger import get_logger
from src.utils.load_params import load_params
load_dotenv()

try:
    logger = get_logger()
except Exception as e: 
    raise RuntimeError("Failed in initializing the logs file")

try:
#parameters 
    preprocess_dir = os.getenv("PREPROCESS_DIR")
    processed_dir = os.getenv("PROCESSED_DIR")
    models_dir = os.getenv("MODELS_DIR")
    params = load_params("params.yaml")
    max_features = params["feature_engineering"]["max_features"]
    preprocess_train_path = os.path.join(preprocess_dir, "preprocess_train.csv")
    preprocess_test_path = os.path.join(preprocess_dir, "preprocess_test.csv")
except Exception as e:
    logger.error(f"Failed in initializing path or variables with error : {e}")
    raise RuntimeError(f"Failed in initializing path or variables with error : {e}")
    

try:
    preprocess_train_loader = LocalStorageLoader(preprocess_train_path)
    preprocess_test_loader = LocalStorageLoader(preprocess_test_path)
    logger.debug("Train and test loaders initialized")

    feature_eng = TextFeatureEngineering(preprocess_train_loader, preprocess_test_loader)
    vectorizer = feature_eng.apply_bow(col="content",return_vectorizer=True, max_features=max_features)
    encoder = feature_eng.label_encode("sentiment", return_encoder=True)
    logger.debug("Vectorization of text and encoding output col is done")

    train_df, test_df = feature_eng.get_data()
    uploader = LocalStorageUploader(processed_dir)
    uploader.upload_data(train_df, "processed_train.csv")
    uploader.upload_data(test_df, "processed_test.csv")
    logger.debug("Processed Data Uploaded successfully to the path ")

except Exception as e:
    logger.error(f"Failed in feature engineering step with error : {e}")
    raise e

try:
    vectorizer_path = os.path.join(models_dir, "text_vectorizer.pkl")
    label_encoder_path = os.path.join(models_dir, "label_encoder.pkl")

    pickle.dump(vectorizer, open(vectorizer_path, "wb"))
    pickle.dump(encoder, open(label_encoder_path, "wb"))

except Exception as e:
    logger.error(f"failed in saving the vectorizer and encoder with error : {e}")
    raise RuntimeError(f"failed in saving the vectorizer and encoder with error : {e}")



