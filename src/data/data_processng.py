from src.data.dataload_classes import LocalStorageLoader
from src.data.dataupl_classes import LocalStorageUploader
from src.data.data_process_classes import TextDataPreProcessing
import os
from src.utils.logger import get_logger

try:
    logger = get_logger()
except Exception as e:
    raise RuntimeError(f"Unable to initialize logger with error : {e}")

#env parameters
try:
    raw_dir = os.getenv("RAW_DATA_DIR")
    preprocess_dir = os.getenv("PREPROCESS_DIR")
    raw_train_path = os.path.join(raw_dir, "train.csv")
    raw_test_path = os.path.join(raw_dir, "test.csv")
    preprcs_train_path = os.path.join(preprocess_dir, "preprocess_train.csv")
    preprcs_test_path = os.path.join(preprocess_dir, "preprocess_test.csv")
except Exception as e:
    logger.error(f"Failed in file {__name__} for initialization of paths and parameters")

#new file data preprocess.py
try:
    raw_train_loader = LocalStorageLoader(raw_train_path)
    raw_test_loader = LocalStorageLoader(raw_test_path)
    logger.debug(f"Data loading step in file : {__name__} processed successfully")
    text_preprocessor = TextDataPreProcessing(raw_train_loader, raw_test_loader)
    train_df, test_df = text_preprocessor.preprocess("content")
    train_df.dropna(inplace=True)
    test_df.dropna(inplace = True)
    logger.debug(f"Data preprocessing completed")
    uploader = LocalStorageUploader(preprocess_dir)
    uploader.upload_data(train_df, "preprocess_train.csv")
    uploader.upload_data(test_df, "preprocess_test.csv")
    logger.debug("Preprocessed Data uploaded successfully")
except Exception as e:
    logger.error(f"Failed in file {__name__} in preprocessing step with error : {e}")
