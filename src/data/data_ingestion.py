#necessary imports
from sklearn.model_selection import train_test_split
from src.data.dataload_classes import DataLinkLoader
from src.data.dataupl_classes import LocalStorageUploader
from dotenv import load_dotenv
import os
from src.utils.logger import get_logger
from src.utils.load_params import load_params

try:
    logger = get_logger()
except Exception as e:
    raise RuntimeError(f"Unable to initialize logger with error : {e}")

load_dotenv()


def prepare_data(df, drop_cols, test_size):
    df.drop(columns=drop_cols, inplace=True)
    df = df[df["sentiment"].isin(["happiness", "sadness"])]
    train_df, test_df = train_test_split(df, test_size=test_size,random_state=42)
    return train_df, test_df

#loading the data
try:
    data_loader = DataLinkLoader(os.getenv("DATA_LINK"))
    params = load_params("params.yaml")
    test_size = params["data_ingestion"]["test_size"]
    logger.debug(f"Intialized test size : {test_size}")
    raw_data_dir = os.getenv("RAW_DATA_DIR")
    logger.debug("Data loader and environment parameters initialized")
    df = data_loader.load_data()
    train_df, test_df = prepare_data(df, ["tweet_id"], test_size)
    logger.debug("train test split performed")
    train_df.dropna(inplace=True)
    test_df.dropna(inplace = True)
    logger.debug("Basic data loading and cleaning is done")
    logger.debug(f"Saving file in the folowing loc : {raw_data_dir}")
    uploader = LocalStorageUploader(raw_data_dir)
    uploader.upload_data(train_df, "train.csv")
    uploader.upload_data(test_df, "test.csv")
    logger.debug("FIle has been uploaded successfully in raw data folder")
except Exception as e:
    logger.error(f"process failed in Data ingestion step with error : {e}")