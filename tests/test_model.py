import unittest
import os
import mlflow
import mlflow.artifacts
import mlflow.tracking
import pickle
import pandas as pd
from src.data.dataload_classes import LocalStorageLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from dotenv import load_dotenv

load_dotenv()

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

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
        
        cls.new_model_name = "emotion_detector_model"
        cls.latest_version, cls.run_id = cls.get_latest_version(model_name= cls.new_model_name)
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.latest_version}"
        cls.new_model = mlflow.pyfunc.load_model(model_uri=cls.new_model_uri, )

        vectorizer_name = "text_vectorizer.pkl" 
        encoder_name = "label_encoder.pkl"
        cls.vectorizer, cls.encoder = cls.get_vectorizer_and_encoder(vectorizer_name, encoder_name, cls.run_id)

        processed_dir = os.getenv("PROCESSED_DIR")
        processed_test_path = os.path.join(processed_dir, "processed_test.csv")
        processed_test_loader = LocalStorageLoader(processed_test_path)

        cls.holdout_data = processed_test_loader.load_data()

    
    @staticmethod
    def get_latest_version(model_name, alias = "staging"):
        client_mlflow = mlflow.tracking.MlflowClient()
        latest_version = client_mlflow.get_model_version_by_alias(name=model_name, alias=alias)
        if latest_version:
            run_id = latest_version.run_id
            return latest_version.version, run_id
        else: 
            return None, None
    
    @staticmethod
    def get_vectorizer_and_encoder(vectorizer_name, encoder_name, run_id):

        try:
            vectorizer_path = mlflow.artifacts.download_artifacts(artifact_path=vectorizer_name, run_id=run_id)
            encoder_path = mlflow.artifacts.download_artifacts(artifact_path=encoder_name, run_id=run_id)

            with open(vectorizer_path, "rb") as f:
                text_vectorizer = pickle.load(f)
            
            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)

            if text_vectorizer:

                if label_encoder:
                    return text_vectorizer, label_encoder
                else:
                    raise RuntimeError(f"Cannot load the label encoder")
            else:
                raise RuntimeError(f"Cannot load the text vectorizer")
            
        except Exception as e:
            raise RuntimeError(f"Failed in getting vectorizer and encoder step with error : {e}")

        
    def test_loadmodel(self):
        self.assertIsNotNone(self.new_model)


    def test_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame.sparse.from_spmatrix(input_data)
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_names_out())

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self): 
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.drop(columns=["sentiment"])
        y_holdout = self.holdout_data["sentiment"]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.60
        expected_precision = 0.60
        expected_recall = 0.60
        expected_f1 = 0.60

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()


