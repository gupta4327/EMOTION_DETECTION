from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

class ModelFactory:
    
    @staticmethod
    def get_model(model: str):

        models_map = {"randomforest": RandomForestClassifier,
                       "gradientboost": GradientBoostingClassifier}
        try:
            if model not in models_map:
                raise ValueError(f"Unsupported model passed. We support following models : {models_map.keys()}")
            return models_map[model]
        except Exception as e:
            raise e
        
    @staticmethod
    def model_params(model_name:str, model):
        
        model_params_map = {"randomforest": {"n_estimators": model.n_estimators, "max_depth":model.max_depth},
                       "gradientboost": {"n_estimators": model.n_estimators, "max_depth":model.max_depth}}
        
        try:
            return model_params_map[model_name]
        except Exception as e:
            raise e

   

class ModelBuilding:
    def __init__(self, model: str, hyperparameters=None):
        try:
            self.model_cls = ModelFactory().get_model(model)
            if hyperparameters:
                self.model = self.model_cls(**hyperparameters)
            else:
                self.model = self.model_cls()
        except Exception as e:
            raise e

    def fit(self, data_loader, output_col):
        try:
            data = data_loader.load_data()
            Y_train = data[output_col]
            X_train = data.drop(columns=[output_col])
            self.model.fit(X_train, Y_train)
            return self.model
        except Exception as e:
            raise e

    def get_model(self):
        return self.model


    def save_model(self, path):
        try:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
        except Exception as e:
            raise e


class ModelEvaluation:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):

        """Load the trained model from a file."""
        try:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Existing file : {self.model_path} does not exist")
        
        except Exception as e:           
            raise

    
    def evaluate(self, data_loader, output_col):

        data = data_loader.load_data()
        X = data.drop(columns = [output_col])
        y_true = data[output_col]
        try:
            y_pred = self.model.predict(X)
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0)
            }
        except Exception as e:
            raise e
        
    def get_model(self):
        if not self.model:
            self.load_model()
        return self.model
    
    def save_metrics(self, metrics, file_path):
        """Save the evaluation metrics to a JSON file."""
        try:
            with open(file_path, 'w') as file:
                json.dump(metrics, file, indent=4)
        except Exception as e:
            raise RuntimeError('Error occurred while saving the metrics: %s', e)

            
