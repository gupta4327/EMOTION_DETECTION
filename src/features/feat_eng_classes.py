from abc import ABC
from src.data.dataload_classes import DataLoader
import inspect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class FeatureEngineering(ABC):

    def __init__(self,train_data_loader:DataLoader, test_data_loader:DataLoader)->None:
        
        """Initialize the DataPreProcessing class with data loaders."""
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        
        #loader validation
        if not isinstance(self.train_data_loader, DataLoader) or not isinstance(self.test_data_loader, DataLoader):
            raise TypeError("train_data_loader and test_data_loader must be instances of DataLoader class.")
        
        if not hasattr(self.train_data_loader, 'load_data') or not hasattr(self.test_data_loader, 'load_data'):
            raise AttributeError("train_data_loader and test_data_loader must have a 'load_data' method.")
        
        if not callable(self.train_data_loader.load_data) or not callable(self.test_data_loader.load_data):
            raise TypeError("load_data must be a callable method in train_data_loader and test_data_loader.")
        
        try:
            self.train_data = self.train_data_loader.load_data()
            self.test_data = self.test_data_loader.load_data()
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
        
        #data validations
        if not isinstance(self.train_data, pd.DataFrame) or not isinstance(self.test_data, pd.DataFrame):
            raise TypeError("train_data and test_data must be pandas DataFrames.")
        
        if self.train_data.empty or self.test_data.empty:
            raise ValueError("train_data and test_data cannot be empty DataFrames.")
        
        if not all(col in self.train_data.columns for col in self.test_data.columns):
            raise ValueError("Test data contains columns that are not present in the training data.")
        
        if not all(col in self.test_data.columns for col in self.train_data.columns):
            raise ValueError("Training data contains columns that are not present in the test data.")
        
    
    def available_tools(self) -> list:
        """Return a list of available feature engineering methods in the class."""
        try:
            methods = [
                name for name, func in inspect.getmembers(self, predicate=inspect.ismethod)
                if func.__self__.__class__ == self.__class__ and not name.startswith("__")
                and name not in ["available_tools", "feature_engineering"]
            ]
            return methods
        
        except AttributeError as e:
            raise RuntimeError(f"Failed to inspect methods for {self.__class__.__name__}: {e}")
        
        except Exception as e:
            raise RuntimeError(f"Unexpected error in available_tools: {e}")
    
    def get_data(self)->tuple:
        return self.train_data, self.test_data
        

class TextFeatureEngineering(FeatureEngineering):

    def __init__(self, train_data_loader : DataLoader, test_data_loader :DataLoader):
        super().__init__(train_data_loader, test_data_loader)
       
        
    def apply_bow(self, col: str, max_features: int, return_vectorizer:bool=False) -> tuple:
        """
        Apply Count Vectorizer to the train and test data column.
        
        Parameters:
            col (str): Name of the text column.
            max_features (int): Max number of features for BOW.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Transformed train and test DataFrames with BOW features.
        """
        try:
            vectorizer = CountVectorizer(max_features=max_features)

            self.train_data.dropna(inplace=True)
            self.test_data.dropna(inplace=True)
            
            X_train = self.train_data[col].values
            X_test = self.test_data[col].values

            # Fit on train and transform both
            X_train_bow = vectorizer.fit_transform(X_train)
            X_test_bow = vectorizer.transform(X_test)

            # Convert to DataFrame with feature names
            feature_names = vectorizer.get_feature_names_out()
            X_train_bow_df = pd.DataFrame(X_train_bow.toarray(), columns=feature_names, index=self.train_data.index)
            X_test_bow_df = pd.DataFrame(X_test_bow.toarray(), columns=feature_names, index=self.test_data.index)

            # Update datasets
            self.train_data = pd.concat([self.train_data.drop(columns=[col]), X_train_bow_df], axis=1)
            self.test_data = pd.concat([self.test_data.drop(columns=[col]), X_test_bow_df], axis=1)

            return vectorizer

        except Exception as e:
            raise e
    
    def apply_tf_idf(self, col:str, return_vectorizer:bool= False):
        
        """Calculate the TF-IDF score for each word in the text."""
        
        self.train_data.dropna(inplace=True)
        self.test_data.dropna(inplace=True)
        
        vectorizer = TfidfVectorizer()
        X_train = self.train_data[col].values
        X_test = self.test_data[col].values

        
        # Fit on train and transform both
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        feature_names = vectorizer.get_feature_names_out()
        # Convert to DataFrame with feature names
        feature_names = vectorizer.get_feature_names_out()
        X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names, index=self.train_data.index)
        X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names, index=self.test_data.index)

        X_train_tfidf_df.drop(columns="sentiment", inplace=True)
        X_test_tfidf_df.drop(columns="sentiment", inplace=True)

        # Update datasets
        self.train_data = pd.concat([self.train_data.drop(columns=[col]), X_train_tfidf_df], axis=1)
        self.test_data = pd.concat([self.test_data.drop(columns=[col]), X_test_tfidf_df], axis=1)

        return vectorizer

    def label_encode(self, col:str, return_encoder:bool =False) :

        from sklearn.preprocessing import LabelEncoder

        lab_enc = LabelEncoder()

        self.train_data[col] = lab_enc.fit_transform(self.train_data[col])
        self.test_data[col] = lab_enc.transform(self.test_data[col])

        return lab_enc
    

        
        