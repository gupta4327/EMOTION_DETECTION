from abc import ABC
from src.data.dataload_classes import DataLoader
import inspect
from nltk import WordNetLemmatizer
import re
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

class DataPreProcessing(ABC):

    def __init__(self, train_data_loader:DataLoader,test_data_loader:DataLoader)->None:
            
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


    def preprocess(self, col:str, tool_flow:list[str])->pd.DataFrame:

        pass
        
        
    def available_tools(self) ->list:
        
        """Return a list of available preprocessing methods in the class."""

        try:
            methods = [
                name for name, func in inspect.getmembers(self, predicate=inspect.ismethod)
                if func.__self__.__class__ == self.__class__ and not name.startswith("__")
                and name not in ["available_tools", "preprocess"]
            ]
            return methods
        
        except AttributeError as e:
            raise RuntimeError(f"Failed to inspect methods for {self.__class__.__name__}: {e}")
        
        except Exception as e:
            raise RuntimeError(f"Unexpected error in available_tools: {e}")
        
    def get_data(self):
        return self.train_data, self.test_data
    

class TextDataPreProcessing(DataPreProcessing):

    def __init__(self, train_data_loader:DataLoader, test_data_loader:DataLoader)->None:

        """Initialize the TextDataPreProcessing class with data loaders."""
        super().__init__(train_data_loader, test_data_loader)

    def lemmatization(self, text:str)->str:

        """Lemmatize the text."""
        
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)

    def remove_stop_words(self, text:str)->str:
        
        """Remove stop words from the text."""
        
        stop_words = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in stop_words]
        return " ".join(text)

    def remove_numbers(self, text:str)->str:
        
        """Remove numbers from the text."""
        
        text = ''.join([char for char in text if not char.isdigit()])
        return text

    def lower_case(self, text:str)->str:
        
        """Convert text to lower case."""
        
        text = text.split()
        text = [word.lower() for word in text]
        return " ".join(text)

    def remove_punctuations(self, text:str)->str:
        
        """Remove punctuations from the text."""
        
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        return text

    def remove_urls(self, text:str)->str:
        
        """Remove URLs from the text."""
        
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_small_sentences(self, text:str):
        
        """Remove sentences with less than 3 words."""
        
        if len(text.split()) < 3:
                return np.nan
        return text

    def available_tools(self):
        return super().available_tools()

    def preprocess(self, col:str, tool_flow=["lower_case", "lemmatization", "remove_stop_words", "remove_numbers", "remove_punctuations", "remove_urls"]) -> pd.DataFrame:
        
        """Preprocess the data using the specified tools in the tool_flow."""
        try:
            
            #tool validations
            if not isinstance(tool_flow, list):
                raise ValueError("tool_flow must be a list of tool names.")
            
            if not all(tool in self.available_tools() for tool in tool_flow):
                raise ValueError("One or more tools in tool_flow are not available in the preprocessing class.")
            
            #column validations
            if not isinstance(col, str):
                raise ValueError("col must be a string representing the column name.")
            
            if col not in self.train_data.columns and col not in self.test_data.columns:
                raise ValueError(f"Column '{col}' does not exist in both Training and Test DataFrames.")
            

            for tool in tool_flow:
                method = getattr(self, tool)
                self.train_data[col] = self.train_data[col].apply(method)
                self.test_data[col] = self.test_data[col].apply(method)

            return self.train_data, self.test_data
        
        except Exception as e:
            raise RuntimeError(f"An error occurred during preprocessing: {e}")
        
    def get_data(self):
        return super().get_data()
    