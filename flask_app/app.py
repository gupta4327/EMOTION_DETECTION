# updated app.py

from flask import Flask, render_template,request
import mlflow
import pickle
import os
import pandas as pd

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


try:
    aws_tracking_uri = os.getenv("AWS_TRACKING_URI")
    
except Exception as e:
    raise RuntimeError(f"Failed in initializing the parameter with error : {e}")

try:
    mlflow.set_tracking_uri(aws_tracking_uri)

except Exception as e:
    raise EnvironmentError(f"Failed in setting up the aws environment for Mlflow with error : {e}")

app = Flask(__name__)

# load model from model registry
def get_latest_version(model_name, alias):
    client_mlflow = mlflow.tracking.MlflowClient()
    latest_version = client_mlflow.get_model_version_by_alias(name=model_name, alias=alias)
    if latest_version:
        run_id = latest_version.run_id
        return latest_version.version, run_id
    else: 
        return None,None
    
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

try:    
    model_name = "emotion_detector_model"
    model_version, run_id = get_latest_version(model_name, "production")

    model_uri = f'models:/{model_name}/{model_version}'
    model = mlflow.pyfunc.load_model(model_uri)

    vectorizer_name = "text_vectorizer.pkl" 
    encoder_name = "label_encoder.pkl"
    vectorizer, encoder = get_vectorizer_and_encoder(vectorizer_name, encoder_name, run_id)
    vectorizer = pickle.load(open('models/text_vectorizer.pkl','rb'))
except Exception as e:
    raise RuntimeError(f"Failed in model loading from MLflow with error : {e}")

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(
                                features.toarray(), 
                                columns=vectorizer.get_feature_names_out())


    # prediction
    result = model.predict(features_df)

    print(f" before convert :{result}")

    print(encoder.classes_)


    result = encoder.inverse_transform(result)

    print(f" after convert :{result}")

    # show
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)