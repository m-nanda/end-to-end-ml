import joblib, warnings, os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import streamlit as st
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()

SAVED_OBJECTS_PATH = os.getenv("SAVED_OBJECTS_PATH") or st.secrets("SAVED_OBJECTS_PATH")
DATA_SOURCE_URL = os.getenv("DATA_SOURCE_URL") or st.secrets("DATA_SOURCE_URL")
DATA_PATH = os.getenv("DATA_PATH") or st.secrets("DATA_PATH")
PREDICTION_FILE = os.getenv("PREDICTION_FILE") or st.secrets("PREDICTION_FILE")

def fetch_data() -> pd.DataFrame:
    """
    Fetch realtime data from UCI Dataset Repository.

    Return:
        - raw_data (pd.DataFrame): The fetched data
    """
    try:
        col_names = [
            "sepal_length", 
            "sepal_width", 
            "petal_length", 
            "petal_width", 
            "species"
        ]
        raw_data = pd.read_csv(DATA_SOURCE_URL, names=col_names)
        raw_data.drop(columns="species", inplace=True)
    except:
        raw_data = pd.DataFrame()
    return raw_data

def load_objects() -> Tuple[
        StandardScaler, 
        MinMaxScaler, 
        LabelEncoder, 
        Dict[str, Any], 
        Dict[str, object]
    ]:
    """
    Load pre-saved machine learning objects including scaler, normalizer, encoder,
    feature engineering dictionary, and models.

    Returns:
        Tuple[StandardScaler, Normalizer, LabelEncoder, Dict[str, Any], Dict[str, object]]:
        - loaded_scaler: Scaler used to scale input data.
        - loaded_normalizer: Normalizer used to normalize input data.
        - loaded_encoder: Encoder used to decode model predictions.
        - loaded_features: Dictionary containing feature engineering instructions.
        - loaded_models: Dictionary of trained machine learning models.
    """

    loaded_scaler = joblib.load(f'{SAVED_OBJECTS_PATH}/scaler.bin') 
    loaded_normalizer = joblib.load(f'{SAVED_OBJECTS_PATH}/normalizer.bin') 
    loaded_encoder = joblib.load(f'{SAVED_OBJECTS_PATH}/encoder.bin') 
    loaded_features = joblib.load(f'{SAVED_OBJECTS_PATH}/features.bin') 
    loaded_models = joblib.load(f'{SAVED_OBJECTS_PATH}/models.bin') 

    return loaded_scaler, loaded_normalizer, loaded_encoder, loaded_features, loaded_models

def data_pipeline_v1(
        raw_input_df: pd.DataFrame, 
        scaler: MinMaxScaler
    ) -> pd.DataFrame:
    """
    Preprocess raw input data by normalizing it to be compatible with 
    model input requirements.

    Args:
        raw_input_df (pd.DataFrame): The raw input data in DataFrame format.
        scaler (MinMaxScaler): Pre-trained scaler to normalize the data.

    Returns:
        pd.DataFrame: Normalized input data ready for model consumption.
    """
    # NOTE: Make sure the input is correct and feature names is correct
    # display(raw_input_df)

    # rearrange dataframe columns to fit normalizer
    rearranged_features = list(scaler.feature_names_in_)
    # display(rearranged_features)

    raw_input_df = raw_input_df[rearranged_features]

    # normalize data
    input_df = scaler.transform(raw_input_df)

    return input_df

def data_pipeline_v2(
        raw_input_df: pd.DataFrame, 
        features: Dict[str, Any], 
        normalizer: StandardScaler
    ) -> pd.DataFrame:
    """
    Preprocess raw input data by applying feature engineering and scaling 
    to prepare it for model input.

    Args:
        raw_input_df (pd.DataFrame): The raw input data in DataFrame format.
        features (dict): Dictionary containing feature engineering instructions.
        normalizer (StandardScaler): Pre-trained scaler to scale the data.

    Returns:
        pd.DataFrame: Scaled input data ready for model consumption.
    """
    # NOTE: Make sure the input is correct and feature names is correct
    # do feature engineering
    for feature in list(features.keys()):
        raw_input_df[feature] = raw_input_df[features[feature]].prod(axis=1)

    # rearrange dataframe columns to fit scaler
    rearranged_features = list(normalizer.feature_names_in_)
    raw_input_df = raw_input_df[rearranged_features]

    # scale data
    input_df = normalizer.transform(raw_input_df)
    
    return input_df

def prediction_pipeline(
        raw_input_from_user: dict, 
        model_name: str,
        data_from_dataset: pd.DataFrame = None, 
    ) -> Tuple[np.ndarray, float, str]:
    """
    Run the prediction pipeline, processing raw user input and making a prediction using the specified model.

    Args:
        raw_input_from_user (dict): Dictionary of user inputs formatted for model compatibility.
        model_name (str): The name of the model to use for prediction.

    Returns:
        Tuple[np.ndarray, float, str]: 
        - prediction: The raw prediction output from the model.
        - prediction_proba: The prediction probability from the model.
        - prediction_str: The human-readable string output decoded from the model prediction.
    """
    # 1. load objects
    (scaler, normalizer, encoder, features, models) = load_objects()

    model = models[model_name]
    n_features = model.n_features_in_

    # 2. read input from user and convert to dataframe
    if data_from_dataset is not None:
        raw_input_df = data_from_dataset
    else:
        raw_input_df = pd.DataFrame([raw_input_from_user])

    # 3. process data / data pipeline
    if n_features == 4:
        input_data_to_model = data_pipeline_v1(raw_input_df, scaler)
    else:
        input_data_to_model = data_pipeline_v2(raw_input_df, features["engineered"], normalizer)

    # 4. prediction
    prediction = model.predict(input_data_to_model)
    prediction_proba = model.predict_proba(input_data_to_model).max() * 100

    # 5. interpret output
    prediction_str = encoder.inverse_transform(prediction)[0]
    
    # 6. save prediction
    save_prediction(raw_input_df, prediction, model_name)

    return prediction, prediction_proba, prediction_str

def save_prediction(input:pd.DataFrame, prediction:pd.DataFrame, model_name:str) -> None:
    """
    Save every prediction to a file
    """

    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    if PREDICTION_FILE not in os.listdir(DATA_PATH):
        pd.DataFrame().to_csv(f"{DATA_PATH}{PREDICTION_FILE}")
    
    current_result = pd.read_csv(f"{DATA_PATH}{PREDICTION_FILE}")
    current_prediction = input.copy()
    current_prediction["prediction"] = prediction
    current_prediction["model"] = [model_name]
    current_prediction["timestamp"] = [datetime.now()]
    current_result = pd.concat([current_result, current_prediction], axis=0)

    current_result.to_csv(f"{DATA_PATH}{PREDICTION_FILE}", index=False)