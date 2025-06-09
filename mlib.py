import numpy as np
import os
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging 

MODEL_CACHE = {}
# global configuration for logging
logging.basicConfig(level=logging.INFO)
# Module specific logger
logger = logging.getLogger(__name__)

import warnings 
# Ignore UserWarnings which are generic warning for user code that are non-fatal to program execution
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(model_path = 'model.joblib'):
    """grabs the model from the disk and returns it"""
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        clf = joblib.load(model_path)
        MODEL_CACHE[model_path] = clf
        logger.info(f"Model loaded from %s", model_path)
        return clf
    except Exception as e:
        logger.exception("Error loading model")
        raise RuntimeWarning(f"Failed to load model from {model_path}: {e}")
    clf = joblib.load(model_path)
    return clf

def data(data_path = 'htwtmlb.csv'):
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("DataFrame is empty. Please check the data file.")
    logger.info(f"Data loaded from %s with shape %s", data_path, df.shape)
    return df

def retrain(t_size=0.1, model_path="model.joblib"):

    df = data()
    y = df["Height"].values  
    y = y.reshape(-1,1)
    x = df["Weight"].values
    x = x.reshape(-1,1)
    scaler = StandardScaler()
    x_scaler = scaler.fit(x)
    x = x_scaler.transform(x)
    y_scaler = scaler.fit(y)
    y = y_scaler.transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=42)
    clf = Ridge(alpha=1.0)
    model = clf.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    logger.info(f"Model retrained with accuracy: %s",accuracy)
    
    joblib.dump(model, model_path)
    #joblib.dump(x_scaler, "x_scaler.joblib")
    #joblib.dump(y_scaler, "y_scaler.joblib")

    MODEL_CACHE[model_path] = model
    logger.info(f"Model saved to %s",model_path)
    return model_path, accuracy 

def format_input(weight):
    """"Takes int and coverts it into a numpy array"""
    if not isinstance(weight, (int, float)):
        raise TypeError("Input must be an integer or float.")
    weight = np.array([[weight]], dtype=np.float64)
    logger.info(f"Input formatted: {weight}")
    return weight

def scale_input(value):
    """Scales the input value using the StandardScaler"""

    df = data()
    features = df["Weight"].values.reshape(-1, 1)
  
    input_scaler = StandardScaler().fit(features)
    scaled_input = input_scaler.transform(value)
    return scaled_input

def scale_target(value):
    """Scales the target value using the StandardScaler"""
    
    df = data()
    target = df["Height"].values.reshape(-1, 1)
    target_scaler = StandardScaler().fit(target)
    scaled_target = target_scaler.inverse_transform(value)
    logger.info(f"Scaled target: %s",scaled_target)
    return scaled_target

def height_human(float_inches):
    """converts float inches to feet and inches"""
    feet = int(round(float_inches // 12))
    inches_left = round(float_inches - feet * 12)
    result  = f"{feet} feet {inches_left} inches"
    logger.info(f"Converted height: {result}")
    return result

def human_readable_payload(predict_value):
    """Converts the prediction value to a human-readable format"""
    height_inches = float(np.round(predict_value, 2))

    results = {
        "height_inches": height_inches,
        "height_readable": height_human(height_inches)
    }

    logger.info(f"Human-readable payload: %s",results)
    return results

def predict(weight):
    """Takes weight and predicts height"""
    model = load_model()
    if not model:
        raise RuntimeError("Model is not loaded. Please retrain the model.")
    input_array_weight = format_input(weight)
    scaled_input = scale_input(input_array_weight)
  
    scaled_prediction = model.predict(scaled_input)
    
    height_prediction = scale_target(scaled_prediction)

    payload = human_readable_payload(height_prediction)
    
    logger.info(f"Prediction made for weight %s: %s", weight,payload)
    return payload
def main():
    """Main function to run the module"""
    try:
        weight = 150  # Example weight
        prediction = predict(weight)
        print(prediction)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
# This allows the module to be run as a script