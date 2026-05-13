import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from constants import test_path, index_slice_size, interval_size_seconds, num_models, model_path
from file_processing import process_file


def encode_labels(label):
    if(label == "no_quake"):
        return 0
    elif(label == "impact_mq"):
        return 1
    elif(label == "deep_mq"):
        return 2
    return 3

def process_predictions(predictions):
    for i in range(len(predictions)):
        predictions[i] = encode_labels(predictions[i])

def read_prediction_models():
    prediction_models = []
    for i in range(num_models):
        filename=f"{model_path}RandomForest_{i}.sav"
        loaded_model = pickle.load(open(filename, "rb"))
        prediction_models.append(loaded_model)
    return prediction_models

def read_outlier_models():
    outlier_models = []
    for i in range(num_models):
        filename=f"{model_path}IsolationForest_{i}.sav"
        loaded_model = pickle.load(open(filename, "rb"))
        outlier_models.append(loaded_model)
    return outlier_models

def average_predictions(prediction_list):
    return np.average(prediction_list, axis=0)

def get_predictions(df: pd.DataFrame, models: list):
    predictions = []
    for model in models:
        y_predict = model.predict(df)
        predictions.append(y_predict)
    return predictions

if __name__ == "__main__":
    filename = "xa.s12.00.mhz.1970-02-18HR00_evid00016.csv"
    test_df = pd.read_csv(f"{test_path}{filename}")

    prediction_models = read_prediction_models()
    outlier_models = read_outlier_models()
    interval_df = process_file(filename, test_df, None, None)

    outlier_predictions = get_predictions(interval_df, outlier_models)
    averaged_outliers = average_predictions(outlier_predictions)
    for i in range(len(averaged_outliers)):
        if(averaged_outliers[i] > 0):
            averaged_outliers[i] = 1
        else:
            averaged_outliers[i] = -1

    interval_df["Outlier"] = averaged_outliers
    
    predictions = get_predictions(interval_df, prediction_models)

    x = 0
    for prediction in predictions:
        process_predictions(prediction)
    predictions = np.average(predictions, axis=0)
    
    x = 0
    for val in predictions:
        if(val == 1):
            print(f"Activity Detected at: \n{interval_df.iloc[x]}")
        x += 1
    

    
