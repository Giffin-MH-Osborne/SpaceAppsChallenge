import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from constants import training_path, estimators
from file_processing import process_file

catalog_df = pd.read_csv("catalogs/apollo12_catalog_GradeA_final.csv")
models = []
model = RandomForestClassifier(n_estimators=estimators)


def fit_model(df: pd.DataFrame):
    global model, models
    X, y = df.drop(["Activity"], axis=1), df["Activity"]
    fitted_model = model.fit(X, y)
    print(fitted_model)
    models.append(fitted_model)


#Loop over the catalog and read the data from each training file
def read_training_data():
    global training_path, models
    for index, entry in catalog_df.iterrows():
        filename = entry["filename"]
        activity = entry["time_rel(sec)"]
        activity_type = entry["mq_type"]

        print(f"Processing File: {filename}....")
        #Read the dataframe in from the filepath specified
        seismic_df = pd.read_csv(f"{training_path}{filename}.csv")
        
        interval_df = process_file(filename, seismic_df, activity, activity_type)
        print(interval_df)
        input()
        fit_model(interval_df)

def save_model(model_name, index):
    global models
    model_path = "models/"
    pickle.dump(models[index], open(f"{model_path}{model_name}", "wb"))
    

if __name__ == "__main__":
    read_training_data()
    for i in range(len(models)):
        model_name = f"RandomForest_{i}.sav"
        print(f"Saving: {model_name}....")
        save_model(model_name, i)