import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from constants import index_slice_size, estimators
import pickle

model_num = 0

def graph_seismic_data(filename: str, df: pd.DataFrame,intervals, expected_range):
    global time_slice_size
    #Graph the seismic data drawing a line for Standard Deviation 
    plt.plot(df["time_rel(sec)"], df["velocity(m/s)"])
    plt.plot(df["time_rel(sec)"], [3*expected_range]*len(df["time_rel(sec)"]), color="black")
    plt.plot(df["time_rel(sec)"], [-3*expected_range]*len(df["time_rel(sec)"]), color="black")

    for val in intervals:
        plt.axvline(x=val, color="black")

    plt.title(filename)
    plt.xlabel("time_rel(sec)")
    plt.ylabel("velocity(m/s)")
    plt.show()

def normalize_data(df: pd.DataFrame):
    df["velocity(m/s)"] = (df["velocity(m/s)"] - np.average(df["velocity(m/s)"]))/(df["velocity(m/s)"].max() - df["velocity(m/s)"].min())
    return df

def sum_of_squares(values: pd.Series):
    x = 0
    for value in values:
        x += np.square(value)
    return x

def get_params(df: pd.DataFrame, start_index: int, end_index: int, expected_activity, activity_type: str, std_dev):
    time_diff = df.iloc[end_index]["time_rel(sec)"] - df.iloc[start_index]["time_rel(sec)"]

    start = df.iloc[start_index]["time_rel(sec)"]
    end = df.iloc[end_index]["time_rel(sec)"]
    
    average = np.average(df.iloc[start_index:end_index]["velocity(m/s)"])
    max = np.max(df.iloc[start_index:end_index]["velocity(m/s)"])
    min = np.min(df.iloc[start_index:end_index]["velocity(m/s)"])

    outlier = 0

    range = max-min

    range_rate = range/time_diff

    energy = sum_of_squares(df.iloc[start_index:end_index]["velocity(m/s)"])
    average_energy = energy/len(df.iloc[start_index:end_index])    

    if(expected_activity != None):
        if(start <= expected_activity <= end):
            activity = activity_type
        else:
            activity= "no_quake"

        return [start, end, average, max, min, range, range_rate, energy, average_energy, outlier, activity]
    return [start, end, average, max, min, range, range_rate, energy, average_energy, outlier]

def assign_intervals(df: pd.DataFrame, expected_activity, type):
    if(expected_activity != None):
        headers = ["StartInterval", "EndInterval", "AverageVelocity", "MaxVelocity", "MinVelocity", 
                "Range", "RangeRate", "Energy", "AverageEnergy","Outlier", "Activity"]
    else:
        headers = ["StartInterval", "EndInterval", "AverageVelocity", "MaxVelocity", "MinVelocity", 
                "Range", "RangeRate", "Energy", "AverageEnergy","Outlier"]
    data = []
    std_dev = np.std(df["velocity(m/s)"])
    for index in range(0,len(df), index_slice_size):
        end_index = index + index_slice_size
        if(end_index > len(df)):
            data.append(get_params(df, index, -1, expected_activity, type, std_dev))
        else:
            data.append(get_params(df, index, end_index, expected_activity, type, std_dev))
    interval_df = pd.DataFrame(data=data, columns=headers)
    return interval_df

def save_model(model_name, model):
    global models, model_num
    print(f"Saving: {model_name}....")
    model_path = "models/"
    pickle.dump(model, open(f"{model_path}{model_name}", "wb"))
    model_num += 1


def assign_outliers(df: pd.DataFrame):
    global model_num
    try:
        X, y = df.drop(["Activity"], axis=1), df["Activity"]
        parameters = df["Activity"].value_counts()
        #Divide the number of quake events by the number of no quake events
        contamination = 1/float(parameters["no_quake"])
        model = IsolationForest(n_estimators=estimators, contamination=contamination)
        model = model.fit(X, y)
        save_model(f"IsolationForest_{model_num}.sav", model)

        outliers = model.predict(X)
        return outliers
    except(KeyError):
        return [None]*len(df)


def process_file(filename: str, df: pd.DataFrame, activity, activity_type):
    #Normalize the velocities (Modify so they fall within a normal distribution (-1 to 1))
    normalized_df = normalize_data(df)

    interval_df = assign_intervals(df=normalized_df, expected_activity=activity, type=activity_type)
    outliers = assign_outliers(df=interval_df)

    interval_df["Outlier"] = outliers

    # Visualize the data
    intervals = interval_df["StartInterval"]
    std_dev = np.std(normalized_df["velocity(m/s)"])
    graph_seismic_data(filename, normalized_df, intervals.values, std_dev)

    return interval_df