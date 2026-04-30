import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle


#GLOBAL CONSANTS

catalog_df = pd.read_csv("catalogs/apollo12_catalog_GradeA_final.csv")
#dataframe containing the catalog of data entries, filename as well as timestamp for activity

data_path = "train_data/"
#path to access the actual data files

time_slice_size = 3600
#every 9 entries is a second. 3600 seconds in an hour

DATA = 0
LABEL = 1
#Constants for the use of indices when creating result dataframe

path = "train_data/"

def graph_seismic_data(df: pd.DataFrame, std_dev):
    plt.plot(df["time_rel(sec)"], df["velocity(m/s)"])
    plt.plot(df["time_rel(sec)"], [3*std_dev]*len(df["time_rel(sec)"]))
    plt.plot(df["time_rel(sec)"], [-3*std_dev]*len(df["time_rel(sec)"]))
    plt.xlabel("time_rel(sec)")
    plt.ylabel("velocity(m/s)")
    plt.show()

def normalize_data(df: pd.DataFrame):
    df["velocity(m/s)"] = (df["velocity(m/s)"] - np.average(df["velocity(m/s)"]))/(df["velocity(m/s)"].max() - df["velocity(m/s)"].min())
    return df

def set_expectation(df: pd.DataFrame):
    global time_slice_size
    std_dev = np.std(df["velocity(m/s)"])
    expected_range = 3*std_dev
    print(f"Standard Deviation: {std_dev}")

    graph_seismic_data(df, std_dev)
    for time in range(0, len(df),time_slice_size):
        end_time = time + time_slice_size
        section = df.iloc[time:end_time]
        peak_velocity = np.max(section["velocity(m/s)"])
        outlier = not(np.abs(peak_velocity) < np.abs(expected_range))
        print(f"Section: {time} - {end_time}   Peak Velocity: {peak_velocity}   Outlier: {outlier}")

def read_training_data():
    forced_stop = 10
    i = 0
    for filename in catalog_df["filename"]:
        seismic_df = pd.read_csv(f"{path}{filename}.csv")
        normalized_df = normalize_data(seismic_df)
        fuck(normalized_df)
        i += 1
        if i== forced_stop:
            break

if __name__ == "__main__":
    read_training_data()