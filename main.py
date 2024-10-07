import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

#GLOBAL CONSANTS
time_slice_size = 3600

cat_df = pd.read_csv("catalogs/apollo12_catalog_GradeA_final.csv")
#dataframe containing the catalog of data entries, filename as well as timestamp for activity

data_path = "test_data/"
#path to access the actual data files

time_intervals = pd.DataFrame(columns=['Standard_Dev'])

def clean_df(df: pd.DataFrame):
     cleaned_df = df.drop(['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], axis=1)
     return cleaned_df

def normalize_data(cleaned_df: pd.DataFrame):
    scalar = StandardScaler()
    normalized_df = scalar.fit_transform(cleaned_df.values)
    return normalized_df
                
def process_data(i: int, filename: str):
    global activity_detected
    #try to read the file, if it doesnt exist carry on
    try:
        df = pd.read_csv(data_path + filename + ".csv")
        activity_detected = cat_df["time_rel(sec)"][i]

        cleaned_df = clean_df(df)
        normalized_df = normalize_data(cleaned_df)

        Iso_Forest = IsolationForest(contamination=0.01,random_state=123) 
        peaks = Iso_Forest.fit_predict(pd.DataFrame(normalized_df))
        return peaks

    except(FileNotFoundError):
        return

def populate_std(peaks: np.ndarray, df: pd.DataFrame):
    row = 0

    peak_index = 0
    while peak_index < len(peaks):
        if(peaks[peak_index] == -1):
            start_time = df['time_rel(sec)'][peak_index]
            upper_bound = start_time+time_slice_size
            lower_bound = start_time - time_slice_size
            sliced_df = df[df['time_rel(sec)'].between(lower_bound, upper_bound)]
            interval_std = sliced_df['velocity(m/s)'].describe().transpose()['std']
            time_intervals.loc[row] = [interval_std]
            row += 1
            peak_index += (time_slice_size*9)
        peak_index += 1

def output_interval(peaks: np.ndarray, df:pd.DataFrame, labels: np.ndarray):
    interval = 0

    peak_index = 0
    while peak_index < len(peaks):
        if(peaks[peak_index] == -1):
            print('peak')
            print(labels[interval])
            if(labels[interval] == 1):
                print('yep')
                start_time = df['time_rel(sec)'][peak_index]
                upper_bound = start_time+time_slice_size
                df[df['time_rel(sec)'].between(start_time, upper_bound)].to_csv(
                     'result_data/{}_{}.csv'.format(
                          df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'][0], df['time_rel(sec)'][0]
                          ), index=False
                          )
            peak_index += (time_slice_size*9)
            interval += 1
        peak_index += 1

if __name__ == '__main__':
    model = pickle.load(open('model/RandomForest.pkl', 'rb'))

    test_filename = 'xa.s12.00.mhz.1970-02-18HR00_evid00016'
    df = pd.read_csv(data_path + test_filename + ".csv")
    print('finding peaks')
    peaks = process_data(0, test_filename)
    print('found\n')
    print('populating DataFrame')
    populate_std(peaks, df)
    print('populated\n')

    print('predicting')
    labels = model.predict(time_intervals)
    print('predicted\n')

    print('outputting')
    output_interval(peaks, df, labels)



