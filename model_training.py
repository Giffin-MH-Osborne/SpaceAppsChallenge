import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle


#GLOBAL CONSANTS

cat_df = pd.read_csv("catalogs/apollo12_catalog_GradeA_final.csv")
#dataframe containing the catalog of data entries, filename as well as timestamp for activity

data_path = "train_data/"
#path to access the actual data files

time_slice_size = 3600
#every 9 entries is a second. 3600 seconds in an hour

DATA = 0
LABEL = 1
#Constants for the use of indices when creating result dataframe

#GLOBAL VARIABLES
activity_detected = 0
#timestamp where there is seismic activity

labels = []
#labels for LogisticRegression

train_row = 0
test_row = 0


time_intervals = []
#initialize list to contain the time intervals

train_files = []
test_files = []

train_engineered_data = pd.DataFrame(columns=['Standard_Dev', 'Label'])
test_engineered_data = pd.DataFrame(columns=['Standard_Dev', 'Label'])

def train_test():
    training_max_index = np.round(len(cat_df['filename'])*0.8)
    index = 0
    for filename in cat_df['filename']:
        if(index <= training_max_index):
            train_files.append(filename)
        else:
            test_files.append(filename)
        index +=1

def clean_df(df: pd.DataFrame):
     cleaned_df = df.drop(['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], axis=1)
     return cleaned_df

def normalize_data(cleaned_df: pd.DataFrame):
    scalar = StandardScaler()
    normalized_df = scalar.fit_transform(cleaned_df.values)
    return normalized_df

def isolate_peaks(peaks: np.ndarray, df: pd.DataFrame, testing: bool):
    global activity_detected, train_row, test_row, train_engineered_data

    peak_index = 0
    while peak_index < len(peaks):
        if(peaks[peak_index] == -1):
            start_time = df['time_rel(sec)'][peak_index]
            upper_bound = start_time+time_slice_size
            lower_bound = start_time - time_slice_size

            peak_index += (time_slice_size*9)
            if(upper_bound >= activity_detected >= lower_bound):
                print('detected')
                sliced_df = df[df['time_rel(sec)'].between(lower_bound, upper_bound)]
                interval_std = sliced_df['velocity(m/s)'].describe().transpose()['std']
                if(not testing):
                    interval_label = 1
                    train_engineered_data.loc[train_row] = [interval_std, interval_label]
                    train_row += 1
                else:
                    interval_label = 1
                    test_engineered_data.loc[test_row] = [interval_std, interval_label]
                    test_row += 1
             
                
            else:
                print('no activity')
                sliced_df = df[df['time_rel(sec)'].between(lower_bound, upper_bound)]
                interval_std = sliced_df['velocity(m/s)'].describe().transpose()['std']
                if(not testing):
                    interval_label = 0
                    train_engineered_data.loc[train_row] = [interval_std, interval_label]
                    train_row += 1
                else:
                    interval_label = 0
                    test_engineered_data.loc[test_row] = [interval_std, interval_label]
                    test_row += 1

        peak_index += 1
                
def process_data(i: int, filename: str, predicting: bool):
    global activity_detected
    #try to read the file, if it doesnt exist carry on
    try:
        df = pd.read_csv(data_path + filename + ".csv")
        activity_detected = cat_df["time_rel(sec)"][i]

        cleaned_df = clean_df(df)
        normalized_df = normalize_data(cleaned_df)

        Iso_Forest = IsolationForest(contamination=0.01,random_state=123) 
        peaks = Iso_Forest.fit_predict(pd.DataFrame(normalized_df))

        isolate_peaks(peaks, df, predicting)
    except(FileNotFoundError):
        return


def fit():
    global train_engineered_data
    rf = RandomForestClassifier()
    train_X = pd.DataFrame(train_engineered_data['Standard_Dev'])
    train_y=pd.DataFrame(train_engineered_data['Label'])

    test_X = pd.DataFrame(test_engineered_data['Standard_Dev'])
    test_y=pd.DataFrame(test_engineered_data['Label'])
    rf.fit(train_X, train_y)
    print(rf.score(test_X, test_y))

    return rf


if __name__ == "__main__":
    i = 0
    train_test()
    # loop over every file in the catalog
    for filename in train_files:
        print('processing training: ' + filename)
        process_data(i, filename, False)
        print('processed\n')
        # if i == 10:
        #         break
        # i+= 1
    
    for filename in test_files:
        print('processing testing: ' + filename)
        process_data(i, filename, True)
        # print(data)
        print('processed\n')
    print(train_engineered_data)
    print('')
    print(test_engineered_data)

    model = fit()

    pickle.dump(model, open('RandomForest.pkl', 'wb'))