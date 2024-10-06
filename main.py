import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


#GLOBAL CONSANTS

cat_df = pd.read_csv("catalogs/apollo12_catalog_GradeA_final.csv")
#dataframe containing the catalog of data entries, filename as well as timestamp for activity

data_path = "data/"
#path to access the actual data files

time_slice_size = 10000
#10000 second slices

DATA = 0
LABEL = 1
#Constants for the use of indices when creating result dataframe

#GLOBAL VARIABLES
activity_detected = 0
#timestamp where there is seismic activity

labels = []
#labels for LogisticRegression

row = 0
# #value used to iterate through result dataframe
data = pd.DataFrame(columns=['avg_chunk_accel', 'label'])
#initialize result dataframe

random_forests = []
#initialize the list to contain the models

train_files = []
test_files = []

predictions = []
actual = []

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

def avg_velocity(spikes: np.ndarray, df: pd.DataFrame, predicting: bool):
    global activity_detected, row, data
    index = 0
    counter = 0
    if(not predicting):
        row = 0

    for spike in spikes:
        if spike == -1 and counter >= index: 
            start_time = df['time_rel(sec)'][index]
            index += time_slice_size
            sliced_frame = (df[df['time_rel(sec)'].between(start_time, start_time+time_slice_size)])
            vel = sliced_frame['velocity(m/s)']
            avg_accel = (vel.iloc[-1] - vel.iloc[0])/len(vel)

            if(start_time+time_slice_size >= activity_detected >= start_time - time_slice_size/2):
                data.loc[row] = [avg_accel,1]
            else:
                data.loc[row] = [avg_accel,0]
            
            row += 1
        if index > len(spikes):
            break
        if index == counter:
            index += 1
            
        counter += 1

def process_data(i: int, filename: str, predicting: bool):
    global activity_detected
    #try to read the file, if it doesnt exist carry on
    try:
        df = pd.read_csv(data_path + filename + ".csv")
        activity_detected = cat_df["time_rel(sec)"][i]

        cleaned_df = clean_df(df)
        normalized_df = normalize_data(cleaned_df)

        Iso_Forest = IsolationForest(contamination=0.001,random_state=123) 
        spikes = Iso_Forest.fit_predict(pd.DataFrame(normalized_df))

        avg_velocity(spikes, df, predicting)
    except(FileNotFoundError):
        return

def stack(predictions: list):
    return np.column_stack(predictions)

def predict():
    global predictions, actual, row, data
    temp_predictions = []
    bag_clf = BaggingClassifier(estimator=RandomForestClassifier(),random_state=123)
    X = pd.DataFrame(data['avg_chunk_accel'])
    y = data['label']
    fitted_clf = bag_clf.fit(X, y)
      
    for file in test_files:
        row = 0
        data = pd.DataFrame(columns=['avg_chunk_accel', 'label'])
        print('row and dataframe reset')
        index = 0 
        print('processing testing file: ' + file)
        process_data(index, file, True)
        print('length: ' + str(len(data['avg_chunk_accel'])))
        print(data)
        print('processed')
        X = pd.DataFrame(data['avg_chunk_accel'])
        y = data['label']
        prediction = fitted_clf.predict(X)
        temp_predictions.append(prediction)
        actual.append(y)
        index += 1 
        print('done')
    predictions.append(np.concatenate(temp_predictions))
    print('done predictions\n')


if __name__ == "__main__":
        i = 0
        train_test()
        # loop over every file in the catalog
        for filename in train_files:
            print('processing training: ' + filename)
            process_data(i, filename, False)
            print(data)
            print('processed\n')
            
            print('beginning prediction')
            predict()
            if i == 2:
                    break
            i+= 1
        stacked_preds = stack(predictions)

        
