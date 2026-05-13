#GLOBAL VARIABLES
test_path = "test_data/"
model_path = "models/"

num_models = 76

#approximately every 9 entries is a second.
entries_per_second = 9

interval_size_minutes = 30

#60 seconds per minute
interval_size_seconds = 60 * interval_size_minutes 

#multiply the number of seconds by number of entries per second to get indices
index_slice_size = interval_size_seconds * entries_per_second

#path to access the training data
training_path = "train_data/"

#number of estimators for the isolation forest
estimators = 100
