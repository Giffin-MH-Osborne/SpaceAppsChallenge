import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from obspy import read
from tqdm import tqdm  # Import tqdm for progress tracking

# Define the directory path
data_dir = '/Users/harman/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'

# Define chunk size (e.g., 5000 points per chunk)
chunk_size = 5000

# Initialize lists to accumulate results across all files
X_all = []
y_all = []

# Get list of all files in the data directory
files = [f for f in os.listdir(data_dir) if f.endswith('.csv') or f.endswith('.mseed')]

# Loop through all files in the data directory with a progress bar
for filename in tqdm(files, desc="Processing files"):
    # Check if the file is a CSV or miniseed file
    if filename.endswith('.csv'):
        # Load the seismic data from CSV
        data_file = os.path.join(data_dir, filename)
        data = pd.read_csv(data_file)

        # Extract velocity data as a NumPy array
        velocity = data['velocity(m/s)'].values

    elif filename.endswith('.mseed'):
        # Load the seismic data from miniseed
        mseed_file = os.path.join(data_dir, filename)
        st = read(mseed_file)
        tr = st[0]  # Get the first trace

        # Extract velocity data as a NumPy array
        velocity = tr.data

    # Determine the number of chunks
    num_chunks = len(velocity) // chunk_size

    # Split velocity data into chunks
    chunks = np.array_split(velocity, num_chunks)

    # Create labels for the chunks (adjust as needed)
    labels = [1 if i % 10 == 0 else 0 for i in range(len(chunks))]  # Example: mark every 10th chunk as seismic event

    # Append the chunks and labels to the overall lists
    X_all.extend(chunks)
    y_all.extend(labels)

# Convert the accumulated data into a pandas DataFrame
df = pd.DataFrame(X_all)

# Add the labels as a new column
df['label'] = y_all

# Split the data into features (X) and labels (y)
X = df.drop(columns=['label'])
y = df['label']

# Perform train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Initialize the Random Forest Classifier
clf = RandomForestClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")