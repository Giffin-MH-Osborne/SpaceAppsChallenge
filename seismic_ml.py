import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load seismic data from CSV
data_file = '/Users/harman/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-01-19HR00_evid00002.csv'
data = pd.read_csv(data_file)

# Extract velocity data as NumPy array
velocity = data['velocity(m/s)'].values

# Define chunk size (e.g., 5000 points per chunk)
chunk_size = 5000

# Determine the number of chunks
num_chunks = len(velocity) // chunk_size

# Split velocity data into chunks
chunks = np.array_split(velocity, num_chunks)

# Create labels for the chunks (1 for seismic events, 0 for noise)
# Example: marking every 10th chunk as a seismic event
labels = [1 if i % 10 == 0 else 0 for i in range(len(chunks))]

# Convert chunks into a DataFrame
df = pd.DataFrame(chunks)

# Add the labels as a new column
df['label'] = labels

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