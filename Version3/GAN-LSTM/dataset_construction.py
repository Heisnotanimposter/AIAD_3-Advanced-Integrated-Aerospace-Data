import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load features and prepare the dataset for LSTM
features_df = pd.read_csv("/content/drive/MyDrive/PBL_Shared_Data/Generated_Images_and_Features/features.csv")
weather_data = pd.read_csv("/content/drive/MyDrive/PBL_Shared_Data/weather_data.csv")

# Combine datasets by timestamp
combined_data = pd.merge(features_df, weather_data, on="timestamp", how="inner")

# Ensure data is sorted by timestamp if your model depends on chronological order
combined_data.sort_values('timestamp', inplace=True)

# Define a function to create input-output sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data.iloc[i:i+n_steps].drop(['target', 'timestamp', 'image_path'], axis=1).values)
        y.append(data.iloc[i + n_steps]['target'])
    return np.array(X), np.array(y)

# Number of timesteps per sequence
n_steps = 10
X, y = create_sequences(combined_data, n_steps)

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save processed datasets
np.save('/content/drive/MyDrive/PBL_Shared_Data/X_train.npy', X_train)
np.save('/content/drive/MyDrive/PBL_Shared_Data/y_train.npy', y_train)
np.save('/content/drive/MyDrive/PBL_Shared_Data/X_val.npy', X_val)
np.save('/content/drive/MyDrive/PBL_Shared_Data/y_val.npy', y_val)

# Simple visualization of the datasets
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(y_train, label='Train Targets')
plt.title('Training Set Targets')
plt.legend()
plt.subplot(122)
plt.plot(y_val, label='Validation Targets')
plt.title('Validation Set Targets')
plt.legend()
plt.show()
