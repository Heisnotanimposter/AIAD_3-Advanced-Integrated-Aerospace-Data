 import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Ensure directory for outputs exists
output_dir = "/content/drive/MyDrive/PBL_Shared_Data/Generated_Images_and_Features/"
os.makedirs(output_dir, exist_ok=True)

# Feature Extraction Functions
def calculate_cloud_coverage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    return np.sum(thresholded == 255) / np.prod(thresholded.shape)

def detect_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges > 0) / np.prod(edges.shape)

# Load pre-trained model for cloud type classification
cloud_classifier = load_model('/content/drive/MyDrive/cloud_classifier_model.h5')

# Generator Model Definition
def Generator_Model():
    model = Sequential([
        Dense(90*90*128, use_bias=False, input_shape=(180,)),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((90, 90, 128)),
        Conv2DTranspose(128, (3, 3), padding="same", use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(3, (3, 3), padding="same", use_bias=False, activation="tanh")
    ])
    return model

# Instantiate and compile models
generator = Generator_Model()
generator.compile(optimizer=RMSprop(lr=0.0001, decay=1e-8), loss='binary_crossentropy')

# Generate and process images
def generate_and_process_images(model, num_images):
    noise = tf.random.normal([num_images, 180])
    images = model(noise, training=False)
    features_list = []

    for img in images:
        img = (img * 127.5 + 127.5).numpy().astype(np.uint8)
        features = {
            'cloud_coverage': calculate_cloud_coverage(img),
            'edge_density': detect_edges(img)
        }
        features_list.append(features)
    
    return features_list

# Generate images and extract features
features = generate_and_process_images(generator, 10)
features_df = pd.DataFrame(features)
features_df.to_csv(os.path.join(output_dir, 'extracted_features.csv'), index=False)

# Example of visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(images[0, :, :, :], cmap='gray')
ax.set_title('Generated Image with Features')
plt.show()
