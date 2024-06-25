
import cv2
import numpy as np
import pandas as pd
import os
from skimage import feature, measure, color
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Directory for saving outputs
output_dir = "/content/drive/MyDrive/PBL_Shared_Data/Generated_Images_and_Features/"
os.makedirs(output_dir, exist_ok=True)

# Function Definitions

# Calculate Cloud Coverage via Thresholding
def calculate_cloud_coverage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    cloud_coverage = np.sum(thresholded == 255) / np.prod(thresholded.shape)
    return cloud_coverage

# Edge Detection to identify cloud boundaries
def detect_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges > 0) / np.prod(edges.shape)

# Texture Analysis for classifying cloud types
def texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = feature.local_binary_pattern(gray_image, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

# Segmentation to isolate cloud regions
def segment_clouds(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(gray_image.reshape(-1, 1))
    segmented = kmeans.labels_.reshape(gray_image.shape)
    return segmented

# Load cloud type classification model (assuming it's pre-trained)
cloud_classifier = load_model('/content/drive/MyDrive/cloud_classifier_model.h5')

# Function to categorize cloud types
def classify_cloud_type(image):
    image_resized = cv2.resize(image, (64, 64))  # Assuming classifier input size is 64x64
    predictions = cloud_classifier.predict(np.expand_dims(image_resized, axis=0))
    return np.argmax(predictions)

# Main Function to Process GAN Outputs
def process_gan_output(image_path):
    image = cv2.imread(image_path)
    features = {}
    features['cloud_coverage'] = calculate_cloud_coverage(image)
    features['edge_density'] = detect_edges(image)
    features['texture_histogram'] = texture_features(image)
    features['segmented_image'] = segment_clouds(image)
    features['cloud_type'] = classify_cloud_type(image)
    return features

# Example Usage
image_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
all_features = [process_gan_output(path) for path in image_paths]

# Convert feature data into a suitable format for further processing or analysis
features_df = pd.DataFrame(all_features)
features_df.to_csv(os.path.join(output_dir, 'extracted_features.csv'), index=False)
