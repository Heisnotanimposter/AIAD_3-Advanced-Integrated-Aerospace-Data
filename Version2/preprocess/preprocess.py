import os
import cv2
import numpy as np
import netCDF4 as nc
from PIL import Image

# Paths
input_dir = 'path/to/netcdf/files'
output_dir = 'path/to/output/images'
numpy_dir = 'path/to/output/numpy'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(numpy_dir, exist_ok=True)

# NetCDF to PNG Conversion
def netcdf_to_png(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.nc'):
            ds = nc.Dataset(os.path.join(input_dir, filename))
            var_name = list(ds.variables.keys())[0]  # assuming the first variable is the target
            data = ds.variables[var_name][:]
            # Normalize and convert to image
            norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
            img = Image.fromarray(np.uint8(norm_data))
            img.save(os.path.join(output_dir, filename.replace('.nc', '.png')))

# Resize PNG Images
def resize_images(input_dir, output_dir, size=(180, 180)):
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(input_dir, filename))
            img = img.resize(size, Image.ANTIALIAS)
            img.save(os.path.join(output_dir, filename))

# Convert PNG to NumPy arrays
def png_to_numpy(input_dir, numpy_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
            np.save(os.path.join(numpy_dir, filename.replace('.png', '.npy')), img)

# Process Flow
netcdf_to_png(input_dir, output_dir)
resize_images(output_dir, output_dir)
png_to_numpy(output_dir, numpy_dir)

print("Preprocessing completed.")
