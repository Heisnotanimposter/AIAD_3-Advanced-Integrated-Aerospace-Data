# Climate Change Prediction with DCGAN

This project explores the use of Deep Convolutional Generative Adversarial Networks (DCGANs) to generate synthetic images potentially representative of climate change-related phenomena. The code is based on a Colab notebook and leverages various Python libraries for image processing, video handling, and deep learning.

## Overview

The core idea is to train a DCGAN on a dataset of images extracted from a video related to climate change (e.g., images of clouds, carbon emissions, or melting ice). The trained generator can then be used to create new, synthetic images that, ideally, capture some of the visual characteristics of the training data.

**Note:** This project is experimental and the generated images may not be scientifically accurate predictions of future climate states. It serves as a demonstration of using GANs for image generation in a climate change context.

## Dependencies

The following Python libraries are required to run the code:

*   **General:** pandas, numpy, seaborn, matplotlib
*   **Path Processing:** os, pathlib, glob
*   **Image Processing:** PIL (Pillow), keras.preprocessing, cv2 (OpenCV), skimage, imageio
*   **Scaler & Transformation:** scikit-learn (sklearn), keras.utils
*   **Accuracy Control:** scikit-learn metrics
*   **Optimizers:** keras.optimizers
*   **Model Layers:** tensorflow, keras
*   **Scikit-learn Classifiers:** xgboost, lightgbm, catboost, sklearn linear models and ensemble methods
*   google.colab
*   nibabel

You can install the necessary packages using pip:

```bash  
pip install pandas numpy seaborn matplotlib scikit-learn Pillow opencv-python scikit-image imageio tensorflow keras xgboost lightgbm catboost nibabel  
pip install catboost # you already have this in your notebook  