import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model

# Load pre-trained Generator and Discriminator models
generator = load_model('Generator.h5')
discriminator = load_model('Discriminator.h5')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Generate random noise
    vector_noise_shape = 180
    seed = tf.random.normal([1, vector_noise_shape])

    # Generate prediction using the generator
    generated_image = generator(seed, training=False)
    generated_image = (generated_image[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    # Save the image as a file
    file_path = 'predicted_image.png'
    cv2.imwrite(file_path, cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))

    # Return the predicted image
    return send_file(file_path, mimetype='image/png')

@app.route('/model_stats', methods=['GET'])
def model_stats():
    # Placeholder stats - replace with real-time values from your training loop
    stats = {
        'accuracy': '92%',
        'loss': '0.05',
        'predicted_time': 'Next 24 hours'
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
