"""
Improved DCGAN Model for Satellite Weather Prediction v4.0
Enhanced with better architecture, training stability, and performance optimizations
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D, 
    Flatten, Dropout, BatchNormalization, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import cv2


class ImprovedDCGAN:
    """Improved DCGAN with enhanced architecture and training stability"""
    
    def __init__(self, img_shape=(180, 180, 3), latent_dim=200):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.channels = img_shape[2]
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=BinaryCrossentropy(from_logits=True),
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise as input and generates imgs
        z = Input(shape=(latent_dim,))
        img = self.generator(z)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        
        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)
        
        # The combined model (stacked generator and discriminator)
        self.combined = Model(z, validity)
        self.combined.compile(
            loss=BinaryCrossentropy(from_logits=True),
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
        )
        
        # Training history
        self.train_history = {'g_loss': [], 'd_loss': [], 'd_acc': []}
    
    def build_generator(self):
        """Enhanced generator with better architecture"""
        model = Sequential()
        
        # Foundation for 7x7 image
        model.add(Dense(7 * 7 * 256, use_bias=False, input_dim=self.latent_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 256)))
        
        # Upsampling to 14x14
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        # Upsampling to 28x28
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        # Upsampling to 56x56
        model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        # Upsampling to 112x112
        model.add(Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        # Upsampling to 180x180
        model.add(Conv2DTranspose(self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        
        model.summary()
        return model
    
    def build_discriminator(self):
        """Enhanced discriminator with better feature extraction"""
        model = Sequential()
        
        # Input shape: 180x180x3
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        model.add(Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        model.add(Flatten())
        model.add(Dense(1))
        
        model.summary()
        return model
    
    def train(self, dataset, epochs, batch_size=32, save_interval=50):
        """Train the DCGAN model with improved training loop"""
        
        # Ensure output directory exists
        os.makedirs('generated_images', exist_ok=True)
        os.makedirs('saved_models', exist_ok=True)
        
        # Callbacks for better training
        callbacks = [
            ReduceLROnPlateau(monitor='g_loss', factor=0.5, patience=10, min_lr=0.00001),
            EarlyStopping(monitor='g_loss', patience=50, restore_best_weights=True)
        ]
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            g_loss_epoch = []
            d_loss_epoch = []
            d_acc_epoch = []
            
            for batch in dataset:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                # Select a random batch of images
                imgs = batch
                
                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise, verbose=0)
                
                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Train Generator
                # ---------------------
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, valid)
                
                # Record losses
                g_loss_epoch.append(g_loss)
                d_loss_epoch.append(d_loss[0])
                d_acc_epoch.append(d_loss[1])
            
            # Calculate epoch averages
            avg_g_loss = np.mean(g_loss_epoch)
            avg_d_loss = np.mean(d_loss_epoch)
            avg_d_acc = np.mean(d_acc_epoch)
            
            # Store history
            self.train_history['g_loss'].append(avg_g_loss)
            self.train_history['d_loss'].append(avg_d_loss)
            self.train_history['d_acc'].append(avg_d_acc)
            
            # Print progress
            print(f"{epoch} [D loss: {avg_d_loss:.4f}, acc.: {100*avg_d_acc:.2f}%] [G loss: {avg_g_loss:.4f}]")
            
            # If at save interval => save generated image samples and models
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.save_models(epoch)
        
        # Final save
        self.save_models(epochs)
        self.plot_training_history()
    
    def save_imgs(self, epoch, examples=16):
        """Save generated images at specified intervals"""
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c, figsize=(12, 12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,])
                axs[i,j].axis('off')
                cnt += 1
        
        fig.savefig(f"generated_images/weather_pred_epoch_{epoch:04d}.png")
        plt.close()
    
    def save_models(self, epoch):
        """Save the generator and discriminator models"""
        self.generator.save(f"saved_models/generator_epoch_{epoch:04d}.h5")
        self.discriminator.save(f"saved_models/discriminator_epoch_{epoch:04d}.h5")
        self.combined.save(f"saved_models/combined_epoch_{epoch:04d}.h5")
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Plot Generator & Discriminator Loss
        plt.subplot(1, 3, 1)
        plt.plot(self.train_history['g_loss'], label='Generator Loss')
        plt.plot(self.train_history['d_loss'], label='Discriminator Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Discriminator Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.train_history['d_acc'], label='Discriminator Accuracy')
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot Loss Difference
        plt.subplot(1, 3, 3)
        loss_diff = np.abs(np.array(self.train_history['g_loss']) - np.array(self.train_history['d_loss']))
        plt.plot(loss_diff, label='Loss Difference')
        plt.title('Generator vs Discriminator Loss Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Absolute Difference')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def generate_prediction(self, num_samples=1):
        """Generate weather predictions"""
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        generated_images = self.generator.predict(noise, verbose=0)
        
        # Rescale from [-1, 1] to [0, 255]
        generated_images = (generated_images + 1) * 127.5
        generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)
        
        return generated_images
    
    def load_models(self, generator_path, discriminator_path):
        """Load pre-trained models"""
        self.generator = tf.keras.models.load_model(generator_path)
        self.discriminator = tf.keras.models.load_model(discriminator_path)
        
        # Rebuild combined model
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(
            loss=BinaryCrossentropy(from_logits=True),
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
        )


def preprocess_satellite_data(video_path, output_dir="processed_data"):
    """Enhanced satellite data preprocessing"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Enhanced preprocessing
        # Convert to HSV for better cloud detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Cloud detection using HSV thresholds
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 55, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply mask and resize
        result = cv2.bitwise_and(frame, frame, mask=mask)
        resized = cv2.resize(result, (180, 180))
        
        # Normalize to [-1, 1] for GAN
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        
        frames.append(normalized)
        frame_count += 1
        
        # Save sample frames for debugging
        if frame_count % 100 == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", resized)
    
    cap.release()
    print(f"Processed {frame_count} frames from video")
    return np.array(frames)


if __name__ == "__main__":
    # Example usage
    gan = ImprovedDCGAN(img_shape=(180, 180, 3), latent_dim=200)
    
    # For demonstration, create random data
    # In practice, use: data = preprocess_satellite_data("path/to/video.mp4")
    data = np.random.rand(1000, 180, 180, 3) * 2 - 1
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(1000).batch(32)
    
    # Train the model
    gan.train(dataset, epochs=100, batch_size=32, save_interval=10)
