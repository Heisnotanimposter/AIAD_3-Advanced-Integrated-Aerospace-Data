import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D, Flatten, Dropout, BatchNormalization
from tqdm import tqdm

# Set parameters
iterations = 30
vector_noise_shape = 180
count_example = 16
batch_size = 12
count_buffer = 60000
prediction_horizon_hours = 24  # Number of hours to predict into the future
prediction_horizon_days = 30   # For long-term (daily) predictions
seed = tf.random.normal([count_example, vector_noise_shape])

# Define video path
Carbon_Video_Set = "/content/drive/MyDrive/PBL_Shared_Data/202101cldpng_unpacked/unpacked_resize.mp4"

# Function to extract white regions (representing clouds)
def extract_white_regions(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    resized_result = cv2.resize(result, (180, 180))
    return (resized_result - 127.5) / 127.5

# Extract frames and preprocess
Video_IMG_List = []
Capture_Video = cv2.VideoCapture(Carbon_Video_Set)

if not Capture_Video.isOpened():
    print(f"Error: Could not open video file {Carbon_Video_Set}. Please check the path.")
else:
    while Capture_Video.isOpened():
        ret, frame = Capture_Video.read()
        if not ret:
            print("End of video file or no frames captured.")
            break
        processed_frame = extract_white_regions(frame)
        Video_IMG_List.append(processed_frame)

Capture_Video.release()

if len(Video_IMG_List) == 0:
    print("No frames were processed. Exiting...")
else:
    Main_Array = np.array(Video_IMG_List)
    print(f"Processed video frames shape: {Main_Array.shape}")

    # Prepare data for training
    Train_Data = tf.data.Dataset.from_tensor_slices(Main_Array).shuffle(count_buffer).batch(batch_size)

    # Define GAN Generator model
    def Generator_Model():
        Model = Sequential()
        Model.add(Dense(90 * 90 * 128, use_bias=False, input_shape=(vector_noise_shape,)))
        Model.add(BatchNormalization())
        Model.add(LeakyReLU())
        Model.add(Reshape((90, 90, 128)))
        Model.add(Conv2DTranspose(128, (3, 3), padding="same", use_bias=False))
        Model.add(BatchNormalization())
        Model.add(LeakyReLU())
        Model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        Model.add(BatchNormalization())
        Model.add(LeakyReLU())
        Model.add(Conv2DTranspose(3, (3, 3), padding="same", use_bias=False, activation="tanh"))
        return Model

    # Define GAN Discriminator model
    def Discriminator_Model():
        Model = Sequential()
        Model.add(Conv2D(64, (3, 3), padding="same", input_shape=[180, 180, 3]))
        Model.add(Dropout(0.3))
        Model.add(LeakyReLU())
        Model.add(Conv2D(128, (3, 3), padding="same"))
        Model.add(Dropout(0.3))
        Model.add(LeakyReLU())
        Model.add(Flatten())
        Model.add(Dense(1))
        return Model

    # Instantiate models
    Generator = Generator_Model()
    Discriminator = Discriminator_Model()

    # Loss and optimizers
    Loss_Function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    Generator_Optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, clipvalue=1.0, decay=1e-8)
    Discriminator_Optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, clipvalue=1.0, decay=1e-8)

    # Define loss functions
    def Discriminator_Loss(real_output, fake_output):
        real_loss = Loss_Function(tf.ones_like(real_output), real_output)
        fake_loss = Loss_Function(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def Generator_Loss(fake_output):
        return Loss_Function(tf.ones_like(fake_output), fake_output)

    # Function to generate and save images with time and loss annotations
    def generate_and_save_function(Model, epoch, test_input, gen_loss, disc_loss, prediction_type='hour'):
        predictions = Model(test_input, training=False)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i] * 127.5 + 127.5).astype(np.uint8))
            plt.axis('off')
            
            # Annotate image with time and loss information
            if prediction_type == 'hour':
                plt.text(0, -5, f'Hour: {i + epoch} hr', fontsize=8, color='white')
            else:
                plt.text(0, -5, f'Day: {i + epoch} days', fontsize=8, color='white')
            plt.text(0, -10, f'Gen Loss: {gen_loss:.4f}', fontsize=8, color='white')
            plt.text(0, -15, f'Disc Loss: {disc_loss:.4f}', fontsize=8, color='white')
        
        plt.savefig(f'pred_{prediction_type}_epoch_{epoch:04d}.png')
        plt.show()

    # Training function
    def train(dataset, epochs, prediction_type='hour'):
        gen_loss_list = []
        disc_loss_list = []

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            for image_batch in dataset:
                noise = tf.random.normal([batch_size, vector_noise_shape])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = Generator(noise, training=True)
                    real_output = Discriminator(image_batch, training=True)
                    fake_output = Discriminator(generated_images, training=True)
                    gen_loss = Generator_Loss(fake_output)
                    disc_loss = Discriminator_Loss(real_output, fake_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, Generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, Discriminator.trainable_variables)
                Generator_Optimizer.apply_gradients(zip(gradients_of_generator, Generator.trainable_variables))
                Discriminator_Optimizer.apply_gradients(zip(gradients_of_discriminator, Discriminator.trainable_variables))

                gen_loss_list.append(gen_loss.numpy())
                disc_loss_list.append(disc_loss.numpy())

            if gen_loss_list and disc_loss_list:
                generate_and_save_function(Generator, epoch + 1, seed, gen_loss.numpy(), disc_loss.numpy(), prediction_type)

        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(gen_loss_list, label='Generator Loss')
        plt.title('Generator Loss Over Epochs')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(disc_loss_list, label='Discriminator Loss')
        plt.title('Discriminator Loss Over Epochs')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Execute the training process for hourly prediction
    train(Train_Data, iterations, prediction_type='hour')

    # Execute the training process for daily prediction
    train(Train_Data, iterations, prediction_type='day')

    # Save the models
    Generator.save("Generator.h5")
    Discriminator.save("Discriminator.h5")
