import numpy as np
import cv2 as cv
import pandas as pd
import os
import random
import ntpath
import matplotlib.pyplot as plt
import matplotlib.image as mp_img
from imgaug import augmenters as aug
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, Flatten

# Configuration to display the full length and width of data columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
data_dir = "track"


def load_data():
    """
    Read the csv file that contains the images paths and their corresponding steering angles, throttle, reverse, speed.
    Each row in the csv file has three images taken from the left, center and right.
    """
    columns_names = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(data_dir, "driving_log.csv"), names=columns_names)
    data['center'] = data['center'].apply(lambda path: ntpath.split(path)[1])
    data['left'] = data['left'].apply(lambda path: ntpath.split(path)[1])
    data['right'] = data['right'].apply(lambda path: ntpath.split(path)[1])
    print(data.head(10))
    return data


def steering_angles_histogram(data):
    """
    Shows the steering angles values that ranges from (-1 to 1) in a Histogram with 25 bins (intervals)
    Steering Angles values ranges from -1 to 1.
    Center the histogram at angle 0.
    """
    hist, bins = np.histogram(data['steering'], 25)
    centered_at_zero = (bins[:-1] + bins[1:]) * 0.5
    plt.figure("Steering Angles")
    plt.bar(centered_at_zero, hist, width=0.05)
    plt.plot((-1, 1), (400, 400))
    return centered_at_zero, bins


def balance_data(data):
    """
    The training data is skewed towards the middle because most of the time the car was driven in a straight line
    while training (See Steering Angles Histogram). And if we train our convolutional neural network based on
    this data then the model could become biased towards driving straight all the time. So we must flatten our data
    distribution and cut off extraneous samples for specific bins whose frequency exceed 400.
    """
    centered_at_zero, bins = steering_angles_histogram(data)

    print('Total: ', len(data))

    removed_list = []
    for i in range(25):
        temp_list = []
        for j in range(len(data['steering'])):
            if data['steering'][j] >= bins[i] and data["steering"][j] <= bins[i + 1]:
                temp_list.append(j)
        temp_list = shuffle(temp_list)
        temp_list = temp_list[400:]
        removed_list.extend(temp_list)

    print('Removed:', len(removed_list))

    data.drop(data.index[removed_list], inplace=True)

    print("Remaining: ", len(data))
    plt.figure("Balanced Steering Angles")
    hist, _ = np.histogram(data['steering'], 25)
    plt.bar(centered_at_zero, hist, width=0.05)
    plt.plot((np.min(data['steering']), np.max(data['steering'])), (400, 400))
    return data


def load_images(data):
    """
    Load the images paths and their corresponding steering angle (Labels) in two separated Numpy arrays
    """
    img_paths = []
    steering_labels = []
    for i in range(len(data)):
        row = data.iloc[i]
        center, left, right, steering_angle = row[0], row[1], row[2], row[3]

        img_paths.append(os.path.join(data_dir + '/IMG', center.strip()))
        steering_labels.append(float(steering_angle))

        img_paths.append(os.path.join(data_dir + '/IMG', left.strip()))
        steering_labels.append(float(steering_angle))

        img_paths.append(os.path.join(data_dir + '/IMG', right.strip()))
        steering_labels.append(float(steering_angle))

    return np.asarray(img_paths), np.asarray(steering_labels)


def train_test_histogram(y_train, y_valid):
    """
    Shows how train_set and validation_set are distributed in a Histogram after calling train_test_split function
    from sklearn package.
    """
    plt.figure("Train - Test Split")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=25, width=0.05, color='blue')
    axes[0].set_title("Training Set")
    axes[1].hist(y_valid, bins=25, width=0.05, color='red')
    axes[1].set_title("Validation Set")


def image_process(img):
    """
    Pre-processing Images and preparing it for use inside the neural network model (Nvidia Model).
    """
    img = img[60:135, :, :]  # Remove any area of the image that is out of region of interest.
    img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    img = cv.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian Blur to reduce image noise.
    img = cv.resize(img, (200, 66))  # Resize the image to match the input shape used in nvidia model architecture.
    img = img / 255  # Normalize the pixels intensity values.
    return img


def show_images_compare(img, img_modified):
    """
    Visualize randomly selected before and after preprocessing
    """
    plt.figure("Before - After")
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[1].imshow(img_modified)
    axes[1].set_title("Preprocessed Image")


def nvidia_model():
    """
    See the full article about nvidia model: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    Defining nvidia model with 5 Convolutional layers:
    1- 24 filters, 3x3 kernel, 2x2 strides, 66x200x3 input shape (66p height, 200p width, 3 channels)
    2- 36 filters, 5x5 kernel, 2x2 strides
    3- 48 filters, 5x5 kernel, 2x2 strides
    4+5- 64 filters, 3x3 kernel, 1x1 strides
    6- Flatten layer to prepare it for the next Dense (Fully connected) layer
    7- 3 Dense layers with 100, 50, 10 nodes respectively
    8- Dense layer with one output node (Outputs the predicted angle according to the input image)
    9- Compile the model with mse (mean squared error) loss function as it is a regression type problem
    10- Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent
        procedure to update network weights iterative based in training data.
    """
    n_model = Sequential()
    n_model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    n_model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    n_model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    n_model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    n_model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))

    n_model.add(Flatten())

    n_model.add(Dense(100, activation='elu'))

    n_model.add(Dense(50, activation='elu'))

    n_model.add(Dense(10, activation='elu'))

    n_model.add(Dense(1))

    n_model.compile(loss='mse', optimizer=Adam(lr=1e-4))

    return n_model


def zoom_img(img):
    """Returns a new 30% zoomed image copy"""
    zoom = aug.Affine(scale=(1, 1.3))
    zoomed_img = zoom.augment_image(img)
    return zoomed_img


def pan_img(img):
    """Image shifting transformation"""
    pan = aug.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
    img = pan.augment_image(img)
    return img


def img_random_brightness(img):
    """Changes the image brightness"""
    bright = aug.Multiply((0.2, 1.2))
    img = bright.augment_image(img)
    return img


def img_flip(img, steering_angle):
    """Image flipping (Mirror)"""
    img = cv.flip(img, 1)
    steering_angle = -steering_angle  # invert the steering angle
    return img, steering_angle


def random_augment(img, steering_angle):
    """Apply image augmentation functions randomly on the training_set images"""
    image = mp_img.imread(img)
    if np.random.rand() < 0.5:
        image = zoom_img(image)
    if np.random.rand() < 0.5:
        image = pan_img(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_flip(image, steering_angle)
    return image, steering_angle


def batch_generator(img_paths, steering_angles, batch_size, is_training):
    """
    The benefit of the generator is that it can create augmented images on the fly rather than
    augmenting all images and storing them using valuable memory space.
    The generator allows you to create small batches of images at a time only when the generator is actually
    called. This is much more memory efficient as data is only used when it's required rather than being store
    in memory even when it is not being used.

    yield is a keyword in Python that is used to return from a function without destroying the states of its local
    variable and when the function is called, the execution starts from the last yield statement.
    Any function that contains a yield keyword is termed as generator.
    """
    while True:

        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(img_paths) - 1)

            if is_training:
                im, steering_angle = random_augment(img_paths[random_index], steering_angles[random_index])
            else:
                im = mp_img.imread((img_paths[random_index]))
                steering_angle = steering_angles[random_index]

            im = image_process(im)
            batch_img.append(im)
            batch_steering.append(steering_angle)
        yield np.asarray(batch_img), np.asarray(batch_steering)


driving_data = load_data()  # Load the data from csv file
driving_data = balance_data(driving_data)  # Balance the steering angles


images, steering = load_images(driving_data)  # Load the images paths and steering angles

# Split 80% as a train data and 20% as validation data
train_data, valid_data, train_labels, valid_labels = train_test_split(images, steering, test_size=0.2, random_state=6)


print(f"The Number of Training Samples: {len(train_data)}\nThe Number of Validation Samples: {len(valid_data)}")
train_test_histogram(train_labels, valid_labels)

# Extra code to check if the image Pre-processing is working as intended
original_img = mp_img.imread(images[random.randint(0, len(images) - 1)])
processed_img = image_process(original_img)
show_images_compare(original_img, processed_img)

# load the nvidia model
model = nvidia_model()
model.summary()

# Train the model by calling the batch generator that returns a batch of augmented, processed images.
history = model.fit(batch_generator(train_data, train_labels, 100, 1),
                    steps_per_epoch=300,
                    epochs=20,
                    validation_data=batch_generator(valid_data, valid_labels, 100, 0),
                    validation_steps=200,
                    verbose=1,
                    shuffle=1)


plt.figure("Model Evaluation")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
model.save("model.h5")
