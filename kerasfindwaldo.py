# Visualization
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import numpy as np
import random
import os
import time

# Model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D


# Constructing the conv-net
def get_conv(input_shape=(64, 64, 3), filename=None):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', input_shape=input_shape, padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (8, 8), activation="relu", name="dense1"))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (14, 14), name="dense2", activation="sigmoid"))

    # for layer in model.layers:
    #     print(layer.input_shape, layer.output_shape)
    if filename:
        model.load_weights(filename)
    return model


model = get_conv()
model.add(Flatten())
model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

heatmodel = get_conv(input_shape=(None, None, 3), filename="Models/localize7.h5")


# Store a 64x64 image (hopefully) without Waldo in it
def store_image(path, name, x, y):
    image = cv2.imread(path)[x:x + 64, y:y + 64]
    image_PIL = Image.open(path).crop((x, y, x + 64, y + 64))
    image_PIL.save('Data/Predictions/' + name + str(x) + '-' + str(y) + '.png')


# Locate waldo in image
def locate(img, filepath="Data/Raw/Train/"):
    num_sub_img = 100
    data = cv2.cvtColor(cv2.imread(filepath + img), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cv2.imread(filepath + img), cv2.COLOR_BGR2GRAY)
    coloured_heat = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    heatmap = heatmodel.predict(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))
    plt.imshow(heatmap[0, :, :, 0])
    plt.title("Heatmap")
    plt.show()

    # Show 95 certainty heatmap and grayscale->color
    # plt.imshow(heatmap[0, :, :, 0] > 0.95, cmap="gray")
    # plt.title("Heatmap")
    # plt.show()
    # coloured_heat = color_gray(data, heatmap, gray).astype(np.uint8)
    # Image.fromarray(coloured_heat).show()

    xx, yy = np.meshgrid(np.arange(heatmap.shape[2]), np.arange(heatmap.shape[1]))
    x = (xx[heatmap[0, :, :, 0] > 0.99])
    y = (yy[heatmap[0, :, :, 0] > 0.99])

    # Store a subset of 64x64 images of this image to train next model on
    # chosen_sub_images = random.sample([[x[num], y[num]] for num in range(len(x))], min(num_sub_img, len(x)))
    # for i, j in chosen_sub_images:
    #     store_image('Data/Raw/Train/' + img, img, int(i*3),int(j*3)))

    for i, j in zip(x, y):
        y_pos = j * 3
        x_pos = i * 3
        cv2.rectangle(data, (x_pos, y_pos), (x_pos + 64, y_pos + 64), (0, 0, 255), 5)

    coloured_heat = coloured_heat.astype(np.uint8)
    # Image.fromarray(coloured_heat).show()
    # if random.randint(0, 10) < 1:
    #     store_image('Data/Raw/Train/'+img, img, i, j)
    return data


def color_gray(image, heatmap, gray):
    height, width = image.shape[:2]
    heat_height, heat_width = heatmap.shape[1:3]
    return np.array([[[int(
        int(gray[y, x]) + heatmap[0, int(y * heat_height / height), int(x * heat_width / width), 0] * (
                int(image[y, x, c]) - gray[y, x])) for c in range(3)] for x in range(width)] for y in
        range(height)])


# Predict all test images
for img in os.listdir("Data/Raw/Test/"):
    print(img)
    try:
        annotated = locate(img, filepath="Data/Raw/Test/")
        Image.fromarray(annotated).show()
        plt.title("Augmented")
        plt.imshow(annotated)
        plt.show()
    except Exception as e:
        print('exception', e)

# Predict a specific image
annotated = locate('14.jpg')
Image.fromarray(annotated).show()
print(annotated.shape[:2])
plt.title("Augmented")
plt.imshow(annotated)
plt.show()
